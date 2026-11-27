import { randomUUID } from 'node:crypto'
import { link, mkdir, open, readFile, realpath, unlink } from 'node:fs/promises'
import { dirname, join, relative } from 'node:path'
import { createEmbedding, embeddingUrl, probeRemoteEmbedding, type EmbeddingOptions } from './embedding.js'
import { downloadLocalModel, isLocalModel, localModels, type LocalModelId } from './model-files.js'

type RemoteCapabilities = Awaited<ReturnType<typeof probeRemoteEmbedding>>
type LocalSelection = { source: 'local'; modelId: LocalModelId }
type RemoteSelection = { source: 'remote'; modelId: string; baseUrl: string; apiKey: string }
type Selection = LocalSelection | RemoteSelection
type Binding = (LocalSelection | (RemoteSelection & RemoteCapabilities)) & { version: 1 }

interface ConfiguredEmbedding {
  options: EmbeddingOptions
  managed: boolean
  credential?: Pick<RemoteSelection, 'apiKey' | 'baseUrl' | 'modelId'>
}

interface BindingStatus {
  status: 'unconfigured' | 'downloading' | 'connecting' | 'initializing' | 'active' | 'failed'
  locked?: boolean
  source?: Selection['source']
  modelId?: string
  baseUrl?: string
  dimensions?: number
  contextLength?: number
  contextSource?: RemoteCapabilities['contextSource']
  supportsMedia?: boolean
  progress?: { receivedBytes: number; totalBytes: number; completedFiles: number; totalFiles: number; currentFile?: string }
  error?: string
}

export class BindingRequestError extends Error {
  constructor(readonly status: 400 | 409, message: string) {
    super(message)
  }
}

function parseSelection(input: unknown): Selection {
  const invalid = (message: string) => new BindingRequestError(400, message)
  if (!input || typeof input !== 'object' || !('source' in input) || !('modelId' in input) || typeof input.modelId !== 'string') {
    throw invalid('请选择嵌入模型来源并填写模型 ID')
  }
  const modelId = input.modelId.trim()
  if (input.source === 'local' && isLocalModel(modelId)) {
    return { source: 'local', modelId }
  }
  if (input.source !== 'remote' || !('baseUrl' in input) || typeof input.baseUrl !== 'string' || input.baseUrl.length > 2048 ||
      !('apiKey' in input) || typeof input.apiKey !== 'string' || !/^[\x21-\x7e]{0,4096}$/u.test(input.apiKey.trim()) ||
      !modelId || modelId.length > 256 || /[\u0000-\u001f\u007f]/u.test(modelId)) {
    throw invalid('请选择支持的本机模型，或填写有效的远程服务地址、密钥和模型 ID')
  }
  try {
    return { source: 'remote', modelId, baseUrl: embeddingUrl(input.baseUrl.trim()), apiKey: input.apiKey.trim() }
  } catch {
    throw invalid('远程服务地址必须是无凭据和查询串的 HTTP 或 HTTPS 地址')
  }
}

function bindingView(binding: Binding): Omit<BindingStatus, 'status'> {
  if (binding.source === 'local') {
    const { dimensions, maxInputTokens } = localModels[binding.modelId]
    return { source: 'local', modelId: binding.modelId, dimensions, contextLength: maxInputTokens,
      contextSource: 'model', supportsMedia: false, locked: true }
  }
  return { source: 'remote', modelId: binding.modelId, baseUrl: binding.baseUrl, dimensions: binding.dimensions,
    contextLength: binding.maxInputTokens, contextSource: binding.contextSource, supportsMedia: binding.supportsMedia, locked: true }
}

function configuredEmbedding(binding: Binding): ConfiguredEmbedding {
  if (binding.source === 'local') {
    return { managed: true, options: { backend: 'onnx', model: binding.modelId, download: false } }
  }
  const { baseUrl, apiKey, modelId, model, dimensions, maxInputTokens } = binding
  return { managed: true, options: { backend: 'openai', baseUrl, modelId, model, dimensions, maxInputTokens,
    requestDimensions: false }, credential: { baseUrl, modelId, apiKey } }
}

async function assertWithin(path: string) {
  if (relative(path, await realpath(path)) !== '') {
    throw new Error('记忆绑定路径解析到配置档的其他位置')
  }
}

async function readBinding(path: string): Promise<Binding | undefined> {
  try {
    await assertWithin(path)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return undefined
    }
    throw error
  }
  const value = JSON.parse(await readFile(path, 'utf8')) as Binding
  if (value?.version !== 1) {
    throw new Error('无法识别记忆绑定版本')
  }
  const selection = parseSelection(value)
  if (selection.source === 'local') {
    return { ...selection, version: 1 }
  }
  if (value.source !== 'remote' || !['reported', 'model', 'fallback'].includes(value.contextSource) ||
      !Number.isSafeInteger(value.dimensions) || value.dimensions <= 0 ||
      !Number.isSafeInteger(value.maxInputTokens) || value.maxInputTokens < (value.supportsMedia ? 64 : 32) ||
      typeof value.supportsMedia !== 'boolean' || typeof value.supportsVideo !== 'boolean' || typeof value.model !== 'string') {
    throw new Error('记忆绑定缺少有效的远程模型参数')
  }
  return { ...value, ...selection }
}

async function saveBinding(path: string, binding: Binding) {
  const directory = dirname(path)
  await mkdir(directory, { recursive: true, mode: 0o700 })
  await assertWithin(directory)
  const temporary = join(directory, `.memory-binding-${randomUUID()}.tmp`)
  const file = await open(temporary, 'wx', 0o600)
  try {
    try {
      await file.writeFile(JSON.stringify(binding) + '\n')
      await file.sync()
    } finally {
      await file.close()
    }
    await link(temporary, path)
  } finally {
    await unlink(temporary).catch(error => {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw error
      }
    })
  }
}

/** 管理每个配置档的绑定及激活任务，关闭设置页时仍可完成下载和初始化。 */
export class MemoryBinding {
  private readonly path: string
  private readonly stop = new AbortController()
  private view: BindingStatus = { status: 'unconfigured' }
  private job?: Promise<void>

  constructor(private readonly profile: string,
    private readonly activate: (selected: ConfiguredEmbedding, signal: AbortSignal) => Promise<void>,
    private readonly report: (message: string) => void) {
    this.path = join(profile, 'data', 'dsh-prism', 'memory-binding.json')
  }

  status(): BindingStatus {
    return this.view
  }

  async restore(legacy?: EmbeddingOptions) {
    try {
      const binding = await readBinding(this.path)
      if (binding) {
        this.track(this.open(binding))
      } else if (legacy?.backend === 'openai') {
        this.view = { status: 'initializing', locked: true, source: 'remote',
          modelId: legacy.modelId ?? legacy.model, baseUrl: embeddingUrl(legacy.baseUrl),
          dimensions: legacy.dimensions, contextLength: legacy.maxInputTokens }
        this.track(this.enable({ options: legacy, managed: false }))
      } else if (legacy) {
        const modelId = legacy.model ?? 'bge-large-zh-v1.5'
        this.view = { ...bindingView({ source: 'local', modelId, version: 1 }), status: 'initializing' }
        this.track(this.enable({ options: legacy, managed: false }))
      }
    } catch {
      const error = '无法读取记忆绑定或已有配置，请检查配置档并重启 DSH'
      this.view = { ...this.view, status: 'failed', locked: true, error }
      this.report(error)
    }
  }

  start(input: unknown): BindingStatus {
    if (this.stop.signal.aborted || this.view.locked || this.job) {
      throw new BindingRequestError(409, '记忆模型已绑定，或正在下载或验证')
    }
    const selection = parseSelection(input)
    this.view = { status: selection.source === 'local' ? 'downloading' : 'connecting', source: selection.source,
      modelId: selection.modelId, ...(selection.source === 'remote' && { baseUrl: selection.baseUrl }) }
    this.track(this.bind(selection))
    return this.view
  }

  private track(job: Promise<void>) {
    this.job = job.finally(() => { this.job = undefined })
  }

  private open(binding: Binding) {
    this.view = { ...bindingView(binding), status: 'initializing' }
    return this.enable(configuredEmbedding(binding))
  }

  private async enable(selected: ConfiguredEmbedding) {
    const signal = this.stop.signal
    try {
      await this.activate(selected, signal)
      if (!signal.aborted) {
        this.view = { ...this.view, status: 'active' }
      }
    } catch {
      if (!signal.aborted) {
        const error = '已绑定模型无法启动，请检查模型文件或远程服务后重启 DSH'
        this.view = { ...this.view, status: 'failed', locked: true, error }
        this.report(error)
      }
    }
  }

  private async bind(selection: Selection) {
    const signal = this.stop.signal
    let failureMessage = selection.source === 'local' ? '模型下载或校验失败，请检查网络、磁盘空间及 ONNX 运行库后重试' :
      '向量服务验证失败，请检查服务地址、密钥及模型 ID，并确认服务可返回向量'
    try {
      const existing = await readBinding(this.path)
      if (existing) {
        await this.open(existing)
        return
      }
      let binding: Binding
      if (selection.source === 'local') {
        await downloadLocalModel(this.profile, selection.modelId, signal, ({ downloadedBytes, ...progress }) => {
          this.view = { ...this.view, progress: { ...progress, receivedBytes: downloadedBytes } }
        })
        failureMessage = '模型文件已下载，但本机推理验证失败，请检查 ONNX 运行环境后重试'
        binding = { ...selection, version: 1 }
      } else {
        binding = { ...selection, ...await probeRemoteEmbedding(selection, signal), version: 1 }
        failureMessage = '远程服务已返回向量，但无法准备模型分词器，请检查网络或服务配置后重试'
      }
      const { options, credential } = configuredEmbedding(binding)
      const embedding = await createEmbedding(this.profile, options.backend === 'openai' && credential
        ? { ...options, apiKey: credential.apiKey } : options, signal)
      try {
        if (binding.source === 'local') {
          await embedding.embed([{ text: 'Prism memory connection test.' }], 'query')
        }
      } finally {
        await embedding.close()
      }
      signal.throwIfAborted()
      failureMessage = '无法保存记忆绑定，请检查配置档目录的写入权限和剩余空间后重试'
      await saveBinding(this.path, binding)
      await this.open(binding)
    } catch (error) {
      if (signal.aborted) {
        return
      }
      try {
        const current = await readBinding(this.path)
        if (current) {
          await this.open(current)
          return
        }
      } catch {
        this.view = { ...this.view, status: 'failed', locked: true, error: '无法读取记忆绑定，请检查配置档并重启 DSH' }
        return
      }
      const detail = error instanceof Error ? /HTTP [45][0-9]{2}/u.exec(error.message)?.[0] : undefined
      this.view = { ...this.view, status: 'failed', error: detail ? `${failureMessage}（${detail}）` : failureMessage }
    }
  }

  async close() {
    this.stop.abort()
    await this.job
  }
}

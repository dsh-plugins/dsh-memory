import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { Tokenizer } from '@huggingface/tokenizers'
import { bge, isLocalModel, loadLocalRuntime, localModels, modelWeights, qwenVision, tokenizerFiles, type LocalModelId } from './model-files.js'
import type { MediaUpload } from './contracts.js'
import { videoSamplingRevision, type VideoSegment } from './video.js'

const qwenInstruction = 'Instruct: Given a query, retrieve relevant memories.\nQuery: '
const maxRemoteDimensions = 32768
const unknownContext = 512

/** 本机加载绑定模型的 ONNX 文件和词表，显式路径相对于配置档。 */
export interface LocalEmbeddingOptions {
  backend?: 'onnx'
  model?: LocalModelId
  modelPath?: string
  tokenizerDirectory?: string
  download?: boolean
  chunkTokens?: number
}

type RemoteModel = LocalModelId | typeof qwenVision | 'custom'

/** 远程模型使用 OpenAI 兼容请求，Qwen 的媒体请求使用 vLLM 消息扩展。 */
export interface RemoteEmbeddingOptions {
  backend: 'openai'
  model: RemoteModel
  baseUrl: string
  apiKey?: string
  apiKeyEnv?: string
  modelId?: string
  revision?: string
  tokenizerDirectory?: string
  maxInputTokens?: number
  dimensions?: number
  requestDimensions?: boolean
  chunkTokens?: number
}

/** npm Bundle 挂载时选择的向量后端。 */
export type EmbeddingOptions = LocalEmbeddingOptions | RemoteEmbeddingOptions

/** 文本及至多一个需要独立向量化的媒体输入。 */
export interface EmbeddingInput {
  text: string
  media?: MediaUpload
  video?: VideoSegment
}

/** 当前索引的模型能力。每条结果为单位向量。 */
export interface Embedding {
  readonly fingerprint: string
  readonly dimensions: number
  readonly supportsMedia: boolean
  readonly supportsVideo: boolean
  readonly chunkTokens: number
  countTokens(text: string): number
  embed(input: readonly EmbeddingInput[], purpose: 'passage' | 'query'): Promise<Float32Array[]>
  close(): Promise<void>
}

function optionsInteger(value: number | undefined, fallback: number, min: number, max: number, name: string): number {
  const next = value ?? fallback
  if (!Number.isSafeInteger(next) || next < min || next > max) {
    throw new Error(`记忆向量配置 ${name} 必须在 ${min} 至 ${max} 之间`)
  }
  return next
}

function fingerprint(...parts: unknown[]) {
  return createHash('sha256').update(JSON.stringify(['prism-memory-embedding-1', ...parts])).digest('hex')
}

function normalize(vector: ArrayLike<number>, dimensions: number): Float32Array {
  if (vector.length !== dimensions) {
    throw new Error('向量服务返回的维度与配置不一致')
  }
  const values = Array.from(vector)
  if (values.some(value => !Number.isFinite(value))) {
    throw new Error('向量服务返回了无效的数值')
  }
  const norm = Math.hypot(...values)
  if (norm === 0 || !Number.isFinite(norm)) {
    throw new Error('向量服务返回了无法归一化的向量')
  }
  return Float32Array.from(values, value => value / norm)
}

async function loadTokenizer(profile: string, directory: string | undefined, model: LocalModelId | typeof qwenVision,
  signal: AbortSignal, download = true) {
  const paths = await tokenizerFiles(profile, directory && resolve(profile, directory), model, signal, download)
  const [data, config] = await Promise.all(paths.map(path => readFile(path, { encoding: 'utf8', signal })))
  return { tokenizer: new Tokenizer(JSON.parse(data), JSON.parse(config)), revision: fingerprint(data, config) }
}

async function local(profile: string, options: LocalEmbeddingOptions, signal: AbortSignal): Promise<Embedding> {
  const model = options.model ?? bge
  if (!isLocalModel(model)) {
    throw new Error('请选择受支持的本机记忆向量模型')
  }
  const { dimensions, pooling, queryPrefix, maxInputTokens } = localModels[model]
  const ort = await loadLocalRuntime()
  const [lexicon, weights] = await Promise.all([
    loadTokenizer(profile, options.tokenizerDirectory, model, signal, options.download),
    modelWeights(profile, model, options.modelPath && resolve(profile, options.modelPath), signal, options.download),
  ])
  const chunkTokens = optionsInteger(options.chunkTokens, maxInputTokens === 512 ? 384 : 1024, 32, maxInputTokens, 'chunkTokens')
  const session = await ort.InferenceSession.create(weights.path, { executionProviders: ['cpu'] })
  const outputName = session.outputNames.includes('last_hidden_state') ? 'last_hidden_state' : 'sentence_embedding'
  if (!session.outputNames.includes(outputName)) {
    await session.release()
    throw new Error('本机 ONNX 模型缺少文本向量输出')
  }
  const encode = (text: string) => lexicon.tokenizer.encode(text, { add_special_tokens: true, return_token_type_ids: true })
  return {
    fingerprint: fingerprint('onnx', model, weights.revision, lexicon.revision, chunkTokens),
    dimensions,
    supportsMedia: false,
    supportsVideo: false,
    chunkTokens,
    countTokens: text => encode(text).ids.length,
    async embed(inputs, purpose) {
      const vectors: Float32Array[] = []
      for (const input of inputs) {
        signal.throwIfAborted()
        if (input.media || input.video) {
          throw new Error('本机模型仅支持文本向量化')
        }
        const { ids, attention_mask, token_type_ids }: { ids: number[]; attention_mask: number[]; token_type_ids: number[] } =
          encode((purpose === 'query' ? queryPrefix : '') + input.text)
        if (!ids.length || ids.length > maxInputTokens) {
          throw new Error(`记忆输入须在 1 至 ${maxInputTokens} token 之间`)
        }
        const available: Record<string, number[]> = {
          input_ids: ids,
          attention_mask,
          token_type_ids: token_type_ids.length === ids.length ? token_type_ids : ids.map(() => 0),
          position_ids: ids.map((_id, index) => index),
        }
        const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {}
        for (const field of session.inputMetadata) {
          const { name } = field
          if (!field.isTensor) {
            throw new Error(`本机 ONNX 模型要求不受支持的输入 ${name}`)
          }
          if (model === 'Qwen3-Embedding-0.6B' && /^past_key_values\.\d+\.(key|value)$/u.test(name)) {
            const [, heads, , channels] = field.shape
            if (field.type !== 'float32' || field.shape.length !== 4 ||
                typeof heads !== 'number' || heads < 1 || typeof channels !== 'number' || channels < 1) {
              throw new Error(`本机 ONNX 模型的缓存输入形状无效 ${name}`)
            }
            // 每段文本独立编码，解码器导出的历史缓存始终为空。
            feeds[name] = new ort.Tensor('float32', new Float32Array(0), [1, heads, 0, channels])
            continue
          }
          if (!Object.hasOwn(available, name) || !['int32', 'int64'].includes(field.type)) {
            throw new Error(`本机 ONNX 模型要求不受支持的输入 ${name}`)
          }
          feeds[name] = field.type === 'int32'
            ? new ort.Tensor('int32', Int32Array.from(available[name]), [1, ids.length])
            : new ort.Tensor('int64', BigInt64Array.from(available[name], BigInt), [1, ids.length])
        }
        const out = await session.run(feeds, [outputName])
        const output = out[outputName]
        if (!output || !(output.data instanceof Float32Array) || output.dims[0] !== 1 || output.dims.at(-1) !== dimensions ||
            (output.dims.length !== 2 && (output.dims.length !== 3 || output.dims[1] !== ids.length))) {
          throw new Error(`本机 ONNX 模型未返回受支持的 ${dimensions} 维输出`)
        }
        const token = output.dims.length === 2 || pooling === 'first' ? 0 : attention_mask.lastIndexOf(1)
        if (token < 0) {
          throw new Error('本机 ONNX 模型缺少可提取向量的 token')
        }
        vectors.push(normalize(output.data.subarray(token * dimensions, (token + 1) * dimensions), dimensions))
      }
      return vectors
    },
    close: () => session.release(),
  }
}

/** 为绑定校验和远程请求生成同一个无凭据地址。 */
export function embeddingUrl(base: string) {
  const endpoint = new URL(base)
  if (!['https:', 'http:'].includes(endpoint.protocol) || endpoint.username || endpoint.password || endpoint.search || endpoint.hash) {
    throw new Error('向量服务地址必须是无凭据和查询串的 HTTP 或 HTTPS 地址')
  }
  const path = endpoint.pathname.replace(/\/+$/u, '')
  endpoint.pathname = path.endsWith('/embeddings') ? path : (path || '/v1') + '/embeddings'
  return endpoint.toString()
}

/** 只持久化可重建模型的公开选项，密钥值始终由运行进程提供。 */
export function savedEmbeddingOptions(options: EmbeddingOptions): string {
  let values: Record<string, unknown>
  if (options.backend === 'openai') {
    const { model, modelId, revision, baseUrl, apiKeyEnv, tokenizerDirectory, maxInputTokens,
      dimensions, requestDimensions, chunkTokens } = options
    values = { backend: options.backend, model, modelId, revision, baseUrl: embeddingUrl(baseUrl), apiKeyEnv, tokenizerDirectory,
      maxInputTokens, dimensions, requestDimensions, chunkTokens, ...(model === qwenVision && { videoRevision: videoSamplingRevision }) }
  } else {
    const { backend = 'onnx', model, modelPath, tokenizerDirectory, download, chunkTokens } = options
    values = { backend, model, modelPath, tokenizerDirectory, download, chunkTokens }
  }
  if (Object.values(values).some(value => value !== undefined && !['string', 'number', 'boolean'].includes(typeof value))) {
    throw new Error('记忆向量配置包含无法保存的值')
  }
  return JSON.stringify(values)
}

function responseVectors(body: unknown, count: number, dimensions?: number) {
  if (!body || typeof body !== 'object' || !('data' in body) || !Array.isArray(body.data) || body.data.length !== count) {
    throw new Error('向量服务返回的条目数量不匹配')
  }
  const vectors: Float32Array[] = new Array(count)
  for (const row of body.data) {
    if (!row || typeof row !== 'object' || !Number.isSafeInteger(row.index) ||
        row.index < 0 || row.index >= count || vectors[row.index] || !Array.isArray(row.embedding) ||
        row.embedding.length === 0 || row.embedding.length > maxRemoteDimensions) {
      throw new Error('向量服务返回了无效的条目')
    }
    const vector: number[] = row.embedding
    dimensions ??= vector.length
    vectors[row.index] = normalize(vector, dimensions)
  }
  return vectors
}

function remoteHeaders(key: string | undefined): Record<string, string> {
  return { ...(key ? { authorization: 'Bearer ' + key } : {}) }
}

async function requestEmbedding(endpoint: string, model: string, key: string | undefined, signal: AbortSignal,
  input: Record<string, unknown>, count: number, dimensions?: number, requestDimensions?: boolean): Promise<Float32Array[]> {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...remoteHeaders(key) },
    body: JSON.stringify({ model, encoding_format: 'float', ...(requestDimensions && { dimensions }), ...input }),
    signal: AbortSignal.any([signal, AbortSignal.timeout(90_000)]),
    redirect: 'error',
  })
  if (!response.ok) {
    await response.body?.cancel()
    throw new Error(`向量服务请求失败，HTTP ${response.status}`)
  }
  const body = await response.json().catch(cause => {
    throw new Error('向量服务须返回 JSON 内容', { cause })
  })
  return responseVectors(body, count, dimensions)
}

function modelKind(modelId: string): RemoteModel {
  const name = modelId.split('/').at(-1)?.toLowerCase()
  if (name === qwenVision.toLowerCase()) {
    return qwenVision
  }
  return (Object.keys(localModels) as LocalModelId[]).find(model => model.toLowerCase() === name) ?? 'custom'
}

function maxModelTokens(model: RemoteModel): number | undefined {
  if (model === 'custom') {
    return undefined
  }
  return model === qwenVision ? 32768 : localModels[model].maxInputTokens
}

function reportedContext(body: unknown, model: string): number | undefined {
  if (!body || typeof body !== 'object') {
    return undefined
  }
  if ('data' in body) {
    const entry = Array.isArray(body.data)
      ? body.data.find(candidate => candidate && typeof candidate === 'object' && candidate.id === model)
      : body.data
    const tokens = reportedContext(entry, model)
    if (tokens !== undefined) {
      return tokens
    }
  }
  const meta = body as Record<string, unknown>
  const tokens = ['max_input_tokens', 'max_model_len', 'max_context_length', 'context_length', 'context_window', 'model_max_length']
    .map(key => meta[key])
    .filter((value): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value >= 32 && value <= 1_048_576)
  return tokens.length ? Math.min(...tokens) : undefined
}

/** 远程连接探测报告实际返回的维度，并标明上下文长度的来源。 */
export interface RemoteEmbeddingProbe {
  model: RemoteModel
  dimensions: number
  maxInputTokens: number
  contextSource: 'reported' | 'model' | 'fallback'
  supportsMedia: boolean
  supportsVideo: boolean
}

/** 使用固定的测试短句检查接口和凭据，只返回可公开显示的模型信息。 */
export async function probeRemoteEmbedding(input: { baseUrl: string; apiKey?: string; modelId: string },
  signal: AbortSignal): Promise<RemoteEmbeddingProbe> {
  const endpoint = embeddingUrl(input.baseUrl)
  const modelId = input.modelId.trim()
  if (!modelId) {
    throw new Error('远程向量模型名称不能为空')
  }
  const probeSignal = AbortSignal.any([signal, AbortSignal.timeout(25_000)])
  const [vector] = await requestEmbedding(endpoint, modelId, input.apiKey, probeSignal,
    { input: ['Prism memory connection test.'] }, 1)
  const model = modelKind(modelId)
  const metadataUrl = endpoint.replace(/\/embeddings$/u, '/models')
  let reported: number | undefined
  for (const metadata of [metadataUrl + '/' + encodeURIComponent(modelId), metadataUrl]) {
    try {
      const response = await fetch(metadata, {
        headers: remoteHeaders(input.apiKey),
        signal: AbortSignal.any([signal, AbortSignal.timeout(4_000)]),
        redirect: 'error',
      })
      if (response.ok) {
        reported = reportedContext(await response.json(), modelId)
        if (reported !== undefined) {
          break
        }
      } else {
        await response.body?.cancel()
      }
    } catch {
      signal.throwIfAborted()
      // 模型列表属于可选接口。提供方未报告长度时使用可用的模型规格或保守预算。
      continue
    }
  }
  signal.throwIfAborted()
  const specified = maxModelTokens(model)
  const maxInputTokens = reported === undefined ? specified ?? unknownContext : Math.min(reported, specified ?? reported)
  if (model === qwenVision && maxInputTokens < 64) {
    throw new Error('远程多模态模型至少须支持 64 token')
  }
  return {
    model,
    dimensions: vector.length,
    maxInputTokens,
    contextSource: reported !== undefined ? 'reported' : specified !== undefined ? 'model' : 'fallback',
    supportsMedia: model === qwenVision,
    supportsVideo: model === qwenVision,
  }
}

async function remote(profile: string, options: RemoteEmbeddingOptions, signal: AbortSignal,
  videoRevision: string | null): Promise<Embedding> {
  const definition = options.model
  if (definition !== qwenVision && definition !== 'custom' && !isLocalModel(definition)) {
    throw new Error('请选择受支持的远程记忆向量模型')
  }
  const vision = definition === qwenVision
  const endpoint = embeddingUrl(options.baseUrl)
  const model = (options.modelId ?? (definition !== 'custom' ? definition : '')).trim()
  const key = options.apiKey ?? (options.apiKeyEnv && process.env[options.apiKeyEnv])
  if (!model || (options.apiKeyEnv && !key)) {
    throw new Error('远程向量模型名称或配置的 API 密钥环境变量为空')
  }
  const modelLimit = maxModelTokens(definition)
  const maxInputTokens = optionsInteger(options.maxInputTokens, modelLimit ?? unknownContext, vision ? 64 : 32,
    modelLimit ?? 1_048_576, 'maxInputTokens')
  const textBudget = maxInputTokens - (vision ? 32 : 0)
  const dimensions = optionsInteger(options.dimensions, vision ? 4096 : 1024, 1, maxRemoteDimensions, 'dimensions')
  const chunkTokens = optionsInteger(options.chunkTokens, Math.min(textBudget, vision ? 1024 : 384),
    32, textBudget, 'chunkTokens')
  const lexicon = definition === 'custom' ? undefined : await loadTokenizer(profile, options.tokenizerDirectory, definition, signal)
  const countTokens = (text: string) => lexicon?.tokenizer.encode(text, { add_special_tokens: true }).ids.length ??
    Buffer.byteLength(text, 'utf8')
  const instruction = vision ? qwenInstruction : definition === 'custom' ? '' : localModels[definition].queryPrefix
  const submit = (input: Record<string, unknown>, count: number) => requestEmbedding(endpoint, model, key, signal, input, count, dimensions,
    options.requestDimensions ?? (vision && options.dimensions !== undefined))
  return {
    fingerprint: fingerprint('openai', definition, endpoint, model, options.revision, lexicon?.revision ?? 'utf8-byte-budget', dimensions,
      textBudget, chunkTokens, ...(vision && videoRevision !== null ? [videoRevision] : [])),
    dimensions,
    supportsMedia: vision,
    supportsVideo: vision && videoRevision === videoSamplingRevision,
    chunkTokens,
    countTokens,
    async embed(inputs, purpose) {
      const vectors: Float32Array[] = []
      let texts: string[] = []
      async function flush() {
        if (texts.length > 0) {
          const values = await submit({ input: texts }, texts.length)
          for (const vector of values) {
            vectors.push(vector)
          }
          texts = []
        }
      }
      for (const entry of inputs) {
        const text = (purpose === 'query' ? instruction : '') + entry.text
        const textTokens = countTokens(text)
        if (textTokens > textBudget) {
          throw new Error('记忆输入超过远程向量模型的 token 上限')
        }
        // 按 Qwen 的 32 像素空间合并单位估算 512 像素帧，给视觉边界留出余量。
        if (entry.video && textTokens + entry.video.frameCount * 256 + 128 > textBudget) {
          throw new Error('视频分片及文本预计超过 Qwen 的输入上限，请提高 maxInputTokens 或缩短查询')
        }
        if (!entry.media && !entry.video) {
          texts.push(text)
          if (texts.length === 16) {
            await flush()
          }
          continue
        }
        if (!vision) {
          throw new Error('当前远程向量模型仅支持文本向量化')
        }
        await flush()
        const file = entry.video
          ? { type: 'video_url', video_url: { url: entry.video.dataUrl } }
          : { type: 'image_url', image_url: { url: imageUrl(entry.media) } }
        const [vector] = await submit({
          messages: [{ role: 'user', content: [...(text.trim() ? [{ type: 'text', text }] : []), file] }],
        }, 1)
        vectors.push(vector)
      }
      await flush()
      return vectors
    },
    async close() {},
  }
}

function imageUrl(media: MediaUpload | undefined) {
  if (!media?.mimeType.startsWith('image/')) {
    throw new Error('视频需要按时间采样后再提交向量化')
  }
  return 'data:' + media.mimeType + ';base64,' + Buffer.from(media.bytes).toString('base64')
}

/** 初始化当前 profile 的向量后端，默认按需准备本机 CPU 模型文件。 */
export async function createEmbedding(profile: string, options: EmbeddingOptions, signal: AbortSignal,
  videoRevision: string | null = videoSamplingRevision): Promise<Embedding> {
  if (options.backend !== undefined && options.backend !== 'openai' && options.backend !== 'onnx') {
    throw new Error('记忆向量后端必须为 onnx 或 openai')
  }
  return options.backend === 'openai'
    ? remote(profile, options, signal, videoRevision)
    : local(profile, options, signal)
}

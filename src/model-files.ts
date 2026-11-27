import { createHash, randomUUID } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import { link, mkdir, stat, unlink } from 'node:fs/promises'
import { basename, join } from 'node:path'
import { Readable, Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream } from 'node:stream/web'

export const bge = 'bge-large-zh-v1.5'
export const qwenVision = 'Qwen3-VL-Embedding-8B'

interface Artifact {
  filename: string
  size: number
  digest: string
}

interface ModelSource {
  snapshot: string
  tokenizer: readonly [Artifact, Artifact]
}

interface LocalModel extends ModelSource {
  weights: Artifact
  dimensions: number
  maxInputTokens: number
  downloadBytes: number
  pooling: 'first' | 'last'
  queryPrefix: string
}

function localModel(options: Omit<LocalModel, 'downloadBytes'>): LocalModel {
  return { ...options, downloadBytes: options.weights.size + options.tokenizer[0].size + options.tokenizer[1].size }
}

/** 本机模型使用固定修订和摘要，保证已绑定的模型可复现。 */
export const localModels = {
  'bge-m3': localModel({
    snapshot: 'Xenova/bge-m3/resolve/4de13258303883538bd53b696b452bf8099f0858',
    tokenizer: [
      { filename: 'tokenizer.json', size: 17082821, digest: '6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790' },
      { filename: 'tokenizer_config.json', size: 1173, digest: '328a00a9a560aadcf2a3064f917517359eb3cc26' },
    ],
    weights: { filename: 'onnx/model_int8.onnx', size: 568456694, digest: 'a206e10e995aa2a833924bcd725ba5dd6c3425cd34bac3cf2b5677cd2a1c51d6' },
    dimensions: 1024,
    maxInputTokens: 8192,
    pooling: 'first',
    queryPrefix: '',
  }),
  [bge]: localModel({
    snapshot: 'Xenova/bge-large-zh-v1.5/resolve/a48549b3259a6165364f226599cd91f39923d5d5',
    tokenizer: [
      { filename: 'tokenizer.json', size: 439124, digest: '15df791881f91e7963c9e0d704de2f2037baa002' },
      { filename: 'tokenizer_config.json', size: 394, digest: '42046bd361a033a24264dc41de1fe6cafd68072f' },
    ],
    weights: { filename: 'onnx/model_quantized.onnx', size: 327363707, digest: '49277b15e898b5cd50f87f13f677609a9cdde71d24fa8817477ca50076bf13ef' },
    dimensions: 1024,
    maxInputTokens: 512,
    pooling: 'first',
    queryPrefix: '为这个句子生成表示以用于检索相关文章：',
  }),
  'bge-large-en-v1.5': localModel({
    snapshot: 'Xenova/bge-large-en-v1.5/resolve/dfeef6070b90658e1b391a6940efdb0925c1de6f',
    tokenizer: [
      { filename: 'tokenizer.json', size: 711396, digest: '688882a79f44442ddc1f60d70334a7ff5df0fb47' },
      { filename: 'tokenizer_config.json', size: 366, digest: '37fca74771bc76a8e01178ce3a6055a0995f8093' },
    ],
    weights: { filename: 'onnx/model_quantized.onnx', size: 336983162, digest: '4842b56e233be1cc74770f57f63b1ebb6cf357cca3dd73fcdec35c019f8a5d6e' },
    dimensions: 1024,
    maxInputTokens: 512,
    pooling: 'first',
    queryPrefix: 'Represent this sentence for searching relevant passages: ',
  }),
  'Qwen3-Embedding-0.6B': localModel({
    snapshot: 'onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/c25a394dd583836952667c12f008335071b3f43d',
    tokenizer: [
      { filename: 'tokenizer.json', size: 11423705, digest: 'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a' },
      { filename: 'tokenizer_config.json', size: 9731, digest: '19d096ecc22dde6d41d64fc64b78cebe12697fbe' },
    ],
    weights: { filename: 'onnx/model_int8.onnx', size: 613527539, digest: '6d0ea863f78b4a84afa3c7fcba1ec341572b5e28121aef77b7092b1dfdf679c7' },
    dimensions: 1024,
    maxInputTokens: 32768,
    pooling: 'last',
    queryPrefix: 'Instruct: Given a query, retrieve relevant memories.\nQuery: ',
  }),
}

/** 供设置页和配置档共同使用的本机模型 ID。 */
export type LocalModelId = keyof typeof localModels

const vision: ModelSource = {
  snapshot: 'Qwen/Qwen3-VL-Embedding-8B/resolve/2c4565515e0f265c6511776e7193b22c0968ddc7',
  tokenizer: [
    { filename: 'tokenizer.json', size: 11423705, digest: 'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a' },
    { filename: 'tokenizer_config.json', size: 5404, digest: '1d4fba2de0e94e8f97e08927cfb7d084160c2880' },
  ],
}

/** 进度按实际字节计算，现有的已校验文件计为完成。 */
export interface ModelDownloadProgress {
  downloadedBytes: number
  totalBytes: number
  completedFiles: number
  totalFiles: number
  currentFile?: string
}

/** 验证来自配置或请求的模型 ID。 */
export function isLocalModel(model: unknown): model is LocalModelId {
  return typeof model === 'string' && Object.hasOwn(localModels, model)
}

/** 提前确认当前平台可以加载本机推理库。 */
export async function loadLocalRuntime() {
  return import('onnxruntime-node').catch(cause => {
    throw new Error('当前平台无法加载本机向量化所需的 onnxruntime-node', { cause })
  })
}

function cacheDirectory(profile: string, model: LocalModelId | typeof qwenVision) {
  return join(profile, 'cache', 'dsh-prism', 'models', model)
}

function digestOf(asset: Artifact) {
  return asset.digest.length === 64 ? createHash('sha256') : createHash('sha1').update(`blob ${asset.size}\0`)
}

async function matches(path: string, asset: Artifact, signal: AbortSignal): Promise<boolean> {
  try {
    if ((await stat(path)).size !== asset.size) {
      return false
    }
    const digest = digestOf(asset)
    for await (const bytes of createReadStream(path, { signal })) {
      digest.update(bytes)
    }
    return digest.digest('hex') === asset.digest
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error
    }
    return false
  }
}

async function fetchArtifact(url: string, path: string, asset: Artifact, signal: AbortSignal, progress?: (bytes: number) => void) {
  const temporary = path + '.' + randomUUID() + '.download'
  const idle = new AbortController()
  const timer = setTimeout(() => idle.abort(new Error('等待记忆模型文件传输超时')), 45_000)
  const download = AbortSignal.any([signal, idle.signal, AbortSignal.timeout(2_700_000)])
  try {
    const response = await fetch(url, { signal: download })
    if (!response.ok || !response.body) {
      await response.body?.cancel()
      throw new Error(`HTTP ${response.status}`)
    }
    let size = 0
    let reported = 0
    const digest = digestOf(asset)
    const verify = new Transform({
      transform(bytes: Buffer, _encoding, callback) {
        size += bytes.length
        if (size > asset.size) {
          callback(new Error('记忆模型文件超过预期大小'))
          return
        }
        timer.refresh()
        digest.update(bytes)
        if (size === asset.size || size - reported >= 1_048_576) {
          progress?.(size)
          reported = size
        }
        callback(null, bytes)
      },
    })
    await pipeline(Readable.fromWeb(response.body as ReadableStream<Uint8Array>), verify,
      createWriteStream(temporary, { flags: 'wx', mode: 0o600 }), { signal: download })
    if (size !== asset.size || digest.digest('hex') !== asset.digest) {
      throw new Error('记忆模型文件大小或摘要不一致')
    }
    signal.throwIfAborted()
    try {
      await link(temporary, path)
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST' || !await matches(path, asset, signal)) {
        throw new Error('无法保存已校验的记忆模型文件，请检查本机缓存', { cause: error })
      }
    }
  } finally {
    clearTimeout(timer)
    await unlink(temporary).catch(error => {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw error
      }
    })
  }
}

async function prepare(directory: string, snapshot: string, asset: Artifact, signal: AbortSignal,
  download: boolean, progress?: (bytes: number) => void): Promise<string> {
  signal.throwIfAborted()
  const filename = basename(asset.filename)
  const path = join(directory, filename)
  if (await matches(path, asset, signal)) {
    return path
  }
  if (await stat(path).catch(error => {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error
    }
    return undefined
  })) {
    if (await matches(path, asset, signal)) {
      return path
    }
    throw new Error(`本机缓存中的 ${filename} 校验失败，请移出该文件后重试`)
  }
  if (!download) {
    throw new Error(`本机记忆模型缺少 ${filename}`)
  }
  await mkdir(directory, { recursive: true })
  const failures: string[] = []
  for (const host of ['hf-mirror.com', 'huggingface.co']) {
    try {
      progress?.(0)
      await fetchArtifact(`https://${host}/${snapshot}/${asset.filename}`, path, asset, signal, progress)
      return path
    } catch (error) {
      signal.throwIfAborted()
      const message = error instanceof Error ? error.message : String(error)
      failures.push(`${host} ${message}`)
    }
  }
  throw new Error(`下载 ${filename} 失败（${failures.join('，')}）`)
}

/** 在配置档内准备和验证绑定模型的完整文件。 */
export async function downloadLocalModel(profile: string, model: LocalModelId, signal: AbortSignal,
  onProgress?: (progress: ModelDownloadProgress) => void): Promise<void> {
  signal.throwIfAborted()
  await loadLocalRuntime()
  const source = localModels[model]
  const assets = [...source.tokenizer, source.weights]
  const cache = cacheDirectory(profile, model)
  const totalBytes = source.downloadBytes
  const totalFiles = assets.length
  let completedFiles = 0
  let completedBytes = 0
  for (const asset of assets) {
    const currentFile = basename(asset.filename)
    const update = (bytes: number) => onProgress?.({
      downloadedBytes: completedBytes + bytes, totalBytes, completedFiles, totalFiles, currentFile,
    })
    update(0)
    await prepare(cache, source.snapshot, asset, signal, true, update)
    completedBytes += asset.size
    completedFiles++
    onProgress?.({ downloadedBytes: completedBytes, totalBytes, completedFiles, totalFiles })
  }
}

/** 准备经过校验的模型词表。显式指定目录时按离线部署处理。 */
export async function tokenizerFiles(profile: string, directory: string | undefined, model: LocalModelId | typeof qwenVision,
  signal: AbortSignal, download = true) {
  if (directory) {
    return [join(directory, 'tokenizer.json'), join(directory, 'tokenizer_config.json')]
  }
  const source = model === qwenVision ? vision : localModels[model]
  const cache = cacheDirectory(profile, model)
  return Promise.all(source.tokenizer.map(file => prepare(cache, source.snapshot, file, signal, download)))
}

/** 显式指定 ONNX 路径时以文件实际摘要标识向量空间。 */
export async function modelWeights(profile: string, model: LocalModelId, path: string | undefined, signal: AbortSignal,
  download = true) {
  if (!path) {
    const { snapshot, weights } = localModels[model]
    return { path: await prepare(cacheDirectory(profile, model), snapshot, weights, signal, download), revision: weights.digest }
  }
  const digest = createHash('sha256')
  for await (const bytes of createReadStream(path, { signal })) {
    digest.update(bytes)
  }
  return { path, revision: digest.digest('hex') }
}

import { createHash, randomUUID } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import { link, mkdir, stat, unlink } from 'node:fs/promises'
import { basename, join } from 'node:path'
import { Readable, Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream } from 'node:stream/web'

const bge = 'https://huggingface.co/Xenova/bge-large-zh-v1.5/resolve/a48549b3259a6165364f226599cd91f39923d5d5/'
const qwen = 'https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/resolve/2c4565515e0f265c6511776e7193b22c0968ddc7/'

interface Artifact {
  filename: string
  base: string
  size: number
  sha256?: string
  blob?: string
}

const weights: Artifact = {
  filename: 'onnx/model_quantized.onnx',
  base: bge,
  size: 327363707,
  sha256: '49277b15e898b5cd50f87f13f677609a9cdde71d24fa8817477ca50076bf13ef',
}
const bgeTokenizer: Artifact[] = [
  { filename: 'tokenizer.json', base: bge, size: 439124, blob: '15df791881f91e7963c9e0d704de2f2037baa002' },
  { filename: 'tokenizer_config.json', base: bge, size: 394, blob: '42046bd361a033a24264dc41de1fe6cafd68072f' },
]
const qwenTokenizer: Artifact[] = [
  { filename: 'tokenizer.json', base: qwen, size: 11423705, sha256: 'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a' },
  { filename: 'tokenizer_config.json', base: qwen, size: 5404, blob: '1d4fba2de0e94e8f97e08927cfb7d084160c2880' },
]

function digestOf(asset: Artifact) {
  return asset.sha256 ? createHash('sha256') : createHash('sha1').update(`blob ${asset.size}\0`)
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
    return digest.digest('hex') === (asset.sha256 ?? asset.blob)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error
    }
    return false
  }
}

async function prepare(directory: string, asset: Artifact, signal: AbortSignal): Promise<string> {
  const path = join(directory, basename(asset.filename))
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
    throw new Error(`本机缓存的 ${basename(asset.filename)} 校验失败，请移出该文件后重试`)
  }
  await mkdir(directory, { recursive: true })
  const download = AbortSignal.any([signal, AbortSignal.timeout(600_000)])
  const response = await fetch(asset.base + asset.filename, { signal: download })
  if (!response.ok || !response.body) {
    throw new Error(`无法取得记忆模型文件 ${basename(asset.filename)}，HTTP ${response.status}`)
  }
  const temporary = path + '.' + randomUUID() + '.download'
  try {
    let size = 0
    const digest = digestOf(asset)
    const verify = new Transform({
      transform(bytes: Buffer, _encoding, callback) {
        size += bytes.length
        if (size > asset.size) {
          callback(new Error('记忆模型文件超过预期大小'))
        } else {
          digest.update(bytes)
          callback(null, bytes)
        }
      },
    })
    await pipeline(Readable.fromWeb(response.body as ReadableStream<Uint8Array>), verify,
      createWriteStream(temporary, { flags: 'wx', mode: 0o600 }), { signal: download })
    if (size !== asset.size || digest.digest('hex') !== (asset.sha256 ?? asset.blob)) {
      throw new Error(`记忆模型文件 ${basename(asset.filename)} 的校验失败`)
    }
    try {
      await link(temporary, path)
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST' || !await matches(path, asset, signal)) {
        throw new Error(`无法写入校验通过的记忆模型文件 ${basename(asset.filename)}，请检查本地缓存`, { cause: error })
      }
    }
    return path
  } finally {
    await unlink(temporary).catch(error => {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw error
      }
    })
  }
}

/** 准备经过校验的模型词表。显式指定目录时按离线部署处理。 */
export async function tokenizerFiles(cache: string, directory: string | undefined, vision: boolean, signal: AbortSignal) {
  if (directory) {
    return [join(directory, 'tokenizer.json'), join(directory, 'tokenizer_config.json')]
  }
  return Promise.all((vision ? qwenTokenizer : bgeTokenizer).map(file => prepare(cache, file, signal)))
}

/** 获取默认 CPU 量化版 BGE 文件，显式指定路径时以文件实际摘要标识向量空间。 */
export async function bgeWeights(cache: string, path: string | undefined, signal: AbortSignal) {
  if (!path) {
    return { path: await prepare(cache, weights, signal), revision: weights.sha256! }
  }
  const digest = createHash('sha256')
  for await (const bytes of createReadStream(path, { signal })) {
    digest.update(bytes)
  }
  return { path, revision: digest.digest('hex') }
}

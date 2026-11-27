import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { Tokenizer } from '@huggingface/tokenizers'
import { bgeWeights, tokenizerFiles } from './model-files.js'
import type { MediaUpload } from './contracts.js'
import { videoSamplingRevision, type VideoSegment } from './video.js'

const bge = 'bge-large-zh-v1.5'
const qwen = 'Qwen3-VL-Embedding-8B'
const bgeInstruction = '为这个句子生成表示以用于检索相关文章：'
const qwenInstruction = 'Instruct: Given a query, retrieve relevant memories.\nQuery: '

/** 本机使用指定 BGE 图和词表，路径相对于当前 DSH profile。 */
export interface LocalEmbeddingOptions {
  backend?: 'onnx'
  modelPath?: string
  tokenizerDirectory?: string
  chunkTokens?: number
}

/** 远程模型使用 OpenAI 兼容请求，Qwen 的媒体请求使用 vLLM 消息扩展。 */
export interface RemoteEmbeddingOptions {
  backend: 'openai'
  model: typeof bge | typeof qwen
  baseUrl: string
  apiKeyEnv?: string
  modelId?: string
  revision?: string
  tokenizerDirectory?: string
  maxInputTokens?: number
  dimensions?: number
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

async function loadTokenizer(cache: string, directory: string | undefined, vision: boolean, signal: AbortSignal) {
  const paths = await tokenizerFiles(cache, directory, vision, signal)
  const [data, config] = await Promise.all(paths.map(path => readFile(path, { encoding: 'utf8', signal })))
  return { tokenizer: new Tokenizer(JSON.parse(data), JSON.parse(config)), revision: fingerprint(data, config) }
}

async function local(profile: string, cache: string, options: LocalEmbeddingOptions, signal: AbortSignal): Promise<Embedding> {
  const ort = await import('onnxruntime-node').catch(cause => {
    throw new Error('当前环境缺少本机向量化所需的 onnxruntime-node', { cause })
  })
  const [lexicon, weights] = await Promise.all([
    loadTokenizer(cache, options.tokenizerDirectory && resolve(profile, options.tokenizerDirectory), false, signal),
    bgeWeights(cache, options.modelPath && resolve(profile, options.modelPath), signal),
  ])
  const chunkTokens = optionsInteger(options.chunkTokens, 384, 32, 512, 'chunkTokens')
  const session = await ort.InferenceSession.create(weights.path, { executionProviders: ['cpu'] })
  const dimensions = 1024
  const encode = (text: string) => lexicon.tokenizer.encode(text, { add_special_tokens: true, return_token_type_ids: true })
  return {
    fingerprint: fingerprint('onnx', bge, weights.revision, lexicon.revision, chunkTokens),
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
          throw new Error('本机 BGE 仅支持文本向量化')
        }
        const { ids, attention_mask, token_type_ids } = encode((purpose === 'query' ? bgeInstruction : '') + input.text)
        if (ids.length > 512) {
          throw new Error('记忆输入超过 BGE 的 512 token 上限')
        }
        const available: Record<string, number[]> = { input_ids: ids, attention_mask, token_type_ids }
        const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {}
        for (const name of session.inputNames) {
          if (!Object.hasOwn(available, name)) {
            throw new Error(`BGE ONNX 图要求未知的输入 ${name}`)
          }
          feeds[name] = new ort.Tensor('int64', BigInt64Array.from(available[name], value => BigInt(value)), [1, ids.length])
        }
        const out = await session.run(feeds)
        const output = out.last_hidden_state ?? out.sentence_embedding
        if (!output || !(output.data instanceof Float32Array) || output.dims.at(-1) !== dimensions) {
          throw new Error('BGE ONNX 图未返回受支持的 1024 维输出')
        }
        vectors.push(normalize(output.data.subarray(0, dimensions), dimensions))
      }
      return vectors
    },
    close: () => session.release(),
  }
}

function embeddingUrl(base: string) {
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
    const { model, modelId, revision, baseUrl, apiKeyEnv, tokenizerDirectory, maxInputTokens, dimensions, chunkTokens } = options
    values = { backend: options.backend, model, modelId, revision, baseUrl: embeddingUrl(baseUrl), apiKeyEnv, tokenizerDirectory,
      maxInputTokens, dimensions, chunkTokens, ...(model === qwen && { videoRevision: videoSamplingRevision }) }
  } else {
    const { backend = 'onnx', modelPath, tokenizerDirectory, chunkTokens } = options
    values = { backend, modelPath, tokenizerDirectory, chunkTokens }
  }
  if (Object.values(values).some(value => value !== undefined && typeof value !== 'string' && typeof value !== 'number')) {
    throw new Error('记忆向量配置包含无法保存的值')
  }
  return JSON.stringify(values)
}

function responseVectors(body: unknown, count: number, dimensions: number) {
  if (!body || typeof body !== 'object' || !('data' in body) || !Array.isArray(body.data) || body.data.length !== count) {
    throw new Error('向量服务返回的条目数量不匹配')
  }
  const vectors: Float32Array[] = new Array(count)
  for (const row of body.data) {
    if (!row || typeof row !== 'object' || !Number.isSafeInteger(row.index) ||
        row.index < 0 || row.index >= count || vectors[row.index] || !Array.isArray(row.embedding)) {
      throw new Error('向量服务返回了无效的条目')
    }
    vectors[row.index] = normalize(row.embedding, dimensions)
  }
  return vectors
}

async function remote(profile: string, cache: string, options: RemoteEmbeddingOptions, signal: AbortSignal,
  videoRevision: string | null): Promise<Embedding> {
  if (options.model !== bge && options.model !== qwen) {
    throw new Error('请选择受支持的远程记忆向量模型')
  }
  const vision = options.model === qwen
  const endpoint = embeddingUrl(options.baseUrl)
  const model = (options.modelId ?? options.model).trim()
  const key = options.apiKeyEnv && process.env[options.apiKeyEnv]
  if (!model || (options.apiKeyEnv && !key)) {
    throw new Error('远程向量模型名称或配置的 API 密钥环境变量为空')
  }
  const maxInputTokens = optionsInteger(options.maxInputTokens, vision ? 32768 : 512, 64,
    vision ? 32768 : 512, 'maxInputTokens')
  const textBudget = maxInputTokens - (vision ? 32 : 0)
  const dimensions = optionsInteger(options.dimensions, vision ? 4096 : 1024,
    vision ? 64 : 1024, vision ? 4096 : 1024, 'dimensions')
  const chunkTokens = optionsInteger(options.chunkTokens, Math.min(textBudget, vision ? 1024 : 384),
    32, textBudget, 'chunkTokens')
  const lexicon = await loadTokenizer(cache, options.tokenizerDirectory && resolve(profile, options.tokenizerDirectory), vision, signal)
  const countTokens = (text: string) => lexicon.tokenizer.encode(text, { add_special_tokens: true }).ids.length
  const instruction = vision ? qwenInstruction : bgeInstruction
  async function submit(input: Record<string, unknown>, count: number) {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(key ? { authorization: 'Bearer ' + key } : {}) },
      body: JSON.stringify({ model, encoding_format: 'float', ...(vision && options.dimensions !== undefined && { dimensions }), ...input }),
      signal: AbortSignal.any([signal, AbortSignal.timeout(90_000)]),
      redirect: 'error',
    })
    if (!response.ok) {
      await response.body?.cancel()
      throw new Error(`向量服务请求失败，HTTP ${response.status}`)
    }
    return responseVectors(await response.json(), count, dimensions)
  }
  return {
    fingerprint: fingerprint('openai', options.model, endpoint, model, options.revision, lexicon.revision, dimensions,
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
          throw new Error('远程 BGE 仅支持文本向量化')
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
  const vision = options.backend === 'openai' && options.model === qwen
  const cache = join(profile, 'cache', 'dsh-prism', 'models', vision ? qwen : bge)
  return options.backend === 'openai'
    ? remote(profile, cache, options, signal, videoRevision)
    : local(profile, cache, options, signal)
}

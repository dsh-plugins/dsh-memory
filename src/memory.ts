import { createHash, randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, realpathSync } from 'node:fs'
import { isAbsolute, join, relative } from 'node:path'
import { chunkText } from './chunks.js'
import { MemoryDatabase, type Attachment, type Fragment } from './database.js'
import { createEmbedding, type Embedding, type EmbeddingInput, type EmbeddingOptions } from './embedding.js'
import type { MediaReference, MediaUpload, MemoryDocument, MemoryDraft, MemoryQuery, PrismMemory } from './contracts.js'

const mediaTypes = new Set([
  'image/jpeg', 'image/png', 'image/webp', 'image/gif',
  'video/mp4', 'video/webm', 'video/quicktime',
])
const documentBytes = 512 * 1024
const mediaBytes = 16 * 1024 * 1024

/** 插件的当前配置档参数。 */
export interface MemoryOptions {
  embedding?: EmbeddingOptions
  maxMediaBytes?: number
}

function boundedText(value: unknown, upTo: number, name: string, optional = false): string {
  if (typeof value !== 'string' || Buffer.byteLength(value) > upTo || (!optional && !value.trim())) {
    throw new Error(`记忆字段 ${name} 缺失或超过字节上限 ${upTo}`)
  }
  return value
}

function canonicalDirectory(path: string) {
  mkdirSync(path, { recursive: true, mode: 0o700 })
  const resolved = realpathSync(path)
  if (relative(path, resolved) !== '') {
    throw new Error('棱镜记忆目录解析到配置档的其他位置')
  }
  return resolved
}

/** 生命周期由 Cordis 插件持有，文档和媒体保存在本机配置档。 */
export class Memory implements PrismMemory {
  readonly workspace: string
  private readonly abort = new AbortController()
  private readonly pending = new Set<Promise<unknown>>()
  private readonly database: MemoryDatabase
  private readonly assetBytes: number
  private readonly profile: string
  private embedding?: Promise<Embedding>
  private refresh?: Promise<void>
  private stopping?: Promise<void>

  constructor(profile: string, private readonly options: MemoryOptions = {}) {
    if (!isAbsolute(profile)) {
      throw new Error('DSH 记忆库需要绝对的配置档路径')
    }
    this.profile = realpathSync(profile)
    this.assetBytes = options.maxMediaBytes ?? mediaBytes
    if (!Number.isSafeInteger(this.assetBytes) || this.assetBytes < 1 || this.assetBytes > 64 * 1024 * 1024) {
      throw new Error('maxMediaBytes 必须在 1 至 67108864 之间')
    }
    const data = canonicalDirectory(join(this.profile, 'data', 'dsh-prism'))
    this.workspace = canonicalDirectory(join(data, 'workspace', 'memories'))
    const database = join(data, 'memory.sqlite')
    if (existsSync(database) && relative(database, realpathSync(database)) !== '') {
      throw new Error('棱镜记忆数据库解析到配置档的其他位置')
    }
    this.database = new MemoryDatabase(database)
  }

  private run<T>(operation: () => T | Promise<T>): Promise<T> {
    if (this.abort.signal.aborted) {
      return Promise.reject(new Error('当前配置档的记忆插件正在停止'))
    }
    const promise = Promise.resolve().then(operation)
    this.pending.add(promise)
    const settled = () => this.pending.delete(promise)
    void promise.then(settled, settled)
    return promise
  }

  private model(): Promise<Embedding> {
    if (!this.embedding) {
      const pending = createEmbedding(this.profile, this.options.embedding ?? {}, this.abort.signal)
      this.embedding = pending
      void pending.catch(() => { this.embedding = undefined })
    }
    return this.embedding
  }

  get(documentId: string): Promise<MemoryDocument | undefined> {
    return this.run(() => this.database.get(boundedText(documentId, 100, 'documentId')))
  }

  async stageMedia(input: MediaUpload): Promise<MediaReference> {
    if (!(input.bytes instanceof Uint8Array) || input.bytes.byteLength === 0 || input.bytes.byteLength > this.assetBytes ||
        typeof input.mimeType !== 'string' || !mediaTypes.has(input.mimeType)) {
      throw new Error('媒体为空、超过字节上限或格式不受支持')
    }
    const stable = {
      bytes: Buffer.from(input.bytes),
      mimeType: input.mimeType,
      ...(input.name !== undefined && { name: boundedText(input.name, 255, 'name') }),
    }
    return this.run(() => this.database.stage(randomUUID(), stable))
  }

  readMedia(assetId: string): Promise<(MediaReference & { bytes: Uint8Array }) | undefined> {
    return this.run(() => this.database.readMedia(boundedText(assetId, 100, 'assetId')))
  }

  pinMedia(assetIds: readonly string[]): Promise<void> {
    return this.run(() => {
      const ids = [...new Set(assetIds.map(id => boundedText(id, 100, 'assetId')))]
      if (ids.length > 16) {
        throw new Error('一次最多可保留 16 个媒体附件')
      }
      if (ids.length) {
        this.database.pinMedia(ids)
      }
    })
  }

  private media(assetId: string) {
    const asset = this.database.readMedia(assetId)
    if (!asset) {
      throw new Error('找不到此媒体或暂存已过期')
    }
    return asset
  }

  private attachments(draft: MemoryDraft): Attachment[] {
    const attachments = draft.attachments ?? []
    if (!Array.isArray(attachments) || attachments.length > 16) {
      throw new Error('一份记忆最多可关联 16 个媒体附件')
    }
    const parsed = attachments.map(attachment => ({
      assetId: boundedText(attachment?.assetId, 100, 'assetId'),
      ...(attachment.description !== undefined && { description: boundedText(attachment.description, 4096, 'description') }),
    }))
    if (new Set(parsed.map(attachment => attachment.assetId)).size !== parsed.length) {
      throw new Error('一份记忆不能重复关联同一媒体')
    }
    return parsed
  }

  private async fragments(doc: MemoryDocument, model: Embedding): Promise<Fragment[]> {
    const text = doc.body.trim() ? doc.title + '\n\n' + doc.body : doc.title
    const passages: EmbeddingInput[] = chunkText(text, model.chunkTokens, part => model.countTokens(part))
      .filter(part => part.trim()).map(part => ({ text: part }))
    for (const attachment of doc.attachments) {
      const label = attachment.description ?? attachment.name ?? (attachment.mimeType.startsWith('image/') ? '图片附件' : '视频附件')
      const parts = chunkText(label, model.chunkTokens, part => model.countTokens(part))
      for (const [index, text] of parts.entries()) {
        passages.push({ text, ...(index === 0 && model.supportsMedia && { media: this.media(attachment.assetId) }) })
      }
    }
    const vectors = await model.embed(passages, 'passage')
    if (vectors.length !== passages.length || !vectors.length) {
      throw new Error('记忆分片的嵌入结果数量不匹配')
    }
    return passages.map((item, index) => ({ text: item.text, vector: vectors[index] }))
  }

  write(draft: MemoryDraft, idempotencyKey: string): Promise<{ documentId: string }> {
    return this.run(async () => {
      const requestKey = boundedText(idempotencyKey, 256, 'idempotencyKey')
      const title = boundedText(draft.title, 1024, 'title').trim()
      const body = boundedText(draft.body, documentBytes, 'body', true)
      const attachments = this.attachments(draft)
      if (!body.trim() && attachments.length === 0) {
        throw new Error('记忆文档需要正文或媒体')
      }
      const digest = createHash('sha256').update(JSON.stringify([title, body, attachments])).digest('hex')
      const previous = this.database.previous(requestKey, digest)
      if (previous) {
        return { documentId: previous }
      }
      const doc: MemoryDocument = {
        documentId: randomUUID(), title, body,
        attachments: attachments.map(ref => {
          const { mimeType, name } = this.media(ref.assetId)
          return { ...ref, mimeType, ...(name !== undefined && { name }) }
        }),
      }
      const model = await this.model()
      const fragments = await this.fragments(doc, model)
      this.abort.signal.throwIfAborted()
      for (const ref of doc.attachments) {
        this.media(ref.assetId)
      }
      const documentId = this.database.save(doc, requestKey, digest, model.fingerprint, fragments)
      return { documentId }
    })
  }

  private reindex(model: Embedding): Promise<void> {
    if (!this.refresh) {
      const pending = this.indexBacklog(model)
      this.refresh = pending
      const clear = () => { this.refresh = undefined }
      void pending.then(clear, clear)
    }
    return this.refresh
  }

  private async indexBacklog(model: Embedding): Promise<void> {
    let missing: string[]
    while ((missing = this.database.unindexed(model.fingerprint)).length > 0) {
      for (const id of missing) {
        this.abort.signal.throwIfAborted()
        const doc = this.database.get(id)
        if (!doc) {
          continue
        }
        const chunks = await this.fragments(doc, model)
        this.abort.signal.throwIfAborted()
        this.database.index(id, model.fingerprint, chunks)
      }
    }
  }

  search(input: MemoryQuery): Promise<{ documentId: string; fragment: string }[]> {
    return this.run(async () => {
      const query = input.query === undefined ? '' : boundedText(input.query, 16_384, 'query', true)
      const media = input.assetId === undefined ? undefined : this.media(boundedText(input.assetId, 100, 'assetId'))
      const limit = input.limit ?? 8
      if ((!query.trim() && !media) || !Number.isSafeInteger(limit) || limit < 1 || limit > 20) {
        throw new Error('记忆检索需要文本或媒体，结果数量须在 1 至 20 之间')
      }
      const model = await this.model()
      if (media && !model.supportsMedia) {
        throw new Error('媒体检索需要远程 Qwen3-VL-Embedding-8B')
      }
      const [vector] = await model.embed([{ text: query, media }], 'query')
      await this.reindex(model)
      this.abort.signal.throwIfAborted()
      return this.database.search(vector, model.fingerprint, limit)
    })
  }

  close(): Promise<void> {
    if (!this.stopping) {
      this.abort.abort()
      this.stopping = (async () => {
        await Promise.allSettled([...this.pending])
        try {
          const model = await this.embedding?.catch(() => undefined)
          await model?.close()
        } finally {
          this.database.close()
        }
      })()
    }
    return this.stopping
  }
}

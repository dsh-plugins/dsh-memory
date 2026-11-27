/** 已暂存的媒体标识与显示信息。 */
export interface MediaReference {
  assetId: string
  mimeType: string
  name?: string
}

/** 由宿主提交的媒体字节与元数据。 */
export interface MediaUpload {
  bytes: Uint8Array
  mimeType: string
  name?: string
}

/** 记忆智能体整理完成的原文与附件。 */
export interface MemoryDraft {
  title: string
  body: string
  attachments?: readonly { assetId: string; description?: string }[]
}

/** 通过全文工具返回的文档，不含附件二进制数据。 */
export interface MemoryDocument {
  documentId: string
  title: string
  body: string
  attachments: (MediaReference & { description?: string })[]
}

/** 可按文本、已暂存媒体或两者组合查询。 */
export interface MemoryQuery {
  query?: string
  assetId?: string
  limit?: number
}

/** 棱镜编排器使用当前配置档的记忆库，调用方负责会话授权。 */
export interface PrismMemory {
  readonly workspace: string
  search(input: MemoryQuery): Promise<{ documentId: string; fragment: string }[]>
  get(documentId: string): Promise<MemoryDocument | undefined>
  write(draft: MemoryDraft, idempotencyKey: string): Promise<{ documentId: string }>
  stageMedia(input: MediaUpload): Promise<MediaReference>
  readMedia(assetId: string): Promise<(MediaReference & { bytes: Uint8Array }) | undefined>
}

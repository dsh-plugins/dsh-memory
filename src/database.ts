import { DatabaseSync } from 'node:sqlite'
import type { MediaReference, MediaUpload, MemoryDocument } from './contracts.js'

export type Attachment = { assetId: string; description?: string }
export type Fragment = { text: string; vector: Float32Array }
export type IndexSelection = { options: string; fingerprint: string }
type DocumentRow = { id: string; title: string; body: string }
type AttachmentRow = { asset_id: string; description: string | null; mime_type: string; name: string | null }
const mediaLifetime = 24 * 60 * 60 * 1000

/** 记忆库中可被引用的媒体。未入库且过期的资产不再可用。 */
export type StoredMedia = MediaReference & MediaUpload

export class MemoryDatabase {
  private readonly db: DatabaseSync

  constructor(path: string) {
    this.db = new DatabaseSync(path, { timeout: 5000 })
    try {
      this.db.exec('PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;')
      this.transaction(() => {
        const version = this.db.prepare('PRAGMA user_version').get()?.user_version
        if (version !== 0 && version !== 1 && version !== 2 && version !== 3) {
          throw new Error('当前棱镜记忆库版本不受此插件支持')
        }
        this.db.exec(`
          CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            request_key TEXT NOT NULL UNIQUE,
            content_digest TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL
          ) STRICT;
          CREATE TABLE IF NOT EXISTS media (
            id TEXT PRIMARY KEY,
            mime_type TEXT NOT NULL,
            name TEXT,
            bytes BLOB NOT NULL,
            staged_at INTEGER NOT NULL,
            pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1))
          ) STRICT;
          CREATE TABLE IF NOT EXISTS attachments (
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            asset_id TEXT NOT NULL REFERENCES media(id),
            description TEXT,
            PRIMARY KEY (document_id, ordinal)
          ) STRICT;
          CREATE INDEX IF NOT EXISTS attachment_media ON attachments(asset_id);
          CREATE TABLE IF NOT EXISTS fragments (
            fingerprint TEXT NOT NULL,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL,
            vector BLOB NOT NULL,
            PRIMARY KEY (fingerprint, document_id, ordinal)
          ) STRICT;
          CREATE TABLE IF NOT EXISTS active_index (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            options TEXT NOT NULL,
            fingerprint TEXT NOT NULL
          ) STRICT;
        `)
        if (version === 1) {
          this.db.exec('ALTER TABLE media ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1))')
        }
        this.db.exec('PRAGMA user_version = 3')
      })
    } catch (error) {
      this.db.close()
      throw error
    }
  }

  private transaction<T>(operation: () => T): T {
    this.db.exec('BEGIN IMMEDIATE')
    try {
      const result = operation()
      this.db.exec('COMMIT')
      return result
    } catch (error) {
      this.db.exec('ROLLBACK')
      throw error
    }
  }

  previous(requestKey: string, digest: string): string | undefined {
    const row = this.db.prepare('SELECT id, content_digest FROM documents WHERE request_key = ?').get(requestKey)
    if (!row) {
      return undefined
    }
    if (row.content_digest !== digest) {
      throw new Error('同一个记忆任务已经以不同内容完成写入')
    }
    return String(row.id)
  }

  get(documentId: string): MemoryDocument | undefined {
    const doc = this.db.prepare('SELECT id, title, body FROM documents WHERE id = ?').get(documentId) as DocumentRow | undefined
    if (!doc) {
      return undefined
    }
    const files = this.db.prepare(`
      SELECT a.asset_id, a.description, m.mime_type, m.name
      FROM attachments a JOIN media m ON m.id = a.asset_id
      WHERE a.document_id = ? ORDER BY a.ordinal
    `).all(doc.id) as AttachmentRow[]
    return {
      documentId: doc.id, title: doc.title, body: doc.body,
      attachments: files.map(row => ({
        assetId: row.asset_id, mimeType: row.mime_type,
        ...(row.name !== null && { name: row.name }),
        ...(row.description !== null && { description: row.description }),
      })),
    }
  }

  stage(assetId: string, input: MediaUpload): MediaReference {
    this.transaction(() => {
      this.db.prepare(`
        DELETE FROM media WHERE pinned = 0 AND staged_at < ?
        AND NOT EXISTS (SELECT 1 FROM attachments WHERE asset_id = media.id)
      `).run(Date.now() - mediaLifetime)
      this.db.prepare('INSERT INTO media (id, mime_type, name, bytes, staged_at) VALUES (?, ?, ?, ?, ?)')
        .run(assetId, input.mimeType, input.name ?? null, input.bytes, Date.now())
    })
    return { assetId, mimeType: input.mimeType, ...(input.name && { name: input.name }) }
  }

  readMedia(assetId: string): StoredMedia | undefined {
    const row = this.db.prepare(`
      SELECT mime_type, name, bytes FROM media
      WHERE id = ? AND (pinned = 1 OR staged_at >= ? OR EXISTS (SELECT 1 FROM attachments WHERE asset_id = media.id))
    `).get(assetId, Date.now() - mediaLifetime) as { mime_type: string; name: string | null; bytes: Uint8Array } | undefined
    return row && { assetId, mimeType: row.mime_type, bytes: row.bytes, ...(row.name !== null && { name: row.name }) }
  }

  pinMedia(assetIds: readonly string[]): void {
    this.transaction(() => {
      const pin = this.db.prepare(`
        UPDATE media SET pinned = 1
        WHERE id = ? AND (pinned = 1 OR staged_at >= ? OR EXISTS (SELECT 1 FROM attachments WHERE asset_id = media.id))
      `)
      const threshold = Date.now() - mediaLifetime
      for (const id of assetIds) {
        if (pin.run(id, threshold).changes !== 1) {
          throw new Error('找不到此媒体或暂存已过期')
        }
      }
    })
  }

  save(doc: MemoryDocument, requestKey: string, digest: string, fingerprint: string, fragments: Fragment[],
    authorize?: () => void): string | undefined {
    return this.transaction(() => {
      const previous = this.previous(requestKey, digest)
      if (previous) {
        authorize?.()
        return previous
      }
      if (this.activeIndex()?.fingerprint !== fingerprint) {
        authorize?.()
        return undefined
      }
      this.db.prepare('INSERT INTO documents (id, request_key, content_digest, title, body) VALUES (?, ?, ?, ?, ?)')
        .run(doc.documentId, requestKey, digest, doc.title, doc.body)
      const attach = this.db.prepare('INSERT INTO attachments (document_id, ordinal, asset_id, description) VALUES (?, ?, ?, ?)')
      for (const [ordinal, asset] of doc.attachments.entries()) {
        attach.run(doc.documentId, ordinal, asset.assetId, asset.description ?? null)
      }
      this.insertFragments(doc.documentId, fingerprint, fragments)
      authorize?.()
      return doc.documentId
    })
  }

  activeIndex(): IndexSelection | undefined {
    return this.db.prepare('SELECT options, fingerprint FROM active_index WHERE id = 1').get() as IndexSelection | undefined
  }

  activateIndex(expected: IndexSelection | undefined, next: IndexSelection): 'active' | 'changed' | 'incomplete' {
    return this.transaction(() => {
      const current = this.activeIndex()
      if (current?.options !== expected?.options || current?.fingerprint !== expected?.fingerprint) {
        return 'changed'
      }
      if (this.unindexed(next.fingerprint).length > 0) {
        return 'incomplete'
      }
      this.db.prepare(`
        INSERT INTO active_index (id, options, fingerprint) VALUES (1, ?, ?)
        ON CONFLICT (id) DO UPDATE SET options = excluded.options, fingerprint = excluded.fingerprint
      `).run(next.options, next.fingerprint)
      return 'active'
    })
  }

  private insertFragments(documentId: string, fingerprint: string, fragments: Fragment[]): void {
    const insert = this.db.prepare('INSERT INTO fragments (fingerprint, document_id, ordinal, text, vector) VALUES (?, ?, ?, ?, ?)')
    for (const [ordinal, item] of fragments.entries()) {
      const encoded = Buffer.allocUnsafe(item.vector.length * 4)
      item.vector.forEach((value, index) => encoded.writeFloatLE(value, index * 4))
      insert.run(fingerprint, documentId, ordinal, item.text, encoded)
    }
  }

  unindexed(fingerprint: string): string[] {
    return this.db.prepare(`
      SELECT id FROM documents WHERE NOT EXISTS (
        SELECT 1 FROM fragments WHERE fingerprint = ? AND document_id = documents.id
      ) ORDER BY id LIMIT 32
    `).all(fingerprint).map(row => String(row.id))
  }

  index(documentId: string, fingerprint: string, fragments: Fragment[]): void {
    this.transaction(() => {
      if (!this.db.prepare('SELECT 1 FROM fragments WHERE fingerprint = ? AND document_id = ? LIMIT 1').get(fingerprint, documentId)) {
        this.insertFragments(documentId, fingerprint, fragments)
      }
    })
  }

  search(vectors: readonly Float32Array[], fingerprint: string, limit: number): { documentId: string; fragment: string }[] {
    const best: { documentId: string; fragment: string; score: number }[] = []
    const rows = this.db.prepare('SELECT document_id, text, vector FROM fragments WHERE fingerprint = ? ORDER BY document_id, ordinal')
      .iterate(fingerprint) as Iterable<{ document_id: string; text: string; vector: Uint8Array }>
    for (const row of rows) {
      if (vectors.some(vector => row.vector.byteLength !== vector.length * 4)) {
        throw new Error('记忆索引中的向量维度与模型不一致')
      }
      const bytes = new DataView(row.vector.buffer, row.vector.byteOffset, row.vector.byteLength)
      let score = -Infinity
      for (const vector of vectors) {
        let similarity = 0
        for (let index = 0; index < vector.length; index++) {
          similarity += vector[index] * bytes.getFloat32(index * 4, true)
        }
        score = Math.max(score, similarity)
      }
      if (best.length < limit || score > best[best.length - 1].score) {
        best.push({ documentId: row.document_id, fragment: row.text, score })
        best.sort((a, b) => b.score - a.score)
        if (best.length > limit) {
          best.pop()
        }
      }
    }
    return best.map(({ documentId, fragment }) => ({ documentId, fragment }))
  }

  close(): void {
    this.db.close()
  }
}

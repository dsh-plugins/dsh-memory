import { isAbsolute } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { Memory, type MemoryOptions } from './memory.js'

export type { MediaReference, MediaUpload, MemoryDocument, MemoryDraft, MemoryQuery, PrismMemory } from './contracts.js'
export type { EmbeddingOptions, LocalEmbeddingOptions, RemoteEmbeddingOptions } from './embedding.js'
export type { MemoryOptions } from './memory.js'
export type { VideoOptions } from './video.js'

export const name = '@dsh-prism/dsh-memory'
export const inject = ['profileContext']

export function apply(ctx: Context, config: MemoryOptions = {}) {
  const profile = ctx.get('profileContext', true) as { dir?: string }
  if (typeof profile?.dir !== 'string' || !isAbsolute(profile.dir)) {
    throw new Error('DSH 未提供当前配置档目录')
  }
  const memory = new Memory(profile.dir, config, message => ctx.logger.warn(message))
  ctx.provide('prismMemory', memory)
  return () => memory.close()
}

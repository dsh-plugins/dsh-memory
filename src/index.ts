import { isAbsolute } from 'node:path'
import { realpath } from 'node:fs/promises'
import type { Context } from '@deepseek-ai/cordis'
import type { HostAPI } from '@dsh-plugin/dsh-loader'
import { MemoryBinding } from './binding.js'
import { Memory, type MemoryOptions } from './memory.js'
import { memorySettingsRoute } from './settings-route.js'

export type { MediaReference, MediaUpload, MemoryDocument, MemoryDraft, MemoryQuery, PrismMemory } from './contracts.js'
export type { EmbeddingOptions, LocalEmbeddingOptions, RemoteEmbeddingOptions } from './embedding.js'
export type { MemoryOptions } from './memory.js'
export type { VideoOptions } from './video.js'

export const name = '@dsh-prism/dsh-memory'
export const inject = ['profileContext']

export async function apply(ctx: Context, config: MemoryOptions = {}) {
  const profile = ctx.get('profileContext', true) as { dir?: string }
  if (typeof profile?.dir !== 'string' || !isAbsolute(profile.dir)) {
    throw new Error('DSH 未提供当前配置档目录')
  }
  const directory = await realpath(profile.dir)
  const report = (message: string) => ctx.logger.warn(message)
  let memory: Memory | undefined
  const binding = new MemoryBinding(directory, async (selected, signal) => {
    signal.throwIfAborted()
    const instance = new Memory(directory, { ...config, embedding: selected.options }, report, selected.credential)
    memory = instance
    try {
      if (selected.managed) {
        await instance.initialize()
      }
      signal.throwIfAborted()
      ctx.provide('prismMemory', instance)
    } catch (error) {
      await instance.close()
      memory = undefined
      throw error
    }
  }, report)
  await binding.restore(config.embedding)
  ctx.inject(['dshLoader'], scope => {
    const loader = scope.get('dshLoader', true) as HostAPI
    scope.effect(() => loader.web.registerOwner(memorySettingsRoute(binding)), '棱镜记忆设置')
  })
  return async () => {
    await Promise.all([binding.close(), memory?.close()])
  }
}

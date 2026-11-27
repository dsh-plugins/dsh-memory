import { useSyncExternalStore } from 'react'
import type { LocaleRuntime } from '@deepseek-ai/dsh-client-locale/client'
import type { SlotCore } from '@deepseek-ai/dsh-client-ui-slots'
import type { SettingsSectionIconOwnerProps } from '@deepseek-ai/dsh-client-ui-settings/client'
import { memoryCopy } from './copy.js'
import { MemorySettingsPage } from './memory-settings.js'

const section = 'prism-memory'

interface BrowserContext {
  locale: Pick<LocaleRuntime, 'subscribe' | 'getSnapshot'>
  slots: Pick<SlotCore, 'register'> & {
    inject(name: 'settings.section' | 'settings.section.icon', mount: () => () => void): void
  }
}

export const name = '@dsh-prism/dsh-memory'
export const inject = ['slots', 'locale', 'dshLoaderUi']

/** 在宿主设置导航注册记忆设置页，并使用宿主当前语言。 */
export function apply(ctx: BrowserContext) {
  function Settings() {
    const language = useSyncExternalStore(listener => ctx.locale.subscribe(listener), () => ctx.locale.getSnapshot().active)
    return <MemorySettingsPage language={language} />
  }

  ctx.slots.inject('settings.section', () => ctx.slots.register({
    id: section, name: 'settings.section', order: 62,
    label: () => memoryCopy(ctx.locale.getSnapshot().active).nav,
  }, Settings))
  ctx.slots.inject('settings.section.icon', () => ctx.slots.register({
    id: section, name: 'settings.section.icon',
  }, ({ size }: SettingsSectionIconOwnerProps) => (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.25"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <ellipse cx="8" cy="3.6" rx="5.2" ry="2.1" />
      <path d="M2.8 3.6v8.8C2.8 13.6 5.1 14.5 8 14.5s5.2-.9 5.2-2.1V3.6M2.8 8c0 1.2 2.3 2.1 5.2 2.1s5.2-.9 5.2-2.1" />
    </svg>
  )))
}

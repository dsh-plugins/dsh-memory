export type ModelSource = 'local' | 'remote'

export type BindingRequest =
  | { source: 'local'; modelId: string }
  | { source: 'remote'; modelId: string; baseUrl: string; apiKey: string }

export type BindingStatus =
  | 'unconfigured' | 'downloading' | 'connecting' | 'initializing' | 'active' | 'failed'

export interface DownloadProgress {
  receivedBytes: number
  totalBytes?: number
  completedFiles: number
  totalFiles: number
  currentFile?: string
}

export interface MemorySettings {
  status: BindingStatus
  locked?: boolean
  source?: ModelSource
  modelId?: string
  baseUrl?: string
  progress?: DownloadProgress
  dimensions?: number
  contextLength?: number
  contextSource?: 'reported' | 'model' | 'fallback'
  supportsMedia?: boolean
  error?: string
}

export async function requestSettings(signal: AbortSignal, binding?: BindingRequest): Promise<MemorySettings> {
  const response = await fetch('/api/prism/memory/settings', {
    method: binding ? 'POST' : 'GET',
    credentials: 'same-origin',
    cache: 'no-store',
    signal,
    ...(binding && {
      headers: { 'Content-Type': 'application/json', 'X-Prism-Request': '1' },
      body: JSON.stringify(binding),
    }),
  })
  const data = await response.json().catch(() => null)
  if (!response.ok || typeof data?.status !== 'string') {
    throw new Error(typeof data?.error === 'string' ? data.error : '')
  }
  return data as MemorySettings
}

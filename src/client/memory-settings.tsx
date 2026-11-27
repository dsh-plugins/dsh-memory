import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { Arco, DshMenu, Icon, Scrollbar } from '@dsh-plugin/dsh-loader/client'
import { memoryCopy } from './copy.js'
import {
  requestSettings, type BindingRequest, type BindingStatus, type DownloadProgress,
  type MemorySettings, type ModelSource,
} from './settings-api.js'

const localModels = ['bge-m3', 'bge-large-zh-v1.5', 'bge-large-en-v1.5', 'Qwen3-Embedding-0.6B']
const emptyRemote = { baseUrl: '', modelId: '', apiKey: '' }
const textStyle = { margin: 0, overflowWrap: 'anywhere' } as const
const fieldStyle = { display: 'grid', gap: 6, minWidth: 0 }
type Copy = ReturnType<typeof memoryCopy>

function inProgress(status?: BindingStatus): boolean {
  return status === 'downloading' || status === 'connecting' || status === 'initializing'
}

function ToggleButtonGroup({ label, name, copy, value, disabled, onChange }: {
  label: string; name: string; copy: Copy; value: ModelSource; disabled: boolean; onChange(value: ModelSource): void
}) {
  return <div role="radiogroup" aria-labelledby={label}>
    <Arco.Radio.Group type="button" name={name} value={value} disabled={disabled} onChange={onChange}
      options={[{ value: 'local', label: copy.local }, { value: 'remote', label: copy.remote }]} />
  </div>
}

function DownloadBar({ progress, language, copy }: { progress?: DownloadProgress; language: string; copy: Copy }) {
  const total = progress?.totalBytes
  const percent = total && progress ? Math.min(100, Math.floor(progress.receivedBytes / total * 100)) : undefined
  const bytes = new Intl.NumberFormat(language, { maximumFractionDigits: 1 })
  return <div style={fieldStyle}>
    {percent === undefined
      ? <progress aria-label={copy.status.downloading} style={{ width: '100%', accentColor: 'rgb(var(--primary-6))' }} />
      : <Arco.Progress percent={percent} />}
    {progress && <div style={textStyle}>
      {copy.files(progress.completedFiles, progress.totalFiles)}
      {progress.receivedBytes > 0 && <> · {bytes.format(progress.receivedBytes / 1024 ** 2)} MiB
        {total && <> / {bytes.format(total / 1024 ** 2)} MiB</>}
      </>}
      {progress.currentFile && <> · {progress.currentFile}</>}
    </div>}
  </div>
}

function BindingDetails({ view, copy, language }: { view: MemorySettings; copy: Copy; language: string }) {
  const count = new Intl.NumberFormat(language)
  return <dl style={{ ...textStyle, display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 18px' }}>
    {view.dimensions !== undefined && <>
      <dt>{copy.dimensions}</dt><dd style={textStyle}>{count.format(view.dimensions)}</dd>
    </>}
    {view.contextLength !== undefined && <>
      <dt>{copy.context}</dt><dd style={textStyle}>{count.format(view.contextLength)} {copy.tokens}
        {view.contextSource === 'fallback' && <> · {copy.fallbackContext}</>}
      </dd>
    </>}
    {view.supportsMedia !== undefined && <>
      <dt>{copy.media}</dt><dd style={textStyle}>{view.supportsMedia ? copy.supported : copy.unsupported}</dd>
    </>}
  </dl>
}

/** 记忆绑定、进度和能力信息均以当前配置档返回的状态为准。 */
export function MemorySettingsPage({ language }: { language: string }) {
  const copy = memoryCopy(language)
  const id = useId()
  const page = useRef<HTMLElement>(null)
  const posting = useRef<AbortController>()
  const [state, setState] = useState<MemorySettings>()
  const [sourceDraft, setSource] = useState<ModelSource>('local')
  const [localDraft, setLocal] = useState(localModels[0])
  const [modelMenu, setModelMenu] = useState<'top' | 'bottom'>()
  const [remoteDraft, setRemote] = useState(emptyRemote)
  const [confirmation, setConfirmation] = useState<BindingRequest>()
  const [refresh, setRefresh] = useState(0)
  const [pending, setPending] = useState(false)
  const [requestError, setRequestError] = useState<{ operation: 'refresh' | 'submit'; message?: string }>()

  const receive = useCallback((next: MemorySettings) => {
    setState(next)
    if (next.source) {
      setSource(next.source)
    }
    if (next.source === 'local' && next.modelId && localModels.includes(next.modelId)) {
      setLocal(next.modelId)
    }
    if (next.source === 'remote') {
      setRemote(previous => ({
        baseUrl: next.baseUrl ?? previous.baseUrl,
        modelId: next.modelId ?? previous.modelId,
        apiKey: next.locked || next.status === 'active' ? '' : previous.apiKey,
      }))
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void requestSettings(controller.signal).then(next => {
      if (!controller.signal.aborted && !posting.current) {
        receive(next)
      }
    }).catch(() => {
      if (!controller.signal.aborted && !posting.current) {
        setRequestError(previous => previous ?? { operation: 'refresh' })
      }
    })
    return () => controller.abort()
  }, [receive, refresh])

  const busy = inProgress(state?.status)
  useEffect(() => {
    if (!busy) {
      return
    }
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const next = await requestSettings(controller.signal)
        if (controller.signal.aborted) {
          return
        }
        receive(next)
        setRequestError(undefined)
        if (inProgress(next.status)) {
          timer = setTimeout(() => { void poll() }, 850)
        }
      } catch {
        if (!controller.signal.aborted) {
          setRequestError({ operation: 'refresh' })
          timer = setTimeout(() => { void poll() }, 2500)
        }
      }
    }
    timer = setTimeout(() => { void poll() }, 850)
    return () => { clearTimeout(timer); controller.abort() }
  }, [busy, receive])

  useEffect(() => () => posting.current?.abort(), [])

  const locked = Boolean(state?.locked) || state?.status === 'active'
  const disabled = !state || locked || busy || pending
  const source = disabled ? state?.source ?? sourceDraft : sourceDraft
  const localModel = disabled && state?.source === 'local' ? state.modelId ?? localDraft : localDraft
  const modelChoices = localModels.includes(localModel) ? localModels : [...localModels, localModel]
  const remoteModel = disabled && state?.source === 'remote' ? state.modelId ?? remoteDraft.modelId : remoteDraft.modelId
  const remoteBase = disabled && state?.source === 'remote' ? state.baseUrl ?? remoteDraft.baseUrl : remoteDraft.baseUrl
  const baseUrl = URL.parse(remoteBase.trim())
  const remoteValid = Boolean(baseUrl && ['http:', 'https:'].includes(baseUrl.protocol)
    && !baseUrl.username && !baseUrl.password && remoteModel.trim())

  async function bind() {
    if (!confirmation || posting.current || disabled) {
      return
    }
    const controller = new AbortController()
    posting.current = controller
    setPending(true)
    setRequestError(undefined)
    setConfirmation(undefined)
    try {
      const next = await requestSettings(controller.signal, confirmation)
      if (!controller.signal.aborted) {
        receive(next)
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setRequestError({ operation: 'submit', message: error instanceof Error ? error.message : undefined })
        setRefresh(value => value + 1)
      }
    } finally {
      if (!controller.signal.aborted) {
        posting.current = undefined
        setPending(false)
      }
    }
  }

  const action = source === 'local' ? copy.download : copy.connect
  const statusMessage = state && (copy.status[state.status] ?? copy.refreshError)
  const hasDetails = state && (state.dimensions !== undefined || state.contextLength !== undefined || state.supportsMedia !== undefined)
  return <section ref={page} aria-labelledby={`${id}-title`}
    onKeyDownCapture={event => {
      if (confirmation && event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        setConfirmation(undefined)
      }
    }}
    style={{ display: 'grid', gap: 14, maxWidth: 620, padding: '8px 0 20px' }}>
    <h2 id={`${id}-title`} style={textStyle}>{copy.title}</h2>
    <div style={{ ...fieldStyle, justifyItems: 'start' }}>
      <span id={`${id}-source-label`}>{copy.source}</span>
      <ToggleButtonGroup label={`${id}-source-label`} name={`${id}-source`} copy={copy}
        value={source} disabled={disabled} onChange={setSource} />
    </div>
    <form style={{ display: 'grid', gap: 12 }} onSubmit={event => {
      event.preventDefault()
      if (disabled || (source === 'remote' && !remoteValid)) {
        return
      }
      setConfirmation(source === 'local' ? { source, modelId: localModel } : {
        source, modelId: remoteModel.trim(), baseUrl: remoteBase.trim(), apiKey: remoteDraft.apiKey,
      })
    }}>
      {source === 'local' ? <div style={fieldStyle}>
        <span>{copy.model}</span>
        <DshMenu portal open={modelMenu !== undefined && !disabled} side={modelMenu} selectedId={localModel}
          items={modelChoices.map(model => ({ id: model, label: model }))} onClose={() => setModelMenu(undefined)}
          onSelect={model => {
            setModelMenu(undefined)
            if (!disabled && modelChoices.includes(model)) {
              setLocal(model)
            }
          }} anchor={<button type="button" aria-label={`${copy.model} ${localModel}`} aria-haspopup="menu"
            aria-expanded={modelMenu !== undefined && !disabled} disabled={disabled}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10,
              padding: '6px 12px', minHeight: 36, width: '100%', border: 0, borderRadius: 'var(--dsw-radius-md, 8px)',
              background: 'var(--dsw-alias-bg-module-platform, var(--color-fill-2))',
              color: 'var(--dsw-alias-label-primary, inherit)', font: 'inherit', opacity: disabled ? 0.55 : 1,
              cursor: disabled ? 'not-allowed' : 'pointer' }}
            onClick={event => {
              const rect = event.currentTarget.getBoundingClientRect()
              const below = window.innerHeight - rect.bottom
              setModelMenu(modelMenu === undefined ? below < Math.min(180, rect.top) ? 'top' : 'bottom' : undefined)
            }}>
            {localModel}<Icon name="ChevronDown" size={14} />
          </button>} />
      </div> : <>
        <label style={fieldStyle}>
          BASE_URL
          <Arco.Input type="url" inputMode="url" autoComplete="off" value={remoteBase} disabled={disabled}
            placeholder="https://example.com/v1" onChange={baseUrl => setRemote(previous => ({ ...previous, baseUrl }))} />
        </label>
        <label style={fieldStyle}>
          API_KEY
          <Arco.Input.Password autoComplete="new-password" value={remoteDraft.apiKey} disabled={disabled}
            placeholder={locked ? copy.passwordLocked : copy.passwordOptional}
            onChange={apiKey => setRemote(previous => ({ ...previous, apiKey }))} />
        </label>
        <label style={fieldStyle}>
          MODEL_ID
          <Arco.Input autoComplete="off" value={remoteModel} disabled={disabled}
            onChange={modelId => setRemote(previous => ({ ...previous, modelId }))} />
        </label>
      </>}
      {!locked && <div>
        <Arco.Button type="primary" htmlType="submit" loading={pending}
          disabled={disabled || (source === 'remote' && !remoteValid)}>{action}</Arco.Button>
      </div>}
    </form>
    {statusMessage && <p role="status" style={textStyle}>{statusMessage}</p>}
    {state?.status === 'downloading' && <DownloadBar progress={state.progress} language={language} copy={copy} />}
    {hasDetails && <BindingDetails view={state} copy={copy} language={language} />}
    {state?.error && <Arco.Alert type="error" content={state.error} style={textStyle} />}
    {requestError && <div style={fieldStyle}>
      <Arco.Alert type="error" content={requestError.message || (requestError.operation === 'submit' ? copy.startError : copy.refreshError)} style={textStyle} />
      <div><Arco.Button onClick={() => { setRequestError(undefined); setRefresh(value => value + 1) }}>{copy.refresh}</Arco.Button></div>
    </div>}
    <Arco.Modal
      visible={confirmation !== undefined} unmountOnExit maskClosable={false} escToExit={false}
      title={confirmation?.source === 'local' ? copy.downloadConfirm : copy.connectConfirm}
      okText={confirmation?.source === 'local' ? copy.download : copy.connect} cancelText={copy.cancel}
      onCancel={() => setConfirmation(undefined)} onOk={() => { void bind() }}
      getPopupContainer={() => page.current?.closest('[data-shortcut-modal="settings"]') ?? document.body}
      style={{ maxWidth: 'calc(100vw - 32px)', maxHeight: 'calc(100dvh - 240px)' }}
    >
      <Scrollbar style={{ maxHeight: 'max(32px, calc(100dvh - 400px))' }}>
        <p style={textStyle}>{copy.confirmWarning}</p>
        {confirmation && <p style={{ ...textStyle, marginTop: 10 }}>{copy.selectedModel(confirmation.modelId)}</p>}
      </Scrollbar>
    </Arco.Modal>
  </section>
}

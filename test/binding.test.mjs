import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { setTimeout as wait } from 'node:timers/promises'
import test from 'node:test'
import { apply } from '../dist/index.js'

async function until(read, predicate) {
  const deadline = Date.now() + 15_000
  while (Date.now() < deadline) {
    const value = await read()
    if (predicate(value)) {
      return value
    }
    await wait(20)
  }
  assert.fail('等待绑定状态超时')
}

test('activates a verified binding immediately and restores it at startup', async t => {
  const path = await mkdtemp(join(tmpdir(), 'prism-memory-binding-'))
  const secret = 'key-for-fixture-only'
  const calls = []
  let settingsRoute
  const server = createServer(async (req, res) => {
    res.setHeader('Connection', 'close')
    if (settingsRoute && req.url === settingsRoute.path) {
      return settingsRoute.handler(req, res)
    }
    const chunks = []
    for await (const chunk of req) {
      chunks.push(chunk)
    }
    calls.push(req.headers.authorization)
    const reply = (status, value) => res.writeHead(status, { 'Content-Type': 'application/json' }).end(JSON.stringify(value))
    if (req.headers.authorization !== `Bearer ${secret}`) {
      return reply(401, { error: 'invalid key' })
    }
    if (req.method === 'GET' && req.url?.includes('/models')) {
      const metadata = { id: 'fixture-model', max_model_len: 256 }
      return reply(200, req.url.endsWith('/models') ? { data: [metadata] } : metadata)
    }
    if (req.method !== 'POST' || req.url !== '/v1/embeddings') {
      return reply(404, { error: 'not found' })
    }
    const body = JSON.parse(Buffer.concat(chunks).toString('utf8'))
    assert.equal(body.model, 'fixture-model')
    const inputs = Array.isArray(body.input) ? body.input : [body.input]
    return reply(200, { data: inputs.map((_, index) => ({ index, embedding: [1, 0, 0, 0] })) })
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const origin = `http://127.0.0.1:${server.address().port}`
  const settings = origin + '/api/prism/memory/settings'
  const requestSettings = (options) => fetch(settings, options).catch(cause => {
    throw new Error(`无法从测试服务获取 ${options?.method ?? 'GET'} 记忆设置响应`, { cause })
  })
  const readStatus = async () => (await requestSettings()).json()
  const submit = (body, from = origin) => requestSettings({ method: 'POST', headers: {
    'Content-Type': 'application/json', Origin: from, 'X-Prism-Request': '1',
  }, body: JSON.stringify(body) })
  const provided = new Map()
  const ctx = {
    get: name => name === 'profileContext' ? { dir: path } : undefined,
    provide: (name, value) => provided.set(name, value),
    inject(_dependencies, callback) {
      callback({ get: () => ({ web: { registerOwner(route) { settingsRoute = route } } }), effect: setup => setup() })
    },
    logger: { warn: t.diagnostic.bind(t) },
  }
  let deactivate
  t.after(async () => {
    await deactivate?.()
    server.closeAllConnections()
    await new Promise(resolve => server.close(resolve))
    await rm(path, { recursive: true, force: true })
  })
  deactivate = await apply(ctx)
  assert.equal(provided.has('prismMemory'), false)
  assert.deepEqual(await readStatus(), { status: 'unconfigured' })

  const remote = { source: 'remote', baseUrl: origin + '/v1', modelId: 'fixture-model' }
  const file = join(path, 'data', 'dsh-prism', 'memory-binding.json')
  assert.equal((await submit({ ...remote, apiKey: secret }, 'https://other.test')).status, 403)
  assert.equal((await submit({ ...remote, apiKey: 'invalid' })).status, 202)
  assert.match((await until(readStatus, ({ status }) => status === 'failed')).error, /401/u)
  await assert.rejects(stat(file), { code: 'ENOENT' })
  assert.equal(provided.has('prismMemory'), false)

  const started = await submit({ ...remote, apiKey: secret })
  assert.equal(started.status, 202)
  assert.equal((await started.json()).status, 'connecting')
  const ready = await until(readStatus, ({ status }) => status === 'active' || status === 'failed')
  assert.equal(ready.status, 'active')
  assert.equal(ready.locked, true)
  assert.equal(ready.dimensions, 4)
  assert.equal(ready.contextLength, 256)
  assert.equal(ready.contextSource, 'reported')
  assert(!JSON.stringify(ready).includes(secret))
  assert.equal(provided.has('prismMemory'), true)
  assert.equal((await submit({ source: 'local', modelId: 'bge-m3' })).status, 409)
  if (process.platform !== 'win32') {
    assert.equal((await stat(file)).mode & 0o777, 0o600)
  }
  assert.equal(JSON.parse(await readFile(file, 'utf8')).modelId, 'fixture-model')

  const memory = provided.get('prismMemory')
  const { documentId } = await memory.write({ title: '绑定验证', body: '验证已绑定的远程向量模型' }, 'binding-test')
  assert.equal((await memory.search({ query: '验证' }))[0].documentId, documentId)
  await deactivate()
  provided.clear()
  deactivate = await apply(ctx)
  await until(readStatus, ({ status }) => status === 'active')
  assert.equal((await provided.get('prismMemory').search({ query: '恢复' }))[0].documentId, documentId)
  assert(calls.includes(`Bearer ${secret}`))
  const database = new DatabaseSync(join(path, 'data', 'dsh-prism', 'memory.sqlite'), { readOnly: true })
  try {
    const actual = database.prepare('SELECT options FROM active_index').get().options
    assert(!actual.includes(secret))
    assert.equal(JSON.parse(actual).modelId, 'fixture-model')
  } finally {
    database.close()
  }
})

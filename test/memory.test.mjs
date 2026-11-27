import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { setTimeout as wait } from 'node:timers/promises'
import test from 'node:test'
import { chunkText } from '../dist/chunks.js'
import { Memory } from '../dist/memory.js'

const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+bN9sAAAAASUVORK5CYII=', 'base64')
const mp4 = Buffer.concat([Buffer.from([0, 0, 0, 16]), Buffer.from('ftypisom'), Buffer.alloc(4)])

async function fixture(t) {
  const profile = await mkdtemp(join(tmpdir(), 'dsh-memory-'))
  const instances = []
  const open = (options, directory = profile, report) => {
    const instance = new Memory(directory, options, report)
    instances.push(instance)
    return instance
  }
  const tokenizerDirectory = join(profile, 'fixture-tokenizer')
  await mkdir(tokenizerDirectory)
  const config = {
    version: '1.0', added_tokens: [], pre_tokenizer: { type: 'WhitespaceSplit' },
    decoder: null, post_processor: null, normalizer: null,
    model: { type: 'WordPiece', unk_token: '[UNK]', continuing_subword_prefix: '##', vocab: { '[UNK]': 0, alpha: 1, beta: 2 } },
  }
  await Promise.all([
    writeFile(join(tokenizerDirectory, 'tokenizer.json'), JSON.stringify(config)),
    writeFile(join(tokenizerDirectory, 'tokenizer_config.json'), JSON.stringify({ unk_token: '[UNK]' })),
  ])
  const calls = []
  const result = { broken: false, failedModel: undefined, beforeEmbedding: undefined, afterEmbedding: undefined }
  const server = createServer(async (request, response) => {
    const parts = []
    for await (const part of request) {
      parts.push(part)
    }
    const body = JSON.parse(Buffer.concat(parts).toString('utf8'))
    calls.push({ url: request.url, body })
    await result.beforeEmbedding?.(body)
    const input = body.input ?? [body.messages]
    const data = input.map((value, index) => {
      const vector = Array(body.dimensions ?? 1024).fill(0)
      const isBeta = typeof value === 'string' && /beta/u.test(value)
      vector[isBeta ? 1 : 0] = 1
      return { index, embedding: result.broken || result.failedModel === body.model ? vector.slice(0, 2) : vector }
    })
    response.setHeader('content-type', 'application/json')
    result.afterEmbedding?.()
    response.end(JSON.stringify({ data: data.reverse() }))
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  t.after(async () => {
    await Promise.all(instances.map(instance => instance.close()))
    server.closeAllConnections()
    await new Promise(resolve => server.close(resolve))
    await rm(profile, { recursive: true, force: true })
  })
  const baseUrl = `http://127.0.0.1:${server.address().port}/v1`
  const bge = { embedding: { backend: 'openai', model: 'bge-large-zh-v1.5', baseUrl, tokenizerDirectory, chunkTokens: 32 } }
  const qwen = { embedding: { backend: 'openai', model: 'Qwen3-VL-Embedding-8B', baseUrl, tokenizerDirectory, dimensions: 64, chunkTokens: 32 } }
  return { profile, calls, result, bge, qwen, open }
}

test('splits at model token budgets without changing Unicode or paragraph boundaries', () => {
  const text = '开头。\n\n' + '🙂中文'.repeat(25) + '\n\n结尾。'
  const tokens = value => Array.from(value).length
  const fragments = chunkText(text, 13, tokens)
  assert.equal(fragments.join(''), text)
  assert(fragments.every(fragment => tokens(fragment) <= 13))
  assert(fragments.every(fragment => !/[\uD800-\uDFFF]/u.test(fragment)))
  assert.throws(() => chunkText('甲', 1, () => 2), /预算/u)
})

test('persists full documents independently of fragments and deduplicates concurrent writes', async t => {
  const { profile, bge, calls, open } = await fixture(t)
  const memory = open(bge)
  assert.equal(memory.workspace, join(profile, 'data', 'dsh-prism', 'workspace', 'memories'))
  const body = 'alpha '.repeat(50).trim()
  const draft = { title: 'alpha', body }
  const [first, retry] = await Promise.all([
    memory.write(draft, 'task-a'), memory.write(draft, 'task-a'),
  ])
  assert.equal(first.documentId, retry.documentId)
  await assert.rejects(memory.write({ ...draft, body: 'beta' }, 'task-a'), /不同内容/u)
  const other = await memory.write({ title: 'beta', body: 'beta' }, 'task-b')
  const fragments = await memory.search({ query: 'alpha', limit: 2 })
  assert.equal(fragments.length, 2)
  assert(fragments.every(item => item.documentId === first.documentId && !Object.hasOwn(item, 'body')))
  assert(calls.every(call => call.url === '/v1/embeddings'))
  assert(calls.some(call => call.body.input.length >= 2))
  await memory.close()
  const reopened = open(bge)
  assert.deepEqual(await reopened.get(first.documentId), { documentId: first.documentId, ...draft, attachments: [] })
  assert.equal((await reopened.search({ query: 'beta', limit: 1 }))[0].documentId, other.documentId)
  const otherProfile = join(profile, 'second-profile')
  await mkdir(otherProfile)
  const independent = open(bge, otherProfile)
  assert.equal(await independent.get(first.documentId), undefined)
})

test('reindexes original documents with Qwen and retains image and video assets', async t => {
  const { bge, qwen, calls, open, result } = await fixture(t)
  const textMemory = open(bge)
  const image = await textMemory.stageMedia({ bytes: png, mimeType: 'image/png', name: 'sample.png' })
  const video = await textMemory.stageMedia({ bytes: mp4, mimeType: 'video/mp4' })
  const doc = await textMemory.write({
    title: 'alpha', body: '', attachments: [{ assetId: image.assetId, description: 'image' }, { assetId: video.assetId }],
  }, 'task-media')
  await assert.rejects(textMemory.search({ assetId: image.assetId }), /Qwen/u)
  await textMemory.close()
  const candidateStarted = Promise.withResolvers()
  const releaseCandidate = Promise.withResolvers()
  result.beforeEmbedding = async body => {
    if (body.model === qwen.embedding.model) {
      candidateStarted.resolve()
      await releaseCandidate.promise
    }
  }
  const vision = open(qwen)
  try {
    await Promise.race([candidateStarted.promise, wait(2000).then(() => { throw new Error('background index did not start') })])
    assert.equal((await vision.search({ query: 'alpha', limit: 1 }))[0].documentId, doc.documentId)
    assert.notEqual((await vision.write({ title: 'beta', body: 'beta' }, 'during-index')).documentId, doc.documentId)
  } finally {
    releaseCandidate.resolve()
  }
  let matches
  for (let attempt = 0; !matches && attempt < 25; attempt++) {
    try {
      matches = await vision.search({ query: 'alpha', assetId: image.assetId, limit: 2 })
    } catch (error) {
      assert.match(error.message, /Qwen/u)
      await wait(50)
    }
  }
  assert(matches, 'The Qwen index never became active')
  assert(matches.every(item => item.documentId === doc.documentId))
  assert.equal(matches.length, 2)
  const messages = calls.flatMap(call => call.body.messages ?? []).flatMap(message => message.content)
  assert(messages.some(item => item.type === 'image_url' && item.image_url.url === 'data:image/png;base64,' + png.toString('base64')))
  assert(messages.some(item => item.type === 'video_url' && item.video_url.url === 'data:video/mp4;base64,' + mp4.toString('base64')))
  assert.deepEqual((await vision.get(doc.documentId)).attachments, [
    { ...image, description: 'image' }, video,
  ])
  assert.deepEqual(Buffer.from((await vision.readMedia(video.assetId)).bytes), mp4)
})

test('keeps the previous index and persisted safe options when a candidate fails', async t => {
  const { profile, bge, qwen, open, result } = await fixture(t)
  const original = open({ embedding: { ...bge.embedding, apiKey: 'do-not-persist-this' } })
  const doc = await original.write({ title: 'alpha', body: 'alpha' }, 'original-document')
  await original.close()
  result.failedModel = qwen.embedding.model
  const report = Promise.withResolvers()
  const replacement = open(qwen, profile, report.resolve)
  assert.match(await report.promise, /重建索引失败/u)
  assert.equal((await replacement.search({ query: 'alpha', limit: 1 }))[0].documentId, doc.documentId)
  const database = new DatabaseSync(join(profile, 'data', 'dsh-prism', 'memory.sqlite'))
  try {
    const { options } = database.prepare('SELECT options FROM active_index').get()
    assert.equal(JSON.parse(options).model, bge.embedding.model)
    assert(!options.includes('do-not-persist-this'))
  } finally {
    database.close()
  }
})

test('rejects invalid embeddings and retries without exposing a partial document', async t => {
  const { bge, result, open } = await fixture(t)
  const memory = open(bge)
  const draft = { title: 'alpha', body: 'alpha' }
  result.broken = true
  await assert.rejects(memory.write(draft, 'retry-key'), /维度/u)
  result.broken = false
  assert.deepEqual(await memory.search({ query: 'alpha' }), [])
  const { documentId } = await memory.write(draft, 'retry-key')
  assert.equal((await memory.get(documentId)).body, draft.body)
  await assert.rejects(memory.stageMedia({ bytes: Uint8Array.of(1), mimeType: 'text/html' }), /不受支持/u)
  await assert.rejects(memory.stageMedia({ bytes: png, mimeType: 'image/jpeg' }), /不受支持/u)
  await assert.rejects(memory.stageMedia({ bytes: Buffer.from('<!doctype html>'), mimeType: 'image/png' }), /不受支持/u)
  await assert.rejects(memory.search({ limit: 1 }), /需要文本或媒体/u)
})

test('migrates the profile database and retains acknowledged chat media past staging expiry', async t => {
  const { profile, bge, open } = await fixture(t)
  const first = open(bge)
  const input = { bytes: png, mimeType: 'image/png' }
  const { assetId: retained } = await first.stageMedia(input)
  const { assetId: expires } = await first.stageMedia(input)
  await first.close()

  const path = join(profile, 'data', 'dsh-prism', 'memory.sqlite')
  const legacy = new DatabaseSync(path)
  try {
    legacy.exec('ALTER TABLE media DROP COLUMN pinned; PRAGMA user_version = 1')
  } finally {
    legacy.close()
  }
  const migrated = open(bge)
  await assert.rejects(migrated.pinMedia([expires, 'missing-asset']), /过期/u)
  await migrated.pinMedia([retained, retained])
  await migrated.pinMedia([retained])
  await migrated.close()

  const clock = new DatabaseSync(path)
  try {
    clock.prepare('UPDATE media SET staged_at = ?').run(Date.now() - 2 * 24 * 60 * 60 * 1000)
  } finally {
    clock.close()
  }
  const reopened = open(bge)
  await reopened.stageMedia(input)
  assert.deepEqual(Buffer.from((await reopened.readMedia(retained)).bytes), png)
  assert.equal(await reopened.readMedia(expires), undefined)
  await assert.rejects(reopened.pinMedia([expires]), /过期/u)
})

test('rejects a write when authorization is revoked during embedding', async t => {
  const { bge, result, open } = await fixture(t)
  const memory = open(bge)
  const draft = { title: 'alpha', body: 'alpha' }
  let allowed = true
  result.afterEmbedding = () => { allowed = false }
  const authorize = () => {
    if (!allowed) {
      throw new Error('authorization revoked')
    }
  }
  await assert.rejects(memory.write(draft, 'revoked-task', authorize), /authorization revoked/u)
  result.afterEmbedding = undefined
  assert.deepEqual(await memory.search({ query: 'alpha' }), [])
  allowed = true
  const { documentId } = await memory.write(draft, 'revoked-task', authorize)
  assert.equal((await memory.get(documentId)).title, draft.title)
  allowed = false
  await assert.rejects(memory.write(draft, 'revoked-task', authorize), /authorization revoked/u)
})

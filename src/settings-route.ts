import type { IncomingMessage } from 'node:http'
import type { WebAPI } from '@dsh-plugin/dsh-loader'
import { BindingRequestError, type MemoryBinding } from './binding.js'

async function readRequest(req: IncomingMessage): Promise<unknown> {
  if (req.headers['content-type']?.split(';')[0]?.trim().toLowerCase() !== 'application/json') {
    throw new BindingRequestError(400, '请提交 JSON 格式的记忆模型设置')
  }
  const parts: Buffer[] = []
  let size = 0
  for await (const part of req.iterator({ destroyOnReturn: false })) {
    const bytes = Buffer.isBuffer(part) ? part : Buffer.from(part)
    size += bytes.length
    if (size > 16 * 1024) {
      req.resume()
      throw new BindingRequestError(400, '记忆模型设置超过请求大小上限')
    }
    parts.push(bytes)
  }
  try {
    return JSON.parse(Buffer.concat(parts).toString('utf8'))
  } catch {
    throw new BindingRequestError(400, '记忆模型设置不是有效的 JSON')
  }
}

export function memorySettingsRoute(binding: MemoryBinding): Parameters<WebAPI['registerOwner']>[0] {
  return { kind: 'exact', path: '/api/prism/memory/settings', async handler(req, res) {
    res.setHeader('Cache-Control', 'private, no-store')
    res.setHeader('Content-Type', 'application/json; charset=utf-8')
    res.setHeader('X-Content-Type-Options', 'nosniff')
    res.setHeader('Referrer-Policy', 'no-referrer')
    const rawOrigin = req.headers.origin
    const origin = typeof rawOrigin === 'string' ? URL.parse(rawOrigin) : null
    const sameOrigin = origin && ['https:', 'http:'].includes(origin.protocol) && origin.host === req.headers.host
    if (req.headers['sec-fetch-site'] === 'cross-site' || (rawOrigin && !sameOrigin) ||
        (req.method === 'POST' && (!sameOrigin || req.headers['x-prism-request'] !== '1'))) {
      res.statusCode = 403
      res.end(JSON.stringify({ error: '记忆设置仅接受来自 DSH 页面的同源请求' }))
      return
    }
    if (req.method === 'GET') {
      res.end(JSON.stringify(binding.status()))
      return
    }
    if (req.method !== 'POST') {
      res.writeHead(405, { Allow: 'GET, POST' }).end()
      return
    }
    try {
      const next = binding.start(await readRequest(req))
      res.statusCode = 202
      res.end(JSON.stringify(next))
    } catch (error) {
      res.statusCode = error instanceof BindingRequestError ? error.status : 503
      res.end(JSON.stringify({ error: error instanceof BindingRequestError ? error.message : '暂时无法提交记忆设置' }))
    }
  } }
}

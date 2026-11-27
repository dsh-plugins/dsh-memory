import { execFile } from 'node:child_process'
import { mkdtemp, rm, rmdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { isAbsolute, join } from 'node:path'
import { performance } from 'node:perf_hooks'
import { promisify } from 'node:util'
import type { MediaUpload } from './contracts.js'

export interface VideoOptions {
  ffmpegPath?: string
  ffprobePath?: string
  signal?: AbortSignal
}

export interface VideoSegment {
  startMs: number
  endMs: number
  dataUrl: string
  frameCount: number
}

export const videoSamplingRevision = 'jpeg-1fps-12s-512-v1'

const segmentMs = 12_000
const maxDurationMs = 180_000
const maxInputBytes = 16 * 1024 * 1024
const maxDecodeMs = 90_000
const boundary = 'prism-frame'
const firstBoundary = Buffer.from(`--${boundary}\r\n`)
const nextBoundary = Buffer.from(`\r\n--${boundary}\r\n`)
const run = promisify(execFile)

const environment = Object.fromEntries(Object.entries(process.env).filter(([key]) => /^(path|systemroot|windir)$/i.test(key)))
const inputFormats = new Map([
  ['video/mp4', { extension: 'mp4', flags: ['-f', 'mov', '-enable_drefs', '0'] }],
  ['video/quicktime', { extension: 'mov', flags: ['-f', 'mov', '-enable_drefs', '0'] }],
  ['video/webm', { extension: 'webm', flags: ['-f', 'matroska'] }],
])

function executable(tool: 'ffmpeg' | 'ffprobe', path?: string): string {
  if (path === undefined) return tool
  if (!isAbsolute(path)) throw new Error(`${tool} 路径必须为绝对路径`)
  return path
}

async function runVideoTool(
  tool: string, args: string[], cwd: string, timeout: number, maxBuffer: number, signal?: AbortSignal,
): Promise<Buffer> {
  try {
    const { stdout } = await run(tool, args, {
      cwd, env: environment, encoding: 'buffer', shell: false, windowsHide: true,
      timeout, maxBuffer, signal, killSignal: 'SIGKILL',
    })
    return stdout
  } catch (error) {
    signal?.throwIfAborted()
    const failure = error as { code?: string; killed?: boolean }
    if (failure.code === 'ENOENT') throw new Error('找不到 ffmpeg 或 ffprobe，请将其加入 PATH 或配置绝对路径')
    if (failure.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER') throw new Error('视频抽帧输出超出大小限制')
    if (failure.killed) throw new Error('视频解析或抽帧超时')
    throw new Error('无法解析或抽取视频帧，请检查文件格式与 FFmpeg 安装')
  }
}

function videoDuration(probeOutput: Buffer): number {
  const probe = JSON.parse(probeOutput.toString('utf8')) as {
    streams?: { width?: number; height?: number; duration?: string }[]
    format?: { duration?: string }
  }
  const stream = probe.streams?.[0]
  const dimensions = [Number(stream?.width), Number(stream?.height)]
  if (!dimensions.every(size => Number.isInteger(size) && size >= 2 && size <= 4096)) {
    throw new Error('视频须包含尺寸在 2 至 4096 像素内的画面')
  }

  const duration = [stream?.duration, probe.format?.duration].map(Number).find(value => Number.isFinite(value) && value > 0)
  if (duration === undefined) throw new Error('无法确定视频时长')
  const durationMs = Math.ceil(duration * 1000)
  if (durationMs > maxDurationMs) throw new Error('视频时长不得超过 180 秒')
  return durationMs
}

export function decodeVideoFrames(output: Buffer, maxFrames: number): Buffer[] {
  const invalid = () => new Error('视频抽帧输出格式无效')
  if (!output.subarray(0, firstBoundary.length).equals(firstBoundary)) throw invalid()
  const frames: Buffer[] = []
  let offset = firstBoundary.length

  while (offset < output.length) {
    const header = /^Content-type: image\/jpeg\r\nContent-length: ([1-9]\d*)\r\n\r\n/i.exec(
      output.toString('ascii', offset, offset + 96),
    )
    if (!header || frames.length === maxFrames) throw invalid()
    const start = offset + header[0].length
    const end = start + Number(header[1])
    const frame = output.subarray(start, end)
    if (end + nextBoundary.length > output.length || !output.subarray(end, end + nextBoundary.length).equals(nextBoundary)) {
      throw invalid()
    }
    if (frame.length < 4 || frame.readUInt16BE(0) !== 0xffd8 || frame.readUInt16BE(frame.length - 2) !== 0xffd9) throw invalid()
    frames.push(frame)
    offset = end + nextBoundary.length
  }

  return frames
}

/** 将本地视频按时间分段，每秒取一张 JPEG 帧供视觉嵌入模型读取。 */
export async function* sampleVideo(
  media: Pick<MediaUpload, 'bytes' | 'mimeType'>,
  options: VideoOptions = {},
): AsyncGenerator<VideoSegment> {
  options.signal?.throwIfAborted()
  const format = inputFormats.get(media.mimeType)
  if (!format) throw new Error('不支持此视频格式')
  if (!media.bytes.byteLength || media.bytes.byteLength > maxInputBytes) throw new Error('视频须在 16 MiB 以内且不能为空')

  const ffprobe = executable('ffprobe', options.ffprobePath)
  const ffmpeg = executable('ffmpeg', options.ffmpegPath)
  const cwd = await mkdtemp(join(tmpdir(), 'dsh-prism-video-'))
  const input = join(cwd, `input.${format.extension}`)
  const common = ['-v', 'error', '-max_alloc', '67108864', '-protocol_whitelist', 'file,pipe', ...format.flags]

  try {
    await writeFile(input, media.bytes, { flag: 'wx', mode: 0o600, signal: options.signal })
    const probe = await runVideoTool(ffprobe, [
      ...common, '-select_streams', 'V:0', '-show_entries', 'stream=width,height,duration:format=duration',
      '-of', 'json', '-i', input,
    ], cwd, 10_000, 64 * 1024, options.signal)
    const duration = videoDuration(probe)
    let decodeMs = 0

    for (let startMs = 0; startMs < duration; startMs += segmentMs) {
      options.signal?.throwIfAborted()
      const endMs = Math.min(startMs + segmentMs, duration)
      const seconds = (endMs - startMs) / 1000
      const maxFrames = Math.ceil(seconds)
      const started = performance.now()
      const timeout = Math.floor(Math.min(maxDecodeMs - decodeMs, 15_000))
      if (timeout <= 0) throw new Error('视频抽帧已达到总时限')

      const output = await runVideoTool(ffmpeg, [
        '-nostdin', ...common, '-threads', '1', '-filter_threads', '1', '-ss', String(startMs / 1000), '-i', input,
        '-t', String(seconds), '-map', '0:V:0', '-an', '-sn', '-dn', '-map_metadata', '-1',
        '-vf', "fps=1:start_time=0:round=up,scale='min(512,iw)':'min(512,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1",
        '-c:v', 'mjpeg', '-q:v', '5', '-threads', '1', '-frames:v', String(maxFrames),
        '-f', 'mpjpeg', '-boundary_tag', boundary, 'pipe:1',
      ], cwd, timeout, 4 * 1024 * 1024, options.signal)
      decodeMs += performance.now() - started

      const frames = decodeVideoFrames(output, maxFrames)
      if (!frames.length) throw new Error(`视频在 ${startMs / 1000} 秒处没有可用画面`)
      yield {
        startMs,
        endMs,
        dataUrl: `data:video/jpeg;base64,${frames.map(frame => frame.toString('base64')).join(',')}`,
        frameCount: frames.length,
      }
    }
  } finally {
    await rm(input, { force: true })
    await rmdir(cwd)
  }
}

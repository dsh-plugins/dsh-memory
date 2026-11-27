import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtemp, readFile, rm, rmdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { decodeVideoFrames, sampleVideo } from '../dist/video.js'

const ffmpegPath = process.env.PRISM_TEST_FFMPEG
const ffprobePath = process.env.PRISM_TEST_FFPROBE
const ffmpeg = ffmpegPath ?? 'ffmpeg'
const ffprobe = ffprobePath ?? 'ffprobe'
const toolsAvailable = [ffmpeg, ffprobe].every(command => spawnSync(command, ['-version'], { timeout: 5_000 }).status === 0)

async function collect(segments) {
  const result = []
  for await (const segment of segments) result.push(segment)
  return result
}

test('按 multipart 长度读取 JPEG，并拒绝截断帧和过多帧', () => {
  const payload = Buffer.from([0xff, 0xd8, 0xff, 0xd9, 0x30, 0xff, 0xd9])
  const packet = Buffer.concat([
    Buffer.from(`Content-type: image/jpeg\r\nContent-length: ${payload.length}\r\n\r\n`),
    payload,
    Buffer.from('\r\n--prism-frame\r\n'),
  ])
  const output = Buffer.concat([Buffer.from('--prism-frame\r\n'), packet, packet])
  assert.deepEqual(decodeVideoFrames(output, 2), [payload, payload])
  assert.throws(() => decodeVideoFrames(output.subarray(0, -2), 2), /格式无效/)
  assert.throws(() => decodeVideoFrames(output, 1), /格式无效/)
})

test('禁止不支持的输入和相对工具路径，并支持取消', async () => {
  const media = { bytes: new Uint8Array([0]), mimeType: 'video/mp4' }
  await assert.rejects(collect(sampleVideo({ ...media, mimeType: 'video/mpegts' })), /不支持/)
  await assert.rejects(collect(sampleVideo({ ...media, bytes: new Uint8Array(0) })), /不能为空/)
  await assert.rejects(collect(sampleVideo(media, { ffmpegPath: './ffmpeg' })), /绝对路径/)
  const reason = new Error('停止抽帧')
  await assert.rejects(collect(sampleVideo(media, { signal: AbortSignal.abort(reason) })), reason)
})

test('用真实 FFmpeg 将包含短尾段的 MP4 转为带准确时间边界的 JPEG 帧', {
  skip: toolsAvailable ? false : '需要 PATH 或 PRISM_TEST_FFMPEG / PRISM_TEST_FFPROBE 提供 FFmpeg',
}, async () => {
  const root = await mkdtemp(join(tmpdir(), 'dsh-prism-video-test-'))
  const input = join(root, 'sample.mp4')

  try {
    const fixture = spawnSync(ffmpeg, [
      '-nostdin', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=800x600:rate=5',
      '-t', '12.2', '-c:v', 'mpeg4', '-threads', '1', '-pix_fmt', 'yuv420p', input,
    ], { timeout: 10_000 })
    assert.equal(fixture.status, 0, fixture.stderr?.toString())

    const media = { bytes: await readFile(input), mimeType: 'video/mp4' }
    const segments = await collect(sampleVideo(media, { ffmpegPath, ffprobePath }))
    assert.deepEqual(segments.map(({ startMs, endMs, frameCount }) => [startMs, endMs, frameCount]), [[0, 12_000, 12], [12_000, 12_200, 1]])
    const [full, tail] = segments.map(({ dataUrl, frameCount }) => {
      assert.ok(dataUrl.startsWith('data:video/jpeg;base64,'))
      const frames = dataUrl.split(',').slice(1).map(frame => Buffer.from(frame, 'base64'))
      assert.equal(frames.length, frameCount)
      for (const frame of frames) {
        assert.equal(frame.readUInt16BE(0), 0xffd8)
        assert.equal(frame.readUInt16BE(frame.length - 2), 0xffd9)
      }
      return frames
    })
    assert.notDeepEqual(full[0], tail[0])

    const jpeg = spawnSync(ffprobe, [
      '-v', 'error', '-f', 'mjpeg', '-show_entries', 'stream=width,height',
      '-of', 'csv=p=0', '-i', 'pipe:0',
    ], { input: full[0], timeout: 5_000 })
    assert.equal(jpeg.status, 0, jpeg.stderr?.toString())
    assert.equal(jpeg.stdout.toString().trim(), '512,384')
  } finally {
    await rm(input, { force: true })
    await rmdir(root)
  }
})

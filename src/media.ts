function marker(bytes: Uint8Array, offset: number, value: string) {
  return [...value].every((character, index) => bytes[offset + index] === character.charCodeAt(0))
}

function isIsoVideo(bytes: Uint8Array, quickTime: boolean) {
  if (bytes.length < 16) {
    return false
  }
  if (!marker(bytes, 4, 'ftyp')) {
    return quickTime && ['moov', 'mdat', 'wide'].some(atom => marker(bytes, 4, atom))
  }
  const brands = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, false)
  if (brands < 16 || brands > bytes.length || brands > 1024) {
    return false
  }
  const expected = quickTime ? ['qt  '] : ['isom', 'iso2', 'iso4', 'iso5', 'iso6', 'mp41', 'mp42', 'avc1', 'dash', 'M4V ', 'MSNV', 'cmfc', 'cmfs']
  if (expected.some(brand => marker(bytes, 8, brand))) {
    return true
  }
  for (let position = 16; position + 4 <= brands; position += 4) {
    if (expected.some(brand => marker(bytes, position, brand))) {
      return true
    }
  }
  return false
}

/** 确认上传声明匹配图片或视频容器的文件签名。 */
export function matchesMedia(bytes: Uint8Array, mimeType: string): boolean {
  switch (mimeType) {
    case 'image/jpeg':
      return bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff
    case 'image/png':
      return [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a].every((byte, index) => bytes[index] === byte)
    case 'image/gif':
      return marker(bytes, 0, 'GIF87a') || marker(bytes, 0, 'GIF89a')
    case 'image/webp':
      return marker(bytes, 0, 'RIFF') && marker(bytes, 8, 'WEBP') &&
        ['VP8 ', 'VP8L', 'VP8X'].some(chunk => marker(bytes, 12, chunk))
    case 'video/mp4':
      return isIsoVideo(bytes, false)
    case 'video/quicktime':
      return isIsoVideo(bytes, true)
    case 'video/webm':
      return [0x1a, 0x45, 0xdf, 0xa3].every((byte, index) => bytes[index] === byte) &&
        Buffer.from(bytes.subarray(0, 1024)).includes('webm')
    default:
      return false
  }
}

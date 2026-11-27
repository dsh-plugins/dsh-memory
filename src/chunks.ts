const sentences = new Intl.Segmenter(undefined, { granularity: 'sentence' })
const maximumFragmentLength = 16_384

/** 保留原文边界，优先按段落和句子切分超过向量模型预算的内容。 */
export function chunkText(text: string, limit: number, tokens: (text: string) => number): string[] {
  const fits = (value: string) => value.length <= maximumFragmentLength && tokens(value) <= limit
  const result: string[] = []
  let current = ''
  function push(piece: string) {
    if (current.length > 0 && !fits(current + piece)) {
      result.push(current)
      current = ''
    }
    current += piece
  }
  for (const paragraph of text.match(/[^]*?(?:\r?\n[ \t]*\r?\n+|$)/gu) ?? []) {
    if (fits(paragraph)) {
      push(paragraph)
      continue
    }
    for (const { segment } of sentences.segment(paragraph)) {
      if (fits(segment)) {
        push(segment)
        continue
      }
      const points = Array.from(segment)
      for (let index = 0; index < points.length;) {
        let low = 1
        let high = Math.min(points.length - index, maximumFragmentLength)
        let length = 0
        while (low <= high) {
          const middle = Math.floor((low + high) / 2)
          if (fits(points.slice(index, index + middle).join(''))) {
            length = middle
            low = middle + 1
          } else {
            high = middle - 1
          }
        }
        if (length === 0) {
          throw new Error('向量模型的输入预算不足以处理单个字符')
        }
        push(points.slice(index, index + length).join(''))
        index += length
      }
    }
  }
  if (current.length > 0) {
    result.push(current)
  }
  return result
}

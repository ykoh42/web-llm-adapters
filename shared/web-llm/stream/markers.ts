/** Leave a partial control marker buffered until its next chunk arrives. */
export function textEnd(text: string, markers: string[]) {
  if (!markers.length) return text.length
  let longest = 0
  for (const marker of markers) longest = Math.max(longest, marker.length)
  const tail = text.slice(1 - longest)
  // Native control markers contain only one opening '<'.
  const start = tail.lastIndexOf('<')
  if (start === -1) return text.length
  const partial = tail.slice(start)
  return markers.some((marker) => marker.startsWith(partial))
    ? text.length - tail.length + start
    : text.length
}

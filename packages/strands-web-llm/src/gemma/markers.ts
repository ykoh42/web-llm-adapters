/** Leave a partial control marker buffered until its next chunk arrives. */
export function textEnd(text: string, markers: string[]) {
  const longest = Math.max(...markers.map((marker) => marker.length))
  let held = Math.min(text.length, longest - 1)
  while (held && !markers.some((marker) => marker.startsWith(text.slice(-held)))) held--
  return text.length - held
}

export function registerSnapshot(words) {
  if (!Array.isArray(words)) throw new TypeError('words must be an array');
  return {widthBits: words.length * 64, words: words.map(v=>BigInt(v).toString())};
}
export function workloadEnvelope(kind, payload) {
  return {schema:'chimera.cpu.workload',version:1,kind,payload};
}

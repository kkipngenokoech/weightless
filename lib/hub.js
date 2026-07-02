/* Hugging Face Hub loading — header-only via HTTP Range requests.
 * A 500 GB sharded model costs a few hundred KB of traffic: we fetch each
 * shard's 8-byte length prefix + JSON header, never the weights.
 * vscode-free (plain fetch), so it's unit-testable. */
const { tensorsFromHeader } = require('./safetensors');

const BASE = 'https://huggingface.co';
const MAX_HEADER_BYTES = 512 * 1024 * 1024;

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fetchRange(url, start, end, token) {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}`, ...authHeaders(token) }, redirect: 'follow' });
  if (!res.ok) throw httpError(res, url);
  return Buffer.from(await res.arrayBuffer());
}

async function fetchJson(url, token) {
  const res = await fetch(url, { headers: authHeaders(token), redirect: 'follow' });
  if (!res.ok) throw httpError(res, url);
  return res.json();
}

function httpError(res, url) {
  const hint = res.status === 401 || res.status === 403
    ? ' (model may be gated/private — accept its license on huggingface.co or set a token)'
    : res.status === 404 ? ' (repo or file not found — check the model id)' : '';
  return new Error(`HTTP ${res.status}${hint} — ${url}`);
}

/** Read only the safetensors header of a remote file. */
async function readRemoteHeader(url, token) {
  const lenBuf = await fetchRange(url, 0, 7, token);
  const headerLen = Number(lenBuf.readBigUInt64LE(0));
  if (!Number.isFinite(headerLen) || headerLen <= 0 || headerLen > MAX_HEADER_BYTES) {
    throw new Error(`Invalid safetensors header length (${headerLen}) — is this a .safetensors file?`);
  }
  const hdrBuf = await fetchRange(url, 8, 8 + headerLen - 1, token);
  return JSON.parse(hdrBuf.toString('utf8'));
}

/** Load a Hub model's structure (same shape as safetensors.loadModel). */
async function loadFromHub(id, opts = {}) {
  const revision = opts.revision || 'main';
  const token = opts.token || process.env.HF_TOKEN || null;
  const resolve = (f) => `${BASE}/${id}/resolve/${encodeURIComponent(revision)}/${f}`;

  const info = await fetchJson(`${BASE}/api/models/${encodeURIComponent(id)}?revision=${encodeURIComponent(revision)}`, token);
  const files = (info.siblings || []).map((s) => s.rfilename);

  const indexFile = files.find((f) => f === 'model.safetensors.index.json')
    || files.find((f) => f.endsWith('.safetensors.index.json') && !f.includes('/'));
  const stFiles = files.filter((f) => f.endsWith('.safetensors') && !f.includes('/'));

  let tensors = [], meta = null, shardFiles = [], fileSize = 0, sharded = false;
  if (indexFile) {
    const idx = await fetchJson(resolve(indexFile), token);
    shardFiles = Array.from(new Set(Object.values(idx.weight_map || {})));
    sharded = true;
    if (idx.metadata && idx.metadata.total_size) fileSize = idx.metadata.total_size;
    const headers = await Promise.all(shardFiles.map((f) => readRemoteHeader(resolve(f), token)));
    headers.forEach((hd, i) => {
      if (!meta && hd.__metadata__) meta = hd.__metadata__;
      tensors = tensors.concat(tensorsFromHeader(hd, shardFiles[i]));
    });
  } else {
    const f = stFiles.find((n) => n === 'model.safetensors') || stFiles[0];
    if (!f) throw new Error(`No .safetensors file found in ${id} — it may only ship .bin/.gguf weights`);
    shardFiles = [f];
    const hd = await readRemoteHeader(resolve(f), token);
    meta = hd.__metadata__ || null;
    tensors = tensorsFromHeader(hd, f);
  }
  if (!fileSize) fileSize = tensors.reduce((a, t) => a + t.bytes, 0);

  let config = null;
  if (files.includes('config.json')) {
    try { config = await fetchJson(resolve('config.json'), token); } catch (_) { /* optional */ }
  }
  const extras = {};
  const SIDE = [['Generation', 'generation_config.json'], ['Adapter', 'adapter_config.json'], ['Tokenizer', 'tokenizer_config.json']];
  await Promise.all(SIDE.map(async ([label, name]) => {
    if (!files.includes(name)) return;
    try { extras[label] = { file: name, data: await fetchJson(resolve(name), token) }; } catch (_) { /* optional */ }
  }));

  return { tensors, fileSize, meta, config, extras, sharded, shardFiles };
}

module.exports = { loadFromHub, readRemoteHeader };

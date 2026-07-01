/* Pure safetensors header parsing — no vscode dependency, so it's unit-testable. */
const fs = require('fs');
const path = require('path');

const MAX_HEADER_BYTES = 512 * 1024 * 1024; // sanity bound; real headers are KB–MB

/** Read ONLY the header (8-byte LE length + JSON), never the weight bytes. */
async function readHeaderOnly(fsPath) {
  const fh = await fs.promises.open(fsPath, 'r');
  try {
    const lenBuf = Buffer.alloc(8);
    await fh.read(lenBuf, 0, 8, 0);
    const headerLen = Number(lenBuf.readBigUInt64LE(0));
    if (!Number.isFinite(headerLen) || headerLen <= 0 || headerLen > MAX_HEADER_BYTES) {
      throw new Error(`Invalid safetensors header length (${headerLen}). Is this a .safetensors file?`);
    }
    const hdrBuf = Buffer.alloc(headerLen);
    await fh.read(hdrBuf, 0, headerLen, 8);
    const header = JSON.parse(hdrBuf.toString('utf8'));
    const { size } = await fh.stat();
    return { header, fileSize: size };
  } finally {
    await fh.close();
  }
}

function tensorsFromHeader(header, shard) {
  const out = [];
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__' || !info || !info.shape) continue;
    const shape = info.shape;
    const params = shape.reduce((a, b) => a * b, 1);
    const bytes = info.data_offsets ? info.data_offsets[1] - info.data_offsets[0] : 0;
    out.push({ name, dtype: info.dtype, shape, params, bytes, shard });
  }
  return out;
}

/** Load a single file, or aggregate a sharded model via its index.json — header-only. */
async function loadModel(fsPath) {
  const dir = path.dirname(fsPath);
  const base = path.basename(fsPath);

  let indexPath = null;
  for (const cand of [base + '.index.json', 'model.safetensors.index.json']) {
    const p = path.join(dir, cand);
    if (fs.existsSync(p)) { indexPath = p; break; }
  }

  let tensors = [];
  let fileSize = 0;
  let meta = null;
  let shardFiles = [base];

  if (indexPath) {
    const idx = JSON.parse(await fs.promises.readFile(indexPath, 'utf8'));
    shardFiles = Array.from(new Set(Object.values(idx.weight_map || {})));
    for (const f of shardFiles) {
      const p = path.join(dir, f);
      if (!fs.existsSync(p)) continue;
      const { header, fileSize: fsz } = await readHeaderOnly(p);
      if (!meta && header.__metadata__) meta = header.__metadata__;
      tensors = tensors.concat(tensorsFromHeader(header, f));
      fileSize += fsz;
    }
  } else {
    const { header, fileSize: fsz } = await readHeaderOnly(fsPath);
    meta = header.__metadata__ || null;
    tensors = tensorsFromHeader(header, base);
    fileSize = fsz;
  }

  let config = null;
  const cfgPath = path.join(dir, 'config.json');
  if (fs.existsSync(cfgPath)) {
    try { config = JSON.parse(await fs.promises.readFile(cfgPath, 'utf8')); } catch (_) { /* ignore */ }
  }

  return { tensors, fileSize, meta, config, sharded: !!indexPath, shardFiles };
}

module.exports = { readHeaderOnly, tensorsFromHeader, loadModel };

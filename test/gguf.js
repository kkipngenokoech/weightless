/* GGUF parser smoke test — writes a tiny llama-style GGUF and parses it back. */
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { parseGGUF, loadGGUFModel } = require('../lib/gguf');

// ---- tiny GGUF writer (spec-conformant subset) ----
function str(s) {
  const b = Buffer.from(s, 'utf8');
  const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(b.length));
  return Buffer.concat([len, b]);
}
function u32(v) { const b = Buffer.alloc(4); b.writeUInt32LE(v); return b; }
function u64(v) { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); return b; }
function kvStr(key, val) { return Buffer.concat([str(key), u32(8), str(val)]); }
function kvU32(key, val) { return Buffer.concat([str(key), u32(4), u32(val)]); }

const ALIGN = 32;
const tensors = [
  { name: 'token_embd.weight', dims: [64, 100], type: 0 },        // F32, stored fastest-first => shape [100,64]
  { name: 'blk.0.attn_q.weight', dims: [64, 64], type: 1 },       // F16
  { name: 'blk.0.ffn_gate.weight', dims: [64, 128], type: 1 },    // F16
  { name: 'output_norm.weight', dims: [64], type: 0 },            // F32
];
// data offsets (relative to data section), aligned
let off = 0;
const sizes = { 0: 4, 1: 2 }; // bytes/elt for F32, F16
for (const t of tensors) {
  t.offset = off;
  const n = t.dims.reduce((a, b) => a * b, 1);
  off = Math.ceil((off + n * sizes[t.type]) / ALIGN) * ALIGN;
}

const kvs = [
  kvStr('general.architecture', 'llama'),
  kvStr('general.name', 'tiny-test'),
  kvU32('llama.block_count', 1),
  kvU32('llama.embedding_length', 64),
  kvU32('llama.attention.head_count', 4),
  kvU32('llama.attention.head_count_kv', 2),
  kvU32('llama.context_length', 2048),
];
const infos = tensors.map((t) => Buffer.concat([
  str(t.name), u32(t.dims.length), ...t.dims.map(u64), u32(t.type), u64(t.offset),
]));
let header = Buffer.concat([
  u32(0x46554747), u32(3), u64(tensors.length), u64(kvs.length),
  ...kvs, ...infos,
]);
const dataStart = Math.ceil(header.length / ALIGN) * ALIGN;
const file = Buffer.concat([header, Buffer.alloc(dataStart - header.length + off, 0xab)]);

const tmp = path.join(__dirname, 'fixtures', 'tiny.gguf');
fs.writeFileSync(tmp, file);

// ---- assertions ----
(async () => {
  const { kv, tensors: ts } = parseGGUF(file, file.length);
  assert.strictEqual(kv['general.architecture'], 'llama');
  assert.strictEqual(ts.length, 4);
  const emb = ts.find((t) => t.name === 'token_embd.weight');
  assert.deepStrictEqual(emb.shape, [100, 64], 'dims must be reversed to torch order');
  assert.strictEqual(emb.dtype, 'F32');
  assert.strictEqual(emb.params, 6400);
  const q = ts.find((t) => t.name === 'blk.0.attn_q.weight');
  assert.strictEqual(q.dtype, 'F16');

  const model = await loadGGUFModel(tmp);
  assert.strictEqual(model.config.model_type, 'llama');
  assert.strictEqual(model.config.num_hidden_layers, 1);
  assert.strictEqual(model.config.num_key_value_heads, 2);
  assert.strictEqual(model.config.max_position_embeddings, 2048);
  assert.ok(model.tensors.every((t) => t.bytes > 0), 'bytes from offset deltas');
  assert.strictEqual(model.fileSize, file.length);

  console.log(`gguf: ${model.tensors.length} tensors · arch=${model.config.model_type} · ctx=${model.config.max_position_embeddings}`);
  console.log('GGUF ASSERTIONS PASSED ✅');
})().catch((e) => { console.error(e); process.exit(1); });

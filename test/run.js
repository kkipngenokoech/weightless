/* Smoke test: build a synthetic .safetensors (header-only, tiny) mimicking a critic's
 * structure, then run the extension's real header reader over it and assert. */
const fs = require('fs');
const path = require('path');
const parser = require('../lib/safetensors');

const DT_BYTES = { F32: 4, BF16: 2, F16: 2, I64: 8, U8: 1 };

function build() {
  const specs = [];
  for (let i = 0; i < 4; i++) {
    specs.push([`vision_tower.vision_model.encoder.layers.${i}.self_attn.q_proj.weight`, [1152, 1152], 'BF16']);
    specs.push([`vision_tower.vision_model.encoder.layers.${i}.self_attn.q_proj.bias`, [1152], 'BF16']);
    specs.push([`vision_tower.vision_model.encoder.layers.${i}.mlp.fc1.weight`, [4304, 1152], 'BF16']);
    specs.push([`vision_tower.vision_model.encoder.layers.${i}.layer_norm1.weight`, [1152], 'BF16']);
  }
  specs.push(['vision_tower.vision_model.embeddings.patch_embedding.weight', [1152, 3, 14, 14], 'BF16']);
  for (let i = 0; i < 4; i++) {
    specs.push([`language_model.model.layers.${i}.self_attn.k_proj.weight`, [256, 640], 'BF16']);
    specs.push([`language_model.model.layers.${i}.mlp.gate_proj.weight`, [2048, 640], 'BF16']);
    specs.push([`language_model.model.layers.${i}.input_layernorm.weight`, [640], 'BF16']);
  }
  specs.push(['language_model.model.embed_tokens.weight', [1000, 640], 'BF16']);
  specs.push(['multi_modal_projector.weight', [640, 1152], 'F32']);
  specs.push(['multi_modal_projector.bias', [640], 'F32']);
  specs.push(['value_head.0.weight', [640, 640], 'F32']);
  specs.push(['value_head.0.bias', [640], 'F32']);
  specs.push(['value_head.2.weight', [201, 640], 'F32']);
  specs.push(['value_head.2.bias', [201], 'F32']);
  specs.push(['value_bin_support', [201], 'F32']);

  const header = { __metadata__: { format: 'pt', producer: 'safetensors-viewer-test' } };
  let off = 0;
  for (const [name, shape, dt] of specs) {
    const n = shape.reduce((a, b) => a * b, 1) * DT_BYTES[dt];
    header[name] = { dtype: dt, shape, data_offsets: [off, off + n] };
    off += n;
  }
  const hjson = Buffer.from(JSON.stringify(header), 'utf8');
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(hjson.length));
  // header-only fixture (the reader never touches weight bytes)
  return { buf: Buffer.concat([len, hjson]), count: specs.length };
}

(async () => {
  const dir = path.join(__dirname, 'fixtures');
  fs.mkdirSync(dir, { recursive: true });
  const p = path.join(dir, 'tiny-critic.safetensors');
  const { buf, count } = build();
  fs.writeFileSync(p, buf);

  const model = await parser.loadModel(p);
  const total = model.tensors.reduce((a, t) => a + t.params, 0);
  const tops = [...new Set(model.tensors.map((t) => t.name.split('.')[0]))];
  const vhead = model.tensors.find((t) => t.name === 'value_head.2.weight');

  console.log(`tensors=${model.tensors.length}  params=${total}  size=${model.fileSize}B`);
  console.log(`top modules: ${tops.join(', ')}`);
  console.log(`meta: ${JSON.stringify(model.meta)}`);

  const assert = (c, m) => { if (!c) { console.error('FAIL:', m); process.exit(1); } };
  assert(model.tensors.length === count, `expected ${count} tensors, got ${model.tensors.length}`);
  assert(total > 0, 'params should be > 0');
  ['vision_tower', 'language_model', 'multi_modal_projector', 'value_head'].forEach(
    (m) => assert(tops.includes(m), `missing top module ${m}`));
  assert(vhead && vhead.shape.join(',') === '201,640', 'value_head.2.weight shape');
  assert(model.meta && model.meta.producer === 'safetensors-viewer-test', 'metadata parsed');
  console.log('\nALL ASSERTIONS PASSED ✅');
})().catch((e) => { console.error(e); process.exit(1); });

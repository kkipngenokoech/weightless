/* Smoke test for media/anatomy.js: synthetic headers shaped like a Llama-style
 * decoder and a mixture-of-experts model; asserts the facts the Explain view shows. */
const assert = require('assert');
require('../media/anatomy');

const T = (name, shape, dtype = 'BF16') => ({ name, shape, dtype, params: shape.reduce((a, b) => a * b, 1), bytes: shape.reduce((a, b) => a * b, 2) });
function decoder(layers, moe) {
  const t = [T('model.embed_tokens.weight', [1000, 64]), T('lm_head.weight', [1000, 64]), T('model.norm.weight', [64])];
  for (let i = 0; i < layers; i++) {
    const p = `model.layers.${i}.`;
    t.push(T(p + 'input_layernorm.weight', [64]), T(p + 'self_attn.q_proj.weight', [64, 64]), T(p + 'self_attn.k_proj.weight', [16, 64]));
    if (moe) {
      t.push(T(p + 'mlp.gate.weight', [4, 64]));
      for (let e = 0; e < 4; e++) t.push(T(p + `mlp.experts.${e}.up_proj.weight`, [32, 64]), T(p + `mlp.experts.${e}.down_proj.weight`, [64, 32]));
    } else t.push(T(p + 'mlp.up_proj.weight', [256, 64]), T(p + 'mlp.down_proj.weight', [64, 256]));
  }
  return { tensors: t, fileSize: 0, config: moe ? { num_experts: 4, num_experts_per_tok: 1 } : null };
}

const a = WLA.analyze(decoder(6));
assert.strictEqual(a.kind.key, 'llm');
assert.strictEqual(a.stacks.length, 1);
assert.strictEqual(a.stacks[0].n, 6);
assert.strictEqual(a.stacks[0].hidden, 64);
assert.strictEqual(a.stacks[0].ffn, 256);
assert.strictEqual(a.vocab, 1000);
assert.deepStrictEqual(a.output, { kind: 'next-token', tied: false, vocab: 1000 });
assert.strictEqual(a.where.embed + a.where.head + a.where.attn + a.where.mlp + a.where.other, a.total);

const m = WLA.analyze(decoder(4, true));
assert.strictEqual(m.kind.key, 'moe');
assert.strictEqual(m.stacks[0].experts, 4);
// 1 of 4 experts runs per token → 3/4 of every block's routed expert params are idle
const routed = 4 * 4 * (32 * 64 * 2);
assert.strictEqual(m.active, m.total - routed * 3 / 4);
const sil = WLA.silhouette(m);
assert.strictEqual(sil.stacks[0].blocks[0][2], 1, 'expert blocks are flagged for hatching');

// GPT-2's causal-mask buffer is not a parameter
assert.strictEqual(WLA.realParams(T('h.0.attn.bias', [1, 1, 1024, 1024], 'F32')), 0);
// BatchNorm keeps running statistics alongside its learned scale and bias.
for (const name of ['bn.running_mean', 'bn.running_var', 'bn.num_batches_tracked']) {
  assert.strictEqual(WLA.realParams(T(name, [64])), 0);
}
assert.strictEqual(WLA.realParams(T('bn.weight', [64])), 64);
assert.strictEqual(WLA.realParams(T('bn.bias', [64])), 64);
console.log('ANATOMY ASSERTIONS PASSED ✅');

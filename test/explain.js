/* UI regression checks using a DOM, without launching a browser.
 * npm install --prefix /tmp/weightless-test --no-save --ignore-scripts jsdom@26.1.0
 * NODE_PATH=/tmp/weightless-test/node_modules node test/explain.js
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<!doctype html><html><body><main></main></body></html>', { runScripts: 'outside-only' });
const { window } = dom;
for (const file of ['anatomy.js', 'gallery.js', 'explain.js']) {
  window.eval(fs.readFileSync(path.join(__dirname, '../media', file), 'utf8'));
}
window.scrollTo = function () {};
const document = window.document;
const mount = document.querySelector('main');
const fire = (element, name) => element.dispatchEvent(new window.Event(name, { bubbles: true }));
function change(select, value) { select.value = value; fire(select, 'change'); }
const opened = [];
window.WLX.gallery(mount, { onOpen: (id) => opened.push(id) });
const allCards = [...mount.querySelectorAll('.wlg-card')];
const visibleCards = () => allCards.filter((c) => !c.hidden);
const search = mount.querySelector('[type="search"]');
search.value = 'whisper'; fire(search, 'input');
assert.equal(visibleCards().length, 1);
visibleCards()[0].click();
assert.deepEqual(opened, ['openai/whisper-small']);
search.value = 'missing-model'; fire(search, 'input');
assert.equal(visibleCards().length, 0);
assert.equal(mount.querySelector('.wl-empty').hidden, false);
mount.querySelector('.wl-empty button').click();
assert.equal(visibleCards().length, allCards.length);
const category = [...mount.querySelectorAll('.wl-chip')].find((b) => b.textContent === 'Listen');
category.click();
assert.equal(visibleCards().length, 1);
mount.querySelector('.wl-filter-chips button').click();
change(mount.querySelector('[aria-label="Sort models"]'), 'large');
for (const grid of mount.querySelectorAll('.wlg-grid')) {
  const sizes = [...grid.children].map((c) => Number(c.dataset.total));
  assert.deepEqual(sizes, [...sizes].sort((a, b) => b - a));
}

const tensor = (name, shape) => ({ name, shape, dtype: 'BF16', params: shape.reduce((a, b) => a * b, 1), bytes: shape.reduce((a, b) => a * b, 2) });
function modelFixture(moe = false) {
  const tensors = [tensor('model.embed_tokens.weight', [1000, 64]), tensor('lm_head.weight', [1000, 64])];
  for (let i = 0; i < 3; i++) {
    const prefix = `model.layers.${i}.`;
    tensors.push(tensor(prefix + 'self_attn.q_proj.weight', [64, 64]), tensor(prefix + 'input_layernorm.weight', [64]));
    if (moe) {
      for (let e = 0; e < 4; e++) tensors.push(tensor(prefix + `mlp.experts.${e}.up_proj.weight`, [256, 64]));
    } else tensors.push(tensor(prefix + 'mlp.up_proj.weight', [256, 64]));
  }
  return { tensors, fileName: 'test/example', fileSize: 1, config: moe ? { num_experts: 4, num_experts_per_tok: 1 } : null, extras: {} };
}
function render(model) {
  const anatomy = window.WLA.analyze(model);
  mount.replaceChildren(window.WLX.explain(model, anatomy));
  return anatomy;
}
for (const moe of [false, true]) {
  const model = modelFixture(moe);
  render(model);
  const tabs = [...mount.querySelectorAll('[role="tab"]')];
  assert.equal(mount.querySelectorAll('[role="tabpanel"]:not([hidden])').length, 1);
  tabs[1].click();
  assert.equal(tabs[1].getAttribute('aria-selected'), 'true');
  assert.equal(mount.querySelector('#wl-anatomy').hidden, false);
  tabs[1].dispatchEvent(new window.KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
  assert.equal(tabs[2].getAttribute('aria-selected'), 'true');
  tabs[2].dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Home', bubbles: true }));
  assert.equal(tabs[0].getAttribute('aria-selected'), 'true');
  const paper = mount.querySelector('.wl-paper-diagram');
  assert.ok(paper);
  assert.match(paper.textContent, /Causal self-attention/);
  assert.match(paper.textContent, /× 3/);
  assert.equal(paper.querySelectorAll('.pd-sum').length, 2);
  const architectureStyle = mount.querySelector('[aria-label="Architecture style"]');
  change(architectureStyle, 'layers');
  assert.equal(mount.querySelector('.wlx-plate-frame').hidden, false);
  assert.equal(mount.querySelector('.wl-paper-frame').hidden, true);
  change(architectureStyle, 'paper');
  const attentionNode = paper.querySelector('.pd-attention');
  attentionNode.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  assert.equal(attentionNode.classList.contains('selected'), true);
  const steps = [...mount.querySelectorAll('.wlx-stage')];
  assert.equal(steps.filter((s) => !s.querySelector('.wlx-stage-b').hidden).length, 1);
  mount.querySelector('.wl-stage-trigger').click();
  mount.querySelector('.wl-journey-controls button:last-child').click();
  assert.match(mount.querySelector('.wl-journey-status').textContent, /Step 2/);
  const stack = mount.querySelector('.pl-stack');
  stack.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  assert.equal(stack.classList.contains('lit'), true);
  const slider = mount.querySelector('[aria-label="Block to inspect"]');
  slider.value = '2'; fire(slider, 'input');
  assert.match(mount.querySelector('.wl-block-info').textContent, /Block 2/);
  assert.equal(mount.querySelector('[data-block="0:2"]').classList.contains('block-selected'), true);
  mount.querySelector('.wl-io-node:last-child').click();
  assert.match(mount.querySelector('.wl-io-detail').textContent, /token scores/);
  assert.equal(mount.querySelector('.wl-io-node:last-child').getAttribute('aria-pressed'), 'true');
  const role = mount.querySelector('.wl-role-button');
  role.click();
  assert.equal(role.getAttribute('aria-pressed'), 'true');
  assert.match(mount.querySelector('.wl-insight').textContent, /Vocabulary table/);
  role.click();
  assert.equal(role.getAttribute('aria-pressed'), 'false');
  assert.equal(mount.querySelectorAll('.wlx-waffle .c').length, 100);
  mount.querySelector('.wlx-waffle .c').click();
  assert.equal(role.getAttribute('aria-pressed'), 'true');
  mount.querySelector('.wl-training-tabs button:last-child').click();
  assert.match(mount.querySelector('.wl-training-scene').textContent, /Adjust the trainable weights/);
  mount.querySelector('#wl-training .wl-control-row button:last-child').click();
  assert.match(mount.querySelector('.wl-training-scene').textContent, /1 \/ 4/);
  const precision = mount.querySelector('[aria-label="Weight precision"]');
  const before = Number(mount.querySelector('.wl-memory-meter').getAttribute('aria-valuenow'));
  change(precision, '0.5');
  assert.equal(Number(mount.querySelector('.wl-memory-meter').getAttribute('aria-valuenow')), before / 4);
  assert.equal(mount.querySelector('.wl-memory-result').classList.contains('fits'), true);
  const ids = [...mount.querySelectorAll('[id]')].map((e) => e.id);
  assert.equal(new Set(ids).size, ids.length);
}
// Multiple towers must keep block selection scoped to the chosen stack.
for (const role of ['vision', 'audio', 'encoder']) {
  const multi = modelFixture();
  for (let i = 0; i < 2; i++) {
    multi.tensors.push(tensor(`${role}.encoder.layers.${i}.self_attn.q_proj.weight`, [64, 64]), tensor(`${role}.encoder.layers.${i}.mlp.up_proj.weight`, [256, 64]));
  }
  if (role === 'vision') multi.tensors.push(tensor('vision.embeddings.patch_embedding.weight', [64, 3, 16, 16]));
  if (role === 'audio') multi.tensors.push(tensor('audio.encoder.conv1.weight', [64, 80, 3]));
  const anatomy = render(multi);
  assert.equal(anatomy.stacks.length, 2);
  const stackPicker = mount.querySelector('[aria-label="Stack to inspect"]');
  change(stackPicker, '1');
  assert.match(mount.querySelector('.wl-block-info').textContent, new RegExp(anatomy.stacks[1].prefix.replaceAll('.', '\\.')));
  assert.equal(mount.querySelector('.pl-stack[data-k="stack:1"]').classList.contains('lit'), true);
}
// Encoder-decoder blocks include cross-attention and a third residual addition.
const crossModel = modelFixture();
for (let i = 0; i < 3; i++) crossModel.tensors.push(tensor(`model.layers.${i}.encoder_attn.q_proj.weight`, [64, 64]));
render(crossModel);
assert.match(mount.querySelector('.wl-paper-diagram').textContent, /Cross-attention/);
assert.equal(mount.querySelectorAll('.wl-paper-diagram .pd-sum').length, 3);
// Trainable and frozen counts track the selected setup and every component toggle.
const splitAnatomy = render(modelFixture());
const setup = mount.querySelector('[aria-label="Training setup"]');
const parameterCount = (name) => Number(mount.querySelector(`[data-count="${name}"]`).textContent.replaceAll(',', ''));
assert.equal(parameterCount('trainable'), splitAnatomy.total);
assert.equal(parameterCount('frozen'), 0);
change(setup, 'frozen');
assert.equal(parameterCount('trainable'), 0);
assert.equal(parameterCount('frozen'), splitAnatomy.total);
change(setup, 'full');
change(setup, 'custom');
const attentionToggle = mount.querySelector('[data-train-role="attn"]');
attentionToggle.click();
assert.equal(parameterCount('frozen'), splitAnatomy.where.attn);
assert.equal(parameterCount('trainable') + parameterCount('frozen'), parameterCount('total'));
assert.equal(mount.querySelector('.wl-trainability-parts').hidden, false);
change(setup, 'frozen');
change(setup, 'custom');
mount.querySelector('[data-train-role="head"]').click();
assert.equal(parameterCount('trainable'), splitAnatomy.where.head);
assert.equal(parameterCount('frozen'), splitAnatomy.total - splitAnatomy.where.head);
change(setup, 'full');
assert.equal(parameterCount('trainable'), splitAnatomy.total);
assert.equal(mount.querySelector('.wl-trainability-parts').hidden, true);
// The learning example must use the selected rate and reset all calculated state.
render(modelFixture());
const loss = () => Number(mount.querySelector('[data-value="loss"]').textContent);
const initialLoss = loss();
mount.querySelector('.wl-learning-lab .primary').click();
assert.ok(loss() < initialLoss);
assert.match(mount.querySelector('.wl-learning-lab').textContent, /1 weight update/);
mount.querySelector('.wl-learning-lab .wl-btn:not(.primary)').click();
assert.equal(loss(), initialLoss);
change(mount.querySelector('[aria-label="Example learning rate"]'), '0.5');
mount.querySelector('.wl-learning-lab .primary').click();
assert.equal(loss(), 0);
// Very large checkpoints must show overflow, then recover when hardware changes.
const large = modelFixture();
const largeAnatomy = window.WLA.analyze(large);
largeAnatomy.total = 30e9;
mount.replaceChildren(window.WLX.explain(large, largeAnatomy));
assert.equal(mount.querySelector('.wl-memory-meter').classList.contains('over'), true);
change(mount.querySelector('[aria-label="Available hardware"]'), '640');
assert.equal(mount.querySelector('.wl-memory-meter').classList.contains('over'), false);
// Sidecar text is rendered as text, including untrusted markup.
const recorded = modelFixture();
recorded.extras.Training = { file: 'trainer_state.json', data: { note: '<img src=x onerror=alert(1)>', learning_rate: 0.0001 } };
render(recorded);
assert.equal(mount.querySelector('.wl-training-records img'), null);
assert.match(mount.querySelector('.wl-record-json').textContent, /learning_rate/);
// Unknown architectures retain transparency, training, distribution and memory views.
render({ tensors: [tensor('projection.weight', [16, 64])], fileName: 'unknown.safetensors', fileSize: 2048, extras: {} });
for (const id of ['wl-io', 'wl-training', 'wl-parameters', 'wl-run']) assert.ok(mount.querySelector('#' + id));
console.log('EXPLORATION INTERACTION ASSERTIONS PASSED');
dom.window.close();

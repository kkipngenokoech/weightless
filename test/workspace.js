/* Run with the same temporary jsdom setup documented in test/explain.js. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<!doctype html><body><div id="app"></div></body>', {
  runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://localhost/',
});
const { window } = dom;
window.scrollTo = () => {};
window.HTMLElement.prototype.scrollIntoView = () => {};
for (const file of ['anatomy.js', 'gallery.js', 'explain.js', 'main.js']) window.eval(fs.readFileSync(path.join(__dirname, '../media', file), 'utf8'));
const document = window.document;
const tensor = (name, shape, dtype = 'BF16') => ({ name, shape, dtype, params: shape.reduce((a, b) => a * b, 1), bytes: shape.reduce((a, b) => a * b, 2) });
const model = { fileName: 'test/workspace', fileSize: 1, config: { hidden_size: 64 }, extras: { Generation: { file: 'generation_config.json', data: { max_new_tokens: 42 } } }, meta: { format: 'pt' }, tensors: [tensor('model.embed_tokens.weight', [1000, 64]), tensor('lm_head.weight', [1000, 64])] };
for (let i = 0; i < 20; i++) {
  model.tensors.push(tensor(`model.layers.${i}.self_attn.q_proj.weight`, [64, 64]), tensor(`model.layers.${i}.mlp.up_proj.weight`, [256, 64]), tensor(`model.layers.${i}.input_layernorm.weight`, [64]));
}
// Packed experts contain two logical parameters per stored byte. Mask buffers contain none.
model.tensors.push(tensor('packed.experts_blocks', [100], 'U8'), tensor('h.0.attn.bias', [1, 1, 128, 128], 'F32'));
function open() {
  window.dispatchEvent(new window.MessageEvent('message', { data: { type: 'model', ...model } }));
  document.querySelector('#wl-tab-1').click();
  return document.querySelector('.wa-workspace');
}
let workspace = open();
assert.ok(workspace, 'normal model message renders the engineering workspace');
assert.doesNotMatch(document.body.textContent, /Expert details|About this view/);
assert.equal(workspace.querySelector('.wa-tensor-panel').open, false);
assert.equal(workspace.querySelector('.wa-inspector').hidden, true);
const dataMenu = document.querySelector('.wl-model-data');
dataMenu.open = true; dataMenu.dispatchEvent(new window.Event('toggle'));
assert.match(dataMenu.textContent, /hidden_size/);
const dataFile = dataMenu.querySelector('[aria-label="Model data file"]');
dataFile.value = '1'; dataFile.dispatchEvent(new window.Event('change'));
assert.match(dataMenu.textContent, /max_new_tokens/);
assert.match(dataMenu.textContent, /42/);
dataFile.value = '2'; dataFile.dispatchEvent(new window.Event('change'));
assert.match(dataMenu.textContent, /format/);
dataMenu.querySelector('[aria-label="Close model data"]').click();
assert.equal(dataMenu.open, false);
assert.equal(workspace.querySelectorAll('.graph-details').length, 0, 'one synchronized inspector');
const expected = window.WLA.analyze(model).total;
assert.match(workspace.querySelector('.wa-inspector').textContent, new RegExp(expected.toLocaleString('en-US')));
const search = workspace.querySelector('[aria-label="Find module or tensor"]');
function find(query) { search.value = query; search.dispatchEvent(new window.Event('input')); }
find('model.layers.10.self_attn.q_proj.weight');
assert.equal(workspace.querySelectorAll('.wa-module-row').length, 1);
workspace.querySelector('.wa-module-row').click();
assert.match(workspace.querySelector('.wa-inspector').textContent, /model.layers.10.self_attn.q_proj/);
assert.doesNotMatch(workspace.querySelector('.wa-inspector').textContent, /null/);
assert.equal(workspace.querySelectorAll('.ttable tbody tr').length, 1);
assert.equal(workspace.querySelector('.wa-table-pages').hidden, true);
assert.equal(workspace.querySelector('.wa-inspector').hidden, false);
assert.match(workspace.querySelector('.ttable').textContent, /\[64, 64\]/);
assert.ok(workspace.querySelector('.gnode.selected[aria-label="model.layers.10.self_attn.q_proj"]'));
workspace.querySelector('.wa-tensor-name').click();
assert.match(workspace.querySelector('.wa-inspector').textContent, /Tensor inspector/);
assert.match(workspace.querySelector('.wa-inspector').textContent, /BF16/);
// Scope boundaries must distinguish layer 1 from layer 10 through layer 19.
find('model.layers.1');
const layerOne = [...workspace.querySelectorAll('.wa-module-row')].find((row) => row.dataset.module === 'model.layers.1');
layerOne.click();
assert.equal(workspace.querySelectorAll('.ttable tbody tr').length, 3);
assert.ok([...workspace.querySelectorAll('.wa-tensor-name')].every((button) => button.textContent.startsWith('model.layers.1.')));
const filter = workspace.querySelector('[aria-label="Filter scoped tensors"]');
filter.value = 'mlp'; filter.dispatchEvent(new window.Event('input'));
assert.equal(workspace.querySelectorAll('.ttable tbody tr').length, 1);
filter.value = ''; filter.dispatchEvent(new window.Event('input'));
// Graph drill-down and breadcrumb navigation update the same selected scope.
find('model.layers');
[...workspace.querySelectorAll('.wa-module-row')].find((row) => row.dataset.module === 'model.layers').click();
workspace.querySelector('.wa-inspector .wl-btn').click();
assert.match(workspace.querySelector('.wa-scope').textContent, /^model.layers$/);
const expand = [...workspace.querySelectorAll('.crumb-link')].find((element) => /show all 20/.test(element.textContent));
assert.ok(expand); expand.click();
assert.equal(workspace.querySelectorAll('.graph-nodes .gnode').length, 13);
assert.equal(workspace.querySelector('[aria-label="Next graph page"]').disabled, false);
workspace.querySelector('[aria-label="Next graph page"]').click();
assert.equal(workspace.querySelectorAll('.graph-nodes .gnode').length, 9);
// Paper mode stays available and clicking its stages selects actual module paths.
const view = workspace.querySelector('[aria-label="Architecture view"]');
view.value = 'paper'; view.dispatchEvent(new window.Event('change'));
assert.ok(workspace.querySelector('.wl-paper-diagram'));
workspace.querySelector('.pd-attention').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
assert.match(workspace.querySelector('.wa-scope').textContent, /^model.layers.\d+$/);
find('<script>');
assert.equal(workspace.querySelectorAll('.wa-module-row').length, 0);
assert.match(workspace.querySelector('.wa-module-tree').textContent, /No matching modules/);
// Selected scope survives reconstruction of the normal viewer.
workspace = open();
assert.match(workspace.querySelector('.wa-scope').textContent, /^model.layers.\d+$/);
// Standalone inventory uses pagination instead of silently dropping rows.
const many = { ...model, fileName: 'test/many', tensors: Array.from({ length: 4201 }, (_, i) => tensor(`weights.part_${i}.weight`, [2, 2])) };
window.dispatchEvent(new window.MessageEvent('message', { data: { type: 'model', ...many } }));
document.querySelector('#wl-tab-1')?.click();
workspace = document.querySelector('.wa-workspace');
assert.equal(workspace.querySelectorAll('.ttable tbody tr').length, 100);
assert.equal(workspace.querySelector('.wa-table-pages').hidden, false);
assert.match(workspace.querySelector('.tensors .count').textContent, /1–100 of 4,201/);
workspace.querySelector('.wa-table-pages button:last-child').click();
assert.match(workspace.querySelector('.tensors .count').textContent, /101–200 of 4,201/);
// Non-transformer checkpoints still expose the complete module/tensor inventory.
const families = [
  ['cnn', [tensor('features.0.conv.weight', [32, 3, 3, 3]), tensor('features.1.conv.weight', [64, 32, 3, 3]), tensor('classifier.weight', [10, 64])]],
  ['state-space', [tensor('backbone.embedding.weight', [1000, 64]), tensor('backbone.layers.0.mixer.A_log', [64, 16]), tensor('backbone.layers.1.mixer.A_log', [64, 16]), tensor('backbone.layers.0.mlp.up_proj.weight', [128, 64]), tensor('backbone.layers.1.mlp.up_proj.weight', [128, 64])]],
  ['adapter', [tensor('base_model.model.layers.0.self_attn.q_proj.lora_A.weight', [8, 64]), tensor('base_model.model.layers.0.self_attn.q_proj.lora_B.weight', [64, 8])]],
  ['arbitrary', [tensor('projection.weight', [3, 7]), tensor('scale', [])]],
];
for (const [family, tensors] of families) {
  window.dispatchEvent(new window.MessageEvent('message', { data: { type: 'model', ...model, fileName: 'test/' + family, tensors } }));
  document.querySelector('#wl-tab-1')?.click();
  workspace = document.querySelector('.wa-workspace');
  assert.ok(workspace, family + ' has a structural workspace');
  assert.equal(workspace.querySelectorAll('.ttable tbody tr').length, tensors.length);
  assert.equal(workspace.querySelector('[aria-label="Architecture view"] option[value="paper"]'), null, family + ' does not receive a transformer schematic');
}
console.log('ARCHITECTURE WORKSPACE ASSERTIONS PASSED');
dom.window.close();

/* Weightless web — landing UI + local-file handling. The viewer itself is the
 * extension's webview code (main.js), fed via the same MessageEvent contract. */
(function () {
  'use strict';
  const P = window.WLP;

  // The viewer (main.js) listens for the same messages the VS Code host posts.
  const post = (data) => window.dispatchEvent(new MessageEvent('message', { data }));
  function showViewer(model) {
    post(model);
    document.title = model.fileName + ' — Weightless';
  }
  function fail(e) { post({ type: 'error', message: String((e && e.message) || e) }); }

  // ---------- Hugging Face Hub ----------
  async function openHub(input) {
    let id;
    try { id = P.normalizeId(input); } catch (e) { fail(e); return; }
    post({ type: 'loading', id });
    try {
      const model = await P.loadFromHub(id);
      showViewer({ type: 'model', fileName: id, ...model });
    } catch (e) { fail(e); }
  }

  // ---------- local files (header-only via File.slice — nothing is uploaded) ----------
  const SIDECARS = [
    ['Training', ['training_args.json', 'trainer_state.json', 'training_config.json', 'args.json', 'hyperparams.json', 'hparams.json', 'train_config.json']],
    ['Generation', ['generation_config.json']],
    ['Adapter', ['adapter_config.json']],
    ['Quantization', ['quantization_config.json', 'quant_config.json']],
    ['Tokenizer', ['tokenizer_config.json']],
  ];

  async function loadSTFile(file) {
    const dv = new DataView(await file.slice(0, 8).arrayBuffer());
    const n = Number(dv.getBigUint64(0, true));
    if (!(n > 0 && n <= 512 * 1024 * 1024)) throw new Error('Invalid safetensors header — is this a .safetensors file?');
    const header = JSON.parse(new TextDecoder().decode(await file.slice(8, 8 + n).arrayBuffer()));
    return {
      tensors: P.tensorsFromHeader(header, file.name), fileSize: file.size,
      meta: header.__metadata__ || null, config: null, extras: {}, sharded: false, shardFiles: [file.name],
    };
  }

  async function loadGGUFFile(file) {
    let cap = Math.min(file.size, 4 * 1024 * 1024);
    for (;;) {
      const buf = await file.slice(0, cap).arrayBuffer();
      try {
        const { kv, tensors } = P.parseGGUF(buf, file.size);
        return {
          tensors, fileSize: file.size, meta: P.kvToMeta(kv), config: P.kvToConfig(kv),
          extras: {}, sharded: false, shardFiles: [file.name],
        };
      } catch (e) {
        if (e.code === 'EOFBUF' && cap < file.size) { cap = Math.min(file.size, cap * 4); continue; }
        throw e;
      }
    }
  }

  async function openFiles(fileList) {
    const files = [...fileList];
    const main = files.find((f) => /\.safetensors$/i.test(f.name)) || files.find((f) => /\.gguf$/i.test(f.name));
    if (!main) { fail('Drop a .safetensors or .gguf file. A config.json can come along with it.'); return; }
    post({ type: 'loading', id: main.name });
    try {
      const model = /\.gguf$/i.test(main.name) ? await loadGGUFFile(main) : await loadSTFile(main);
      // sidecar JSONs dropped in the same gesture
      const byName = {};
      files.forEach((f) => { byName[f.name.toLowerCase()] = f; });
      if (byName['config.json'] && !model.config) {
        try { model.config = JSON.parse(await byName['config.json'].text()); } catch (_) { /* ignore */ }
      }
      for (const [label, cands] of SIDECARS) {
        for (const name of cands) {
          const f = byName[name];
          if (!f) continue;
          try { model.extras[label] = { file: name, data: JSON.parse(await f.text()) }; break; }
          catch (_) { /* not JSON */ }
        }
      }
      showViewer({ type: 'model', fileName: main.name, ...model });
    } catch (e) { fail(e); }
  }

  // ---------- wiring ----------
  function pickFiles() {
    const i = document.createElement('input');
    i.type = 'file'; i.multiple = true;
    i.onchange = () => { if (i.files.length) openFiles(i.files); };
    i.click();
  }
  ['dragover', 'dragenter'].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); document.body.classList.add('dragging'); }));
  document.addEventListener('dragleave', (e) => { e.preventDefault(); document.body.classList.remove('dragging'); });
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    document.body.classList.remove('dragging');
    if (e.dataTransfer && e.dataTransfer.files.length) openFiles(e.dataTransfer.files);
  });

  const q = new URLSearchParams(location.search).get('model');
  if (q) { openHub(q); return; }
  post({ type: 'gallery' });
  const slot = document.querySelector('.wlg-slot');
  if (slot) {
    const drop = document.createElement('div');
    drop.className = 'wl-drop';
    drop.tabIndex = 0;
    drop.innerHTML = 'Have a file? Drop a <b>.safetensors</b> or <b>.gguf</b> here, or click to choose one. Add its <b>config.json</b> for more detail. Only the index is read, and nothing leaves your computer.';
    drop.addEventListener('click', pickFiles);
    drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickFiles(); } });
    const links = document.createElement('div');
    links.className = 'wl-links';
    links.innerHTML = 'Also in your editor: <a href="https://marketplace.visualstudio.com/items?itemName=kipngenokoech.weightless" target="_blank" rel="noopener">VS Code</a> · <a href="https://open-vsx.org/extension/kipngenokoech/weightless" target="_blank" rel="noopener">Open VSX</a> · <a href="https://github.com/kkipngenokoech/weightless" target="_blank" rel="noopener">GitHub</a>';
    slot.append(drop, links);
  }
})();

/* Weightless web — landing UI + local-file handling. The viewer itself is the
 * extension's webview code (main.js), fed via the same MessageEvent contract. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const P = window.WLP;

  function showViewer(model) {
    $('hero').classList.add('hidden');
    $('topbar').classList.remove('hidden');
    window.dispatchEvent(new MessageEvent('message', { data: model }));
    document.title = model.fileName + ' — Weightless';
  }
  function heroError(msg) {
    const e = $('hero-err');
    e.textContent = msg;
    e.classList.remove('hidden');
    busy(false);
  }
  function busy(b) {
    $('go').disabled = b;
    $('go').textContent = b ? 'Reading header…' : 'Inspect';
  }

  // ---------- Hugging Face Hub ----------
  async function openHub(input, updateUrl) {
    busy(true);
    $('hero-err').classList.add('hidden');
    try {
      const id = P.normalizeId(input);
      const model = await P.loadFromHub(id);
      if (updateUrl) history.replaceState(null, '', '?model=' + encodeURIComponent(id));
      showViewer({ type: 'model', fileName: id, ...model });
    } catch (e) {
      heroError(String((e && e.message) || e));
    }
    busy(false);
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
    if (!main) { heroError('Drop a .safetensors or .gguf file (a config.json can ride along).'); return; }
    busy(true);
    $('hero-err').classList.add('hidden');
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
      history.replaceState(null, '', location.pathname);
      showViewer({ type: 'model', fileName: main.name, ...model });
    } catch (e) {
      heroError(String((e && e.message) || e));
    }
    busy(false);
  }

  // ---------- wiring ----------
  $('go').addEventListener('click', () => { const v = $('hubid').value.trim(); if (v) openHub(v, true); });
  $('hubid').addEventListener('keydown', (e) => { if (e.key === 'Enter') { const v = $('hubid').value.trim(); if (v) openHub(v, true); } });
  $('again').addEventListener('click', () => { location.href = location.pathname; });

  const drop = $('drop');
  ['dragover', 'dragenter'].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  document.addEventListener('dragleave', (e) => { e.preventDefault(); drop.classList.remove('over'); });
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    if (e.dataTransfer && e.dataTransfer.files.length) openFiles(e.dataTransfer.files);
  });
  drop.addEventListener('click', () => {
    const i = document.createElement('input');
    i.type = 'file'; i.multiple = true;
    i.onchange = () => { if (i.files.length) openFiles(i.files); };
    i.click();
  });

  const q = new URLSearchParams(location.search).get('model');
  if (q) { $('hubid').value = q; openHub(q, false); }
})();

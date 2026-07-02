/* Safetensors Viewer — webview. Builds the module-hierarchy "architecture" from
 * tensor names, collapses repeated blocks, infers layer types, and renders the UI.
 * Everything here is derived from the header (names/shapes/dtypes) — see README for
 * why a true forward-pass graph needs the model code, not just the weights. */
(function () {
  'use strict';
  const app = document.getElementById('app');
  let MODEL = null;

  // ---------- formatting ----------
  const fmtNum = (n) => {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
  };
  const fmtBytes = (b) => {
    if (b >= 1024 ** 3) return (b / 1024 ** 3).toFixed(2) + ' GB';
    if (b >= 1024 ** 2) return (b / 1024 ** 2).toFixed(1) + ' MB';
    if (b >= 1024) return (b / 1024).toFixed(1) + ' KB';
    return b + ' B';
  };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ---------- module tree ----------
  function newNode(key, fullPath) {
    return { key, path: fullPath, children: new Map(), tensors: [], params: 0, bytes: 0, count: 0 };
  }
  function buildTree(tensors) {
    const root = newNode('', '');
    for (const t of tensors) {
      const segs = t.name.split('.');
      const leaf = segs.pop(); // e.g. "weight" / "bias"
      let node = root;
      root.params += t.params; root.bytes += t.bytes; root.count += 1;
      let acc = '';
      for (const s of segs) {
        acc = acc ? acc + '.' + s : s;
        if (!node.children.has(s)) node.children.set(s, newNode(s, acc));
        node = node.children.get(s);
        node.params += t.params; node.bytes += t.bytes; node.count += 1;
      }
      node.tensors.push({ leaf, shape: t.shape, dtype: t.dtype, params: t.params, bytes: t.bytes });
    }
    return root;
  }

  // ---------- layer-type inference (heuristic, from names/shapes) ----------
  function collectLeafNames(node, acc) {
    for (const t of node.tensors) acc.push(t.leaf);
    for (const c of node.children.values()) collectLeafNames(c, acc);
    return acc;
  }
  function inferType(node) {
    const childKeys = [...node.children.keys()];
    const numericChildren = childKeys.length > 0 && childKeys.every((k) => /^\d+$/.test(k));
    if (numericChildren) return { label: `ModuleList ×${childKeys.length}`, cls: 'list' };
    const names = collectLeafNames(node, [...node.children.keys()]).join(' ').toLowerCase() + ' ' + node.key.toLowerCase();
    const has = (re) => re.test(names);
    if (has(/q_proj|k_proj|v_proj|o_proj|out_proj|\battn\b|attention|attn_q|attn_k|attn_v|attn_output/)) return { label: 'Attention', cls: 'attn' };
    if (has(/gate_proj|up_proj|down_proj|\bmlp\b|fc1|fc2|feed_forward|\bffn\b|ffn_gate|ffn_up|ffn_down/)) return { label: 'MLP', cls: 'mlp' };
    if (has(/patch_embed|patch_embedding/)) return { label: 'PatchEmbed', cls: 'conv' };
    if (has(/embed|embd/)) return { label: 'Embedding', cls: 'embed' };
    if (has(/layernorm|rmsnorm|\bnorm\b|_norm\b|ln_/)) return { label: 'Norm', cls: 'norm' };
    if (has(/\bconv\b/)) return { label: 'Conv', cls: 'conv' };
    // pure linear leaf-module: only weight(+bias), 2D
    if (node.children.size === 0 && node.tensors.length && node.tensors.every((t) => t.shape.length <= 2)) {
      const w = node.tensors.find((t) => t.leaf === 'weight');
      if (w && w.shape.length === 2) return { label: `Linear ${w.shape[1]}→${w.shape[0]}`, cls: 'linear' };
    }
    return { label: 'Module', cls: 'mod' };
  }
  // guess "frozen backbone" vs "trained head" — heuristic, purely nominal
  function kindOf(topKey) {
    return /vision|language|encoder|backbone|siglip|gemma|llama|bert|clip|vit/i.test(topKey)
      ? { label: 'likely backbone', cls: 'frozen' }
      : { label: 'head / other', cls: 'trained' };
  }

  // ---------- rendering ----------
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const kid of kids) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
    return e;
  }

  function summaryBar(model) {
    const total = model.tensors.reduce((a, t) => a + t.params, 0);
    const dt = {};
    for (const t of model.tensors) dt[t.dtype] = (dt[t.dtype] || 0) + 1;
    const chips = Object.entries(dt).sort((a, b) => b[1] - a[1])
      .map(([d, c]) => h('span', { class: 'chip' }, `${d} · ${c}`));
    const cards = [
      ['Parameters', fmtNum(total)],
      ['Tensors', String(model.tensors.length)],
      ['File size', fmtBytes(model.fileSize)],
      ['Shards', model.sharded ? String(model.shardFiles.length) : '1'],
    ].map(([k, v]) => h('div', { class: 'card' }, h('div', { class: 'card-v' }, v), h('div', { class: 'card-k' }, k)));
    return h('div', { class: 'summary' },
      h('div', { class: 'title' }, model.fileName),
      h('div', { class: 'cards' }, ...cards),
      h('div', { class: 'chips' }, h('span', { class: 'chips-label' }, 'dtypes:'), ...chips));
  }

  function tabs(names, onSwitch) {
    const bar = h('div', { class: 'tabs' });
    const btns = names.map((n, i) => h('button', {
      class: 'tab' + (i === 0 ? ' active' : ''),
      onclick: () => { [...bar.children].forEach((b) => b.classList.remove('active')); btns[i].classList.add('active'); onSwitch(n); },
    }, n));
    btns.forEach((b) => bar.append(b));
    return bar;
  }

  // ---------- flow ordering + graph helpers ----------
  // Heuristic ordering of top-level components so the spine reads roughly
  // input → backbone → head. Purely nominal, from names.
  function flowScore(key) {
    const k = key.toLowerCase();
    if (/patch_embed|embed_tokens|\bembed|\btok/.test(k)) return 0;
    if (/vision|image|visual|siglip|clip|\bvit\b|img/.test(k)) return 1;
    if (/language|\btext\b|decoder|transformer|llama|gemma|qwen|mistral|\bmodel\b/.test(k)) return 2;
    if (/proj|adapter|connector|merger/.test(k)) return 3;
    if (/head|value|classifier|score|lm_head|\boutput\b/.test(k)) return 4;
    if (/norm|ln_f|final/.test(k)) return 5;
    return 2.5;
  }
  const SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs, ...kids) {
    const e = document.createElementNS(SVGNS, tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    for (const kid of kids) if (kid != null) e.append(kid);
    return e;
  }
  // Collapse an all-numeric ModuleList to one representative block + count.
  function listInfo(node) {
    const kids = [...node.children.values()];
    const numeric = kids.length > 0 && kids.every((k) => /^\d+$/.test(k.key));
    if (numeric) return { rep: kids.find((k) => k.key === '0') || kids[0], count: kids.length };
    return null;
  }
  function graphChildren(node) {
    const li = listInfo(node);
    if (li) return [{ node: li.rep, repeat: li.count }];
    return [...node.children.values()].filter((c) => c.params > 0)
      .sort((a, b) => b.params - a.params).map((c) => ({ node: c, repeat: 1 }));
  }
  function expandDepth(node, expanded, maxDepth) {
    (function walk(n, d) {
      if (d > maxDepth || n.children.size === 0) return;
      expanded.add(n.path);
      for (const ch of graphChildren(n)) walk(ch.node, d + 1);
    })(node, 0);
  }
  function expandAll(tops, expanded) { for (const t of tops) expandDepth(t, expanded, 6); }
  function findByPath(root, path) {
    if (!path) return root;
    let node = root;
    for (const seg of path.split('.')) { node = node.children.get(seg); if (!node) return null; }
    return node;
  }
  function allTensors(node, acc) {
    acc = acc || [];
    for (const t of node.tensors) acc.push({ ...t, mod: node.path });
    for (const c of node.children.values()) allTensors(c, acc);
    return acc;
  }
  // Infer input/output feature dims straight from weight shapes. Honest and exact
  // for leaf modules; for containers we surface the residual-stream width instead.
  function inferIO(node) {
    const w = node.tensors.find((t) => t.leaf === 'weight') || node.tensors.find((t) => t.shape && t.shape.length);
    if (w) {
      const s = w.shape;
      if (s.length === 2) {
        if (/embed/.test(node.path.toLowerCase())) return { in: s[0], out: s[1], kind: 'embed' }; // vocab → dim
        return { in: s[1], out: s[0], kind: 'linear' };                                           // [out,in]
      }
      if (s.length === 1) return { in: s[0], out: s[0], kind: 'norm' };
      if (s.length >= 3) return { in: s[1], out: s[0], kind: 'conv' };                             // [out,in,*k]
    }
    // container: hidden width = most common 1D (norm) dim in the subtree, else common 2D out-dim
    const freq = new Map();
    for (const t of allTensors(node)) {
      if (!t.shape) continue;
      const d = t.shape.length === 1 ? t.shape[0] : (t.shape.length === 2 ? t.shape[0] : null);
      if (d) freq.set(d, (freq.get(d) || 0) + (t.shape.length === 1 ? 2 : 1));
    }
    let best = null, bestN = 0;
    for (const [d, n] of freq) if (n > bestN) { best = d; bestN = n; }
    return best ? { width: best, kind: 'container' } : null;
  }
  function ioLabel(io) {
    if (!io) return null;
    if (io.width) return `width ${io.width}`;
    if (io.kind === 'embed') return `vocab ${io.in} → dim ${io.out}`;
    if (io.kind === 'norm') return `dim ${io.in}`;
    return `${io.in} → ${io.out}`;
  }
  // The scalar dim carried out of / into a module, for labelling flow arrows.
  function ioLabelDim(io, side) {
    if (!io) return null;
    if (io.width) return io.width;
    return side === 'in' ? io.in : io.out;
  }

  // ---------- input / output typing + role classification ----------
  // A model can have several entry branches (e.g. an image tower AND token
  // embeddings). We detect each branch's input modality and each head's output
  // type from the weights, so the overview can show them explicitly.
  function subtreeNames(node) { return allTensors(node).map((t) => (t.mod + '.' + t.leaf).toLowerCase()); }
  function inModality(node) {
    const names = subtreeNames(node);
    const k = (node.key || '').toLowerCase();
    const has = (re) => names.some((n) => re.test(n));
    if ((/vision|image|visual|siglip|clip|vit/.test(k) || has(/patch_embed|patch_embedding|patchifier|pixel/))
        && has(/patch|embed|conv|proj/)) return { kind: 'image', label: 'image / pixels' };
    const emb = allTensors(node).find((t) => t.shape.length === 2 &&
      /embed_tokens|word_embeddings|\bwte\b|tok_embeddings|token_embd|shared\.weight/.test((t.mod + '.' + t.leaf).toLowerCase()));
    if (emb) return { kind: 'tokens', label: `token ids · vocab ${emb.shape[0]}` };
    return null;
  }
  function headType(node) {
    const k = (node.key || '').toLowerCase();
    const w = allTensors(node).find((t) => t.leaf === 'weight' && t.shape.length === 2);
    if (/lm_head|logits/.test(k)) return { label: w ? `logits · ${w.shape[0]}` : 'logits' };
    if (/value/.test(k)) return { label: 'value (scalar)' };
    if (/reward/.test(k)) return { label: 'reward (scalar)' };
    if (/classifier|score|\bcls\b/.test(k)) return { label: w ? `classes · ${w.shape[0]}` : 'class logits' };
    if (/\bhead\b|proj_out/.test(k) || k === 'output') return { label: w ? `logits · ${w.shape[0]}` : 'output' };
    return null;
  }
  function roleOf(node) {
    const k = (node.key || '').toLowerCase();
    if (headType(node) && !/model|backbone|transformer|encoder|decoder|tower|language|vision/.test(k)) return 'head';
    if (/proj|adapter|connector|merger/.test(k) && !/vision|language|text/.test(k)) return 'connector';
    const m = inModality(node);
    if (m && m.kind === 'image') return 'input';
    return 'trunk';
  }

  // ---------- backbone identification ----------
  // Best-effort naming of the architecture family, from config.json
  // (model_type / architectures) and module naming. Null when unknown.
  const BACKBONE_PATTERNS = [
    [/paligemma/, 'PaliGemma'], [/llava/, 'LLaVA'], [/idefics/, 'Idefics'], [/qwen2?[_-]?vl/, 'Qwen-VL'],
    [/siglip/, 'SigLIP'], [/dinov2/, 'DINOv2'], [/\bclip\b/, 'CLIP'], [/convnext/, 'ConvNeXt'],
    [/resnet/, 'ResNet'], [/swin/, 'Swin'], [/\bvit\b|vision_transformer/, 'ViT'],
    [/gemma[_-]?3/, 'Gemma 3'], [/gemma[_-]?2/, 'Gemma 2'], [/gemma/, 'Gemma'],
    [/llama[_-]?3/, 'Llama 3'], [/llama/, 'LLaMA'],
    [/qwen[_-]?3/, 'Qwen3'], [/qwen[_-]?2/, 'Qwen2'], [/qwen/, 'Qwen'],
    [/mixtral/, 'Mixtral'], [/mistral/, 'Mistral'], [/phi[_-]?3/, 'Phi-3'], [/\bphi\b/, 'Phi'],
    [/gpt[_-]?neox/, 'GPT-NeoX'], [/gpt2/, 'GPT-2'], [/roberta/, 'RoBERTa'], [/\bbert\b/, 'BERT'],
    [/\bt5\b/, 'T5'], [/whisper/, 'Whisper'], [/wav2vec/, 'Wav2Vec2'],
  ];
  // Strict scoped lookup: only an actual matching sub-config (vision_config,
  // text_config, audio_config…) — never the whole config for a component.
  function scopedConfig(config, key) {
    if (!config) return null;
    const k = (key || '').toLowerCase();
    for (const [ck, val] of Object.entries(config)) {
      if (!val || typeof val !== 'object' || !/config$/i.test(ck)) continue;
      const kk = ck.toLowerCase();
      if ((/vision|image/.test(k) && /vision|image/.test(kk)) ||
          (/audio|speech/.test(k) && /audio|speech/.test(kk)) ||
          ((k === 'model' || /text|language|decoder/.test(k)) && /text|language/.test(kk))) return val;
    }
    return null;
  }
  // Structural fingerprints: identify the family from weight layout alone
  // (norm scheme, vocab size, characteristic widths) when names/config are mute.
  // Prefixed with ≈ because it's an inference, not a stored fact.
  function structFingerprint(node) {
    const names = subtreeNames(node).join(' ');
    const ts = allTensors(node);
    const emb = ts.find((t) => t.shape.length === 2 && /embed_tokens|word_embeddings|\bwte\b|tok_embeddings|token_embd/.test(t.mod + '.' + t.leaf));
    const vocab = emb ? emb.shape[0] : 0;
    const prePostFFN = /pre_feedforward_layernorm/.test(names) && /post_feedforward_layernorm/.test(names);
    const qkNorm = /q_norm|k_norm/.test(names);
    if (prePostFFN) {
      if (qkNorm || vocab === 262144) return '≈ Gemma 3';
      return '≈ Gemma 2';
    }
    if (vocab === 262144 || vocab === 256000) return '≈ Gemma';
    if (/vision_model|vision_tower|image_encoder/.test(node.path + ' ' + names)) {
      if (ts.some((t) => t.shape.length === 2 && t.shape[0] === 4304 && t.shape[1] === 1152)) return '≈ SigLIP So400m';
      if (/class_embedding/.test(names)) return '≈ CLIP ViT';
      if (/patch_embedding|patch_embed/.test(names)) return '≈ ViT';
    }
    if (/gate_proj/.test(names) && /input_layernorm/.test(names)) return '≈ LLaMA-style decoder';
    return null;
  }
  function detectBackbone(node, config) {
    const hay = [];
    const cfg = node.path ? scopedConfig(config, node.key) : config;
    if (cfg) {
      if (cfg.model_type) hay.push(String(cfg.model_type));
      if (Array.isArray(cfg.architectures)) hay.push(cfg.architectures.join(' '));
    }
    hay.push(node.key || '', subtreeNames(node).slice(0, 60).join(' '));
    const s = hay.join(' ').toLowerCase();
    for (const [re, label] of BACKBONE_PATTERNS) if (re.test(s)) return label;
    return structFingerprint(node);
  }
  // Exact provenance: the checkpoint this model was actually initialized from,
  // when the sidecar files recorded it. This is a stored fact, not an inference.
  function provenanceOf(node, model) {
    const out = [];
    const cfg = node.path ? scopedConfig(model.config, node.key) : model.config;
    if (cfg && cfg._name_or_path) out.push({ name: String(cfg._name_or_path), src: 'config.json · _name_or_path' });
    if (!node.path) {
      const ad = model.extras && model.extras.Adapter;
      if (ad && ad.data && ad.data.base_model_name_or_path)
        out.push({ name: String(ad.data.base_model_name_or_path), src: ad.file + ' · base_model_name_or_path' });
      const tr = model.extras && model.extras.Training;
      if (tr && tr.data) for (const k of ['model_name_or_path', 'pretrained_model_name_or_path', 'base_model', 'model_name'])
        if (tr.data[k]) { out.push({ name: String(tr.data[k]), src: tr.file + ' · ' + k }); break; }
      if (model.meta) for (const k of ['base_model', 'model_name', 'source'])
        if (model.meta[k]) { out.push({ name: String(model.meta[k]), src: 'header __metadata__ · ' + k }); break; }
    }
    const seen = new Set();
    return out.filter((p) => p.name && !seen.has(p.name) && seen.add(p.name));
  }

  // ---------- composition summary ----------
  // Counts actual parametric layers in a subtree, straight from weight shapes.
  function summarize(node) {
    const counts = { linear: 0, attn: 0, mlp: 0, norm: 0, embed: 0, conv: 0 };
    const lists = []; const dt = {};
    let params = 0, bytes = 0, tensors = 0;
    (function walk(n) {
      const kids = [...n.children.values()];
      if (kids.length && kids.every((c) => /^\d+$/.test(c.key))) lists.push({ key: n.key, size: kids.length });
      if (kids.length === 0) {
        const w = n.tensors.find((t) => t.leaf === 'weight') || n.tensors[0];
        if (w && w.shape) {
          if (w.shape.length >= 3) counts.conv++;
          else if (w.shape.length === 2) (/embed/.test(n.path.toLowerCase()) ? counts.embed++ : counts.linear++);
          else if (w.shape.length === 1) counts.norm++;
        }
      } else {
        const kk = n.key.toLowerCase();
        if (/self_attn|attention|\battn\b/.test(kk)) counts.attn++;
        else if (/mlp|feed_forward|\bffn\b|moe/.test(kk)) counts.mlp++;
      }
      for (const t of n.tensors) { params += t.params; bytes += t.bytes; tensors++; dt[t.dtype] = (dt[t.dtype] || 0) + 1; }
      for (const c of n.children.values()) walk(c);
    })(node);
    return { counts, lists, dt, params, bytes, tensors };
  }
  function pickConfig(config, scopeKey) {
    if (!config) return null;
    const k = (scopeKey || '').toLowerCase();
    for (const [key, val] of Object.entries(config)) {
      if (val && typeof val === 'object' && /config$/i.test(key)) {
        const kk = key.toLowerCase();
        if ((/vision|image/.test(k) && /vision|image/.test(kk)) ||
            (/text|language|decoder/.test(k) && /text|language/.test(kk))) return val;
      }
    }
    return config;
  }
  function configFacts(cfg) {
    if (!cfg) return [];
    const pick = (...keys) => { for (const key of keys) if (cfg[key] != null && typeof cfg[key] !== 'object') return cfg[key]; return null; };
    return [
      ['activation', pick('hidden_act', 'activation_function', 'activation', 'hidden_activation', 'act_fn')],
      ['hidden', pick('hidden_size', 'n_embd', 'd_model')],
      ['layers', pick('num_hidden_layers', 'n_layer', 'num_layers')],
      ['heads', pick('num_attention_heads', 'n_head')],
      ['kv-heads', pick('num_key_value_heads')],
      ['ffn', pick('intermediate_size', 'ffn_dim', 'n_inner')],
      ['vocab', pick('vocab_size')],
    ].filter(([, v]) => v != null);
  }
  function compositionEl(node, config, model) {
    const s = summarize(node);
    const wrap = h('div', { class: 'sumcard' });
    const chips = h('div', { class: 'sum-chips' });
    const bb = detectBackbone(node, config);
    if (bb) chips.append(h('span', { class: 'sum-chip bb', title: 'architecture family (from config.json / naming / structure)' }, h('b', null, bb)));
    if (model) for (const p of provenanceOf(node, model))
      chips.append(h('span', { class: 'sum-chip bb', title: p.src }, 'base: ', h('b', null, p.name)));
    const add = (label, n, cls) => { if (n) chips.append(h('span', { class: 'sum-chip ' + (cls || '') }, h('b', null, String(n)), ' ' + label)); };
    add('Linear', s.counts.linear, 't-linear');
    add('Attention', s.counts.attn, 't-attn');
    add('MLP', s.counts.mlp, 't-mlp');
    add('Norm', s.counts.norm, 't-norm');
    add('Embedding', s.counts.embed, 't-embed');
    add('Conv', s.counts.conv, 't-conv');
    if (!chips.children.length) chips.append(h('span', { class: 'sum-chip' }, 'no parametric layers'));
    wrap.append(chips);
    if (s.lists.length) {
      const b = h('div', { class: 'sum-sub' }, h('span', { class: 'sum-k' }, 'blocks: '));
      s.lists.sort((a, c) => c.size - a.size).slice(0, 8).forEach((l) => b.append(h('span', { class: 'sum-mini' }, `${l.key} ×${l.size}`)));
      wrap.append(b);
    }
    const cf = configFacts(pickConfig(config, node.key || (node.path || '').split('.')[0]));
    if (cf.length) {
      const c = h('div', { class: 'sum-sub' }, h('span', { class: 'sum-k' }, 'from config.json: '));
      cf.forEach(([k, v]) => c.append(h('span', { class: 'sum-mini' }, `${k} ${v}`)));
      wrap.append(c);
    } else if (!node.path) {
      wrap.append(h('div', { class: 'sum-note' }, 'Activation functions aren’t stored in weights — add a config.json next to the file to show them.'));
    }
    return wrap;
  }

  // ---------- PyTorch-style definition, reconstructed from tensor shapes ----------
  function codeFor(node) {
    const lines = []; const MAX = 400; let truncated = false;
    const push = (s) => { if (lines.length < MAX) lines.push(s); else truncated = true; };
    function leafDef(n) {
      const w = n.tensors.find((t) => t.leaf === 'weight');
      const b = n.tensors.some((t) => t.leaf === 'bias');
      const lp = n.path.toLowerCase();
      if (w) {
        if (w.shape.length === 2) return /embed/.test(lp)
          ? `Embedding(${w.shape[0]}, ${w.shape[1]})`
          : `Linear(in_features=${w.shape[1]}, out_features=${w.shape[0]}, bias=${b ? 'True' : 'False'})`;
        if (w.shape.length === 1) return /rms/.test(lp) ? `RMSNorm((${w.shape[0]},))` : `LayerNorm((${w.shape[0]},))`;
        if (w.shape.length >= 3) return `Conv${w.shape.length - 2}d(${w.shape[1]}, ${w.shape[0]}, kernel_size=(${w.shape.slice(2).join(', ')}))`;
      }
      return null;
    }
    function body(n, ind) {
      const pad = '  '.repeat(ind);
      for (const t of n.tensors) push(`${pad}${t.leaf}: Parameter([${t.shape.join(', ')}])  # ${t.dtype}`);
      for (const c of n.children.values()) emit(c, c.key, ind);
    }
    function emit(n, label, ind) {
      const pad = '  '.repeat(ind);
      const kids = [...n.children.values()];
      const numeric = kids.length > 0 && kids.every((k) => /^\d+$/.test(k.key));
      if (!kids.length) {
        const def = leafDef(n);
        if (def) push(`${pad}(${label}): ${def}`);
        else {
          push(`${pad}(${label}):`);
          for (const t of n.tensors) push(`${pad}  ${t.leaf}: Parameter([${t.shape.join(', ')}])  # ${t.dtype}`);
        }
        return;
      }
      if (numeric) {
        const sorted = kids.slice().sort((a, b) => Number(a.key) - Number(b.key));
        push(`${pad}(${label}): ModuleList(`);
        if (sorted.every((k) => k.children.size === 0)) {
          // leaf blocks: group consecutive identical defs; call out index gaps
          // (a missing index = a parameter-free module, e.g. an activation)
          let prev = -1, i = 0;
          while (i < sorted.length) {
            const a = Number(sorted[i].key);
            if (a > prev + 1) push(`${pad}  (${prev + 1}${a - 1 > prev + 1 ? '-' + (a - 1) : ''}): # parameter-free (e.g. activation — not stored in weights)`);
            const def = leafDef(sorted[i]) || 'Module(…)';
            let j = i;
            while (j + 1 < sorted.length && Number(sorted[j + 1].key) === Number(sorted[j].key) + 1 &&
                   (leafDef(sorted[j + 1]) || 'Module(…)') === def) j++;
            const b2 = Number(sorted[j].key);
            push(`${pad}  (${a}${b2 > a ? '-' + b2 : ''}): ${b2 > a ? (b2 - a + 1) + ' x ' : ''}${def}`);
            prev = b2; i = j + 1;
          }
        } else {
          const rep = sorted.find((k) => k.key === '0') || sorted[0];
          push(`${pad}  (0-${sorted.length - 1}): ${sorted.length} x Module(`);
          body(rep, ind + 2);
          push(`${pad}  )`);
        }
        push(`${pad})`);
        return;
      }
      push(`${pad}(${label}): Module(`);
      body(n, ind + 1);
      push(`${pad})`);
    }
    push('# reconstructed from tensor names/shapes — class names and forward() are not stored in weights');
    emit(node, node.key || 'model', 0);
    if (truncated) lines.push('# … truncated');
    return lines.join('\n');
  }

  // ---------- GitHub-style code block (line numbers, token colors, copy) ----------
  const CODE_TOKS = /(#.*)|\b(Linear|Embedding|ModuleList|Module|LayerNorm|RMSNorm|Parameter|Conv[123]d)\b|\b(True|False)\b|([A-Za-z_]\w*)(?==)|(\d+(?:\.\d+)?)/g;
  function hlLine(line) {
    const out = h('span', { class: 'cl-text' });
    let i = 0, m;
    CODE_TOKS.lastIndex = 0;
    while ((m = CODE_TOKS.exec(line))) {
      if (m.index > i) out.append(line.slice(i, m.index));
      const cls = m[1] ? 'tok-com' : m[2] ? 'tok-cls' : m[3] ? 'tok-kw' : m[4] ? 'tok-attr' : 'tok-num';
      out.append(h('span', { class: cls }, m[0]));
      i = m.index + m[0].length;
    }
    if (i < line.length) out.append(line.slice(i));
    return out;
  }
  // JSON tokenizer: keys, strings, numbers, booleans/null
  const JSON_TOKS = /("(?:[^"\\]|\\.)*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  function hlJsonLine(line) {
    const out = h('span', { class: 'cl-text' });
    let i = 0, m;
    JSON_TOKS.lastIndex = 0;
    while ((m = JSON_TOKS.exec(line))) {
      if (m.index > i) out.append(line.slice(i, m.index));
      if (m[1]) { out.append(h('span', { class: m[2] ? 'tok-attr' : 'tok-str' }, m[1])); if (m[2]) out.append(m[2]); }
      else out.append(h('span', { class: m[3] ? 'tok-kw' : 'tok-num' }, m[0]));
      i = m.index + m[0].length;
    }
    if (i < line.length) out.append(line.slice(i));
    return out;
  }
  function codeBlockEl(title, text, mode) {
    const lines = text.split('\n');
    const hl = mode === 'json' ? hlJsonLine : hlLine;
    const btn = h('button', { class: 'gbtn' }, 'Copy');
    btn.onclick = () => {
      try { navigator.clipboard.writeText(text); btn.textContent = 'Copied ✓'; setTimeout(() => { btn.textContent = 'Copy'; }, 1200); }
      catch (_) { btn.textContent = 'copy failed'; }
    };
    const block = h('div', { class: 'code-block' });
    for (const ln of lines) block.append(h('div', { class: 'cl-line' }, hl(ln)));
    return h('div', { class: 'code-frame' },
      h('div', { class: 'code-tools' },
        h('span', { class: 'code-title' }, title),
        h('span', { class: 'code-meta' }, `${lines.length} lines · ${mode === 'json' ? 'JSON' : 'pseudo-PyTorch'}`),
        h('span', { class: 'gspacer' }), btn),
      block);
  }

  // ---------- I/O sizes + training footprint ----------
  function ioTrainCard(model, root) {
    const ts = allTensors(root);
    const chipEl = (text, cls, title) => h('span', { class: 'sum-chip ' + (cls || ''), title: title || '' }, text);
    const chips = h('div', { class: 'sum-chips' });

    // image input resolution: patch conv (in_channels=3) + position-embedding grid
    const conv = ts.find((t) => t.shape.length === 4 && t.shape[1] === 3);
    if (conv) {
      const comp = conv.mod.split('.')[0];
      const pos = ts.find((p) => p.mod.split('.')[0] === comp && p.shape.length === 2 &&
        /position_embedding|pos_embed/.test(p.mod + '.' + p.leaf));
      let done = false;
      if (pos) for (const n of [pos.shape[0], pos.shape[0] - 1]) {
        const g = Math.sqrt(n);
        if (Number.isInteger(g)) {
          chips.append(chipEl(`image input ≈ ${g * conv.shape[2]}×${g * conv.shape[3]}`, 't-conv',
            `${g}×${g} patches of ${conv.shape[2]}px — derived from patch + position embeddings`));
          done = true; break;
        }
      }
      if (!done) chips.append(chipEl(`image patch ${conv.shape[2]}×${conv.shape[3]}px`, 't-conv', 'patch embedding kernel'));
    }
    // context length: learned position embeddings, else config
    const tpos = ts.find((t) => t.shape.length === 2 && !/vision|patch|image/.test(t.mod) &&
      /(^|\.)(wpe|embed_positions|position_embeddings?)($|\.)/.test(t.mod + '.' + t.leaf));
    const cfgCtx = model.config && (model.config.max_position_embeddings ||
      (model.config.text_config && model.config.text_config.max_position_embeddings));
    if (tpos) chips.append(chipEl(`context ${fmtNum(tpos.shape[0])}`, 't-embed', 'learned position embeddings'));
    else if (cfgCtx) chips.append(chipEl(`context ${fmtNum(cfgCtx)}`, 't-embed', 'max_position_embeddings from config.json'));
    // vocab + tied/untied output projection
    const emb = ts.filter((t) => t.shape.length === 2 &&
      /embed_tokens|word_embeddings|\bwte\b|tok_embeddings|token_embd/.test(t.mod + '.' + t.leaf))
      .sort((a, b) => b.shape[0] - a.shape[0])[0];
    if (emb) {
      chips.append(chipEl(`vocab ${fmtNum(emb.shape[0])}`, 't-embed'));
      const sep = ts.some((t) => t !== emb && t.shape.length === 2 && t.shape[0] === emb.shape[0] &&
        /head|logits|output/.test(t.mod + '.' + t.leaf));
      chips.append(chipEl(sep ? 'separate lm_head' : 'no separate lm_head (tied or headless)', 't-linear',
        sep ? 'a vocab-sized output projection is stored' : 'no vocab-sized output matrix in the file'));
    }
    // LoRA adapter detection
    const loraA = model.tensors.filter((t) => /lora_a/i.test(t.name));
    if (loraA.length) {
      const rank = Math.min(...loraA.map((t) => Math.min(...t.shape)));
      const targets = new Set(loraA.map((t) => t.name.replace(/\.lora_a.*/i, ''))).size;
      chips.append(chipEl(`LoRA · rank ${rank} · ${targets} target modules`, 'bb'));
    }

    // memory: weights at common precisions + finetune rules of thumb
    const P = root.params;
    const mem = h('div', { class: 'sum-sub' }, h('span', { class: 'sum-k' }, 'weights in memory: '));
    [['as stored', model.fileSize], ['bf16', 2 * P], ['fp32', 4 * P], ['int8', P], ['int4', Math.round(P / 2)]]
      .forEach(([k, v]) => mem.append(h('span', { class: 'sum-mini' }, `${k} ${fmtBytes(v)}`)));
    const tr = h('div', { class: 'sum-sub' }, h('span', { class: 'sum-k' }, 'finetune est.: '));
    tr.append(h('span', { class: 'sum-mini', title: 'bf16 weights + bf16 grads + fp32 master/momentum/variance ≈ 16 bytes/param' },
      `full (AdamW) ≈ ${fmtBytes(16 * P)}`));
    tr.append(h('span', { class: 'sum-mini', title: 'frozen bf16 weights; adapter params add ~1–2%' },
      `LoRA ≈ ${fmtBytes(2 * P)} + adapter`));
    const tc = model.config && (model.config.text_config || model.config);
    if (tc && tc.num_hidden_layers && tc.num_attention_heads && tc.hidden_size) {
      const kvH = tc.num_key_value_heads || tc.num_attention_heads;
      const hd = tc.head_dim || Math.round(tc.hidden_size / tc.num_attention_heads);
      tr.append(h('span', { class: 'sum-mini', title: `2 × ${tc.num_hidden_layers} layers × ${kvH} kv-heads × ${hd} head_dim × 2 bytes` },
        `KV cache ≈ ${fmtBytes(2 * tc.num_hidden_layers * kvH * hd * 2 * 1024)} / 1k tokens (bf16)`));
    }

    const wrap = h('div', { class: 'sumcard' }, chips, mem, tr,
      h('div', { class: 'sum-note' }, 'estimates are rules of thumb — activations, KV cache growth and framework overhead not included'));
    return wrap;
  }

  // Nominal ordering of a block's sub-modules so the flow reads
  // norm → attention → norm → MLP. Sequence within a block is not knowable from
  // weights alone, so this is a heuristic (see the hint under the canvas).
  function seqRank(key) {
    const k = key.toLowerCase();
    if (/patch|embed|\btok/.test(k)) return 0;
    if (/input_layernorm|ln_?1|norm_?1|pre_?norm|ln_in/.test(k)) return 1;
    if (/self_attn|\battn\b|attention|self_attention/.test(k)) return 2;
    if (/post_attention|ln_?2|norm_?2|post_?norm/.test(k)) return 3;
    if (/mlp|feed_forward|\bffn\b|moe|experts/.test(k)) return 4;
    if (/final|ln_f|last|out_?norm/.test(k)) return 8;
    if (/head|lm_head|proj_out|classifier|score|\bout\b/.test(k)) return 9;
    return 5;
  }
  function flowSequence(node) {
    return graphChildren(node)
      .sort((a, b) => (seqRank(a.node.key) - seqRank(b.node.key)) || (b.node.params - a.node.params));
  }

  // ---------- SVG flowchart view (drill-down) ----------
  // Returns { el, focus(path) }. Shows one level at a time: the focused module's
  // sub-modules as a top→bottom pipeline of boxes joined by arrows. Drill into a
  // box (double-click / ⤢) to descend; breadcrumb / Up to ascend.
  function graphView(model, root, total, onScope) {
    let focusPath = null;     // null = model root (top-level components)
    let selected = null;      // path of selected node (details panel)
    let listExpanded = null;  // container path whose ×N blocks are shown individually

    const NODE_W = 300, NODE_H = 66, VGAP = 40, PAD = 28, CX = NODE_W / 2;
    const wrap = h('div', { class: 'graph-wrap' });
    const crumb = h('div', { class: 'graph-crumb' });
    const svg = sv('svg', { class: 'graph' });
    // arrowhead marker
    const marker = sv('marker', { id: 'stv-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
    marker.append(sv('path', { class: 'garrow', d: 'M 0 0 L 10 5 L 0 10 z' }));
    svg.append(sv('defs', null, marker));
    const viewport = sv('g');
    const gEdges = sv('g'), gNodes = sv('g');
    viewport.append(gEdges, gNodes); svg.append(viewport);
    const canvas = h('div', { class: 'graph-canvas' }, svg);
    const details = h('div', { class: 'graph-details' });

    let scale = 1, ox = PAD, oy = PAD, contentW = 0, contentH = 0, moved = false;
    const applyT = () => viewport.setAttribute('transform', `translate(${ox},${oy}) scale(${scale})`);
    // size the SVG to the (scaled) content so the canvas scrolls instead of shrinking cards
    function syncSize() {
      svg.setAttribute('width', Math.max(canvas.clientWidth - 2, contentW * scale + PAD * 2));
      svg.setAttribute('height', Math.max(canvas.clientHeight - 2, contentH * scale + PAD * 2));
    }
    const zoomBy = (f) => { scale = Math.min(2.5, Math.max(0.25, scale * f)); applyT(); syncSize(); };
    const focusNode = () => findByPath(root, focusPath) || root;

    const OW = 250, HGAP = 34; // overview layout
    function edgeBetween(fx, fy, tx, ty, opts) {
      opts = opts || {};
      const my = (fy + ty) / 2;
      gEdges.append(sv('path', { class: 'gedge ' + (opts.dashed ? 'inferred' : 'flow'), 'marker-end': 'url(#stv-arrow)',
        d: `M ${fx} ${fy} C ${fx} ${my}, ${tx} ${my}, ${tx} ${ty}` }));
      if (opts.label) { const t = sv('text', { class: 'gedge-lbl', x: (fx + tx) / 2 + 6, y: my + 3 }); t.textContent = opts.label; gEdges.append(t); }
    }
    function arrow(prevY, nextY, label) { edgeBetween(CX, prevY + NODE_H, CX, nextY, { label }); }
    // pill width fits the text (estimated from length), so labels never cut off
    function pillWidth(text) { return Math.min(300, Math.max(90, Math.round(text.length * 6.4 + 28))); }
    function pill(cx, edgeY, dir, label, kind) {
      const text = (dir === 'in' ? 'input: ' : 'output: ') + label;
      const PW = pillWidth(text), PH = 26, GAP = 24;
      const py = dir === 'in' ? edgeY - GAP - PH : edgeY + GAP;
      const fo = sv('foreignObject', { x: cx - PW / 2, y: py, width: PW, height: PH });
      fo.append(h('div', { class: 'gpill ' + (dir === 'in' ? 'gpill-in k-' + kind : 'gpill-out') }, text));
      gNodes.append(fo);
      if (dir === 'in') edgeBetween(cx, py + PH, cx, edgeY, {});
      else edgeBetween(cx, edgeY, cx, py, {});
    }
    function nodeBox(item) {
      const { node, repeat } = item;
      const x = item.x || 0, y = item.y, W = item.w || NODE_W;
      const type = inferType(node);
      const pct = (node.params / total) * 100;
      const kids = node.children.size > 0;
      const io = inferIO(node);
      const bb = item.showBB ? detectBackbone(node, model.config) : null;
      const fo = sv('foreignObject', { x, y, width: W, height: NODE_H });
      const card = h('div', {
        class: 'gnode t-' + type.cls + (kids ? ' drillable' : '') + (repeat > 1 ? ' stacked' : '') + (selected === node.path ? ' selected' : ''),
      },
        h('div', { class: 'gnode-top' },
          h('span', { class: 'gnode-name' }, /^\d+$/.test(node.key) ? '[' + node.key + ']' : (node.key || 'root')),
          bb ? h('span', { class: 'gnode-bb' }, bb) : null,
          repeat > 1 ? h('span', { class: 'gnode-xn clickable', title: 'Show all ' + repeat + ' blocks',
            onclick: (e) => { e.stopPropagation(); const par = node.path.split('.').slice(0, -1).join('.'); listExpanded = par; focus(par, { keepList: true }); } }, '×' + repeat) : null,
          kids ? h('span', { class: 'gnode-open', title: 'Drill into this module' }, '⤢') : null),
        h('div', { class: 'gnode-bot' },
          h('span', { class: 'gtype t-' + type.cls }, type.label),
          io ? h('span', { class: 'gnode-io' }, ioLabel(io)) : null,
          h('span', { class: 'gnode-pct' }, `${fmtNum(node.params)} · ${pct.toFixed(1)}%`)),
        h('div', { class: 'gbar' }, h('div', { class: 'gbar-fill', style: `width:${Math.max(pct, 1.5)}%` })));
      card.addEventListener('click', (e) => { if (moved) return; e.stopPropagation(); selected = node.path; build(); showDetails(node); });
      card.addEventListener('dblclick', (e) => { e.stopPropagation(); if (kids) focus(node.path); });
      const openBtn = card.querySelector('.gnode-open');
      if (openBtn) openBtn.addEventListener('click', (e) => { if (moved) return; e.stopPropagation(); focus(node.path); });
      fo.append(card); gNodes.append(fo);
      // input/output type pills (overview only): 'in', 'out', or true for both
      if (item.pills) {
        const cx = x + W / 2;
        if (item.pills !== 'out') { const im = inModality(node); if (im) pill(cx, y, 'in', im.label, im.kind); }
        if (item.pills !== 'in') { const ht = headType(node); if (ht) pill(cx, y + NODE_H, 'out', ht.label, 'out'); }
      }
    }
    // Top-level overview: branch-and-merge. Each input branch (vision tower,
    // audio encoder, …) is its own vertical lane — with its own input pill —
    // converging into the primary trunk (which also shows ITS own input, e.g.
    // token ids), then fanning out to the heads.
    function buildOverview() {
      const comps = graphChildren(root);
      if (!comps.length) { contentW = NODE_W; contentH = NODE_H; return; }
      const groups = { input: [], connector: [], trunk: [], head: [] };
      for (const c of comps) groups[roleOf(c.node)].push(c);
      // primary trunk = largest trunk-classified comp, else largest overall
      let trunk = groups.trunk.slice().sort((a, b) => b.node.params - a.node.params)[0]
        || comps.slice().sort((a, b) => b.node.params - a.node.params)[0];
      const heads = groups.head.filter((c) => c !== trunk);
      // branch lanes: every input-bearing comp (and extra trunks) that isn't THE trunk
      const lanes = [];
      for (const c of [...groups.input, ...groups.trunk]) if (c !== trunk) lanes.push([c]);
      lanes.sort((a, b) => flowScore(a[0].node.key) - flowScore(b[0].node.key));
      // connectors chain onto the lane they bridge (name affinity, else largest lane)
      for (const c of groups.connector) {
        if (c === trunk) continue;
        const ck = c.node.key.toLowerCase();
        let lane = lanes.find((l) => {
          const lk = l[0].node.key.toLowerCase();
          return (/vision|image|visual/.test(ck) && /vision|image|visual/.test(lk)) ||
                 (/audio|speech/.test(ck) && /audio|speech/.test(lk));
        });
        if (!lane && lanes.length) lane = lanes.reduce((m, l) => (l[0].node.params > m[0].node.params ? l : m), lanes[0]);
        if (lane) lane.push(c); else lanes.push([c]);
      }
      for (const l of lanes) l.sort((a, b) => flowScore(a.node.key) - flowScore(b.node.key));

      const PILLROOM = 56, PH = 26, JGAP = 58, JR = 7;
      const trunkIn = inModality(trunk.node);
      // the trunk's own input gets its own column, so branches read side-by-side
      const tokenCol = !!(trunkIn && lanes.length);
      const nCols = lanes.length + (tokenCol ? 1 : 0);
      const colsW = nCols ? nCols * OW + (nCols - 1) * HGAP : 0;
      const headsW = heads.length ? heads.length * OW + (heads.length - 1) * HGAP : 0;
      contentW = Math.max(colsW, headsW, OW);
      const colX = (i) => (contentW - colsW) / 2 + i * (OW + HGAP);
      const laneY0 = PILLROOM;
      const maxLen = lanes.length ? Math.max(...lanes.map((l) => l.length)) : 0;
      const lanesBottom = laneY0 + (maxLen ? maxLen * (NODE_H + VGAP) - VGAP : 0);
      // branches meet at a shared junction between the columns
      const junc = nCols > 1 ? { x: contentW / 2, y: lanesBottom + JGAP } : null;
      const trunkY = junc ? junc.y + JGAP : (lanes.length ? lanesBottom + 84 : PILLROOM);
      const trunkX = (contentW - OW) / 2;

      function junction(x, y, label) {
        gNodes.append(sv('circle', { class: 'gjunc', cx: x, cy: y, r: JR }));
        const t = sv('text', { class: 'gjunc-lbl', x: x + 13, y: y + 4 }); t.textContent = label; gNodes.append(t);
      }

      // input-branch columns
      lanes.forEach((lane, i) => {
        lane.forEach((c, j) => {
          const y = laneY0 + j * (NODE_H + VGAP);
          if (j > 0) {
            const d = ioLabelDim(inferIO(lane[j - 1].node), 'out');
            edgeBetween(colX(i) + OW / 2, y - VGAP, colX(i) + OW / 2, y, { label: d ? String(d) : null });
          }
          nodeBox({ node: c.node, repeat: c.repeat, x: colX(i), y, w: OW, pills: j === 0 ? 'in' : null, showBB: true });
        });
        const exitY = laneY0 + lane.length * (NODE_H + VGAP) - VGAP;
        const d = ioLabelDim(inferIO(lane[lane.length - 1].node), 'out');
        const tgt = junc ? [junc.x, junc.y - JR] : [trunkX + OW / 2, trunkY];
        edgeBetween(colX(i) + OW / 2, exitY, tgt[0], tgt[1], { dashed: true, label: d ? String(d) : null });
      });
      // the trunk's own input as its own column, meeting the branches at the junction
      if (tokenCol) {
        const tx = colX(nCols - 1) + OW / 2;
        const txt = 'input: ' + trunkIn.label;
        const PW = pillWidth(txt);
        const fo = sv('foreignObject', { x: tx - PW / 2, y: laneY0 - 24 - PH, width: PW, height: PH });
        fo.append(h('div', { class: 'gpill gpill-in k-' + trunkIn.kind }, txt));
        gNodes.append(fo);
        edgeBetween(tx, laneY0 - 24, junc.x, junc.y - JR, { dashed: true });
      }
      if (junc) {
        junction(junc.x, junc.y, 'merge');
        const d = ioLabelDim(inferIO(trunk.node), 'in');
        edgeBetween(junc.x, junc.y + JR, trunkX + OW / 2, trunkY, { dashed: true, label: d ? String(d) : null });
      }
      nodeBox({ node: trunk.node, repeat: trunk.repeat, x: trunkX, y: trunkY, w: OW,
        pills: (trunkIn && !tokenCol) ? 'in' : null, showBB: true });
      // heads break out into their own columns from a split point
      let bottomY = trunkY;
      if (heads.length) {
        const headX = (j) => (contentW - headsW) / 2 + j * (OW + HGAP);
        if (heads.length > 1) {
          const split = { x: trunkX + OW / 2, y: trunkY + NODE_H + JGAP };
          edgeBetween(trunkX + OW / 2, trunkY + NODE_H, split.x, split.y - JR, { dashed: true });
          junction(split.x, split.y, 'split');
          const headY = split.y + JGAP;
          heads.forEach((c, j) => {
            const d = ioLabelDim(inferIO(c.node), 'in');
            edgeBetween(split.x, split.y + JR, headX(j) + OW / 2, headY, { dashed: true, label: d ? String(d) : null });
            nodeBox({ node: c.node, repeat: c.repeat, x: headX(j), y: headY, w: OW, pills: 'out', showBB: true });
          });
          bottomY = headY;
        } else {
          const headY = trunkY + NODE_H + 72;
          const d = ioLabelDim(inferIO(heads[0].node), 'in');
          edgeBetween(trunkX + OW / 2, trunkY + NODE_H, headX(0) + OW / 2, headY, { dashed: true, label: d ? String(d) : null });
          nodeBox({ node: heads[0].node, repeat: heads[0].repeat, x: headX(0), y: headY, w: OW, pills: 'out', showBB: true });
          bottomY = headY;
        }
      }
      contentH = bottomY + NODE_H + (heads.length ? PILLROOM : 24);
    }
    // Drill-down: the focused module's sub-modules as a top→bottom pipeline.
    // If the focused module is a ModuleList and the user asked to expand it,
    // show every block individually instead of the collapsed ×N representative.
    function buildSequence() {
      const fn = focusNode();
      const li = listInfo(fn);
      const seq = (li && listExpanded === focusPath)
        ? [...fn.children.values()].sort((a, b) => Number(a.key) - Number(b.key)).map((c) => ({ node: c, repeat: 1 }))
        : flowSequence(fn);
      const items = []; let cursorY = 0;
      for (const ch of seq) { items.push({ node: ch.node, repeat: ch.repeat, y: cursorY }); cursorY += NODE_H + VGAP; }
      for (let i = 0; i < items.length; i++) {
        if (i > 0) { const prevOut = ioLabelDim(inferIO(items[i - 1].node), 'out'); arrow(items[i - 1].y, items[i].y, prevOut ? String(prevOut) : null); }
        nodeBox(items[i]);
      }
      if (!items.length) {
        const fo = sv('foreignObject', { x: 0, y: 0, width: NODE_W, height: NODE_H });
        fo.append(h('div', { class: 'gnode t-mod' }, h('div', { class: 'gnode-name' }, 'Leaf module — no sub-modules'),
          h('div', { class: 'gnode-bot' }, h('span', { class: 'gnode-io' }, 'see tensors below'))));
        gNodes.append(fo); cursorY = NODE_H;
      }
      contentW = NODE_W; contentH = Math.max(cursorY - VGAP, NODE_H);
    }
    function build() {
      gEdges.textContent = ''; gNodes.textContent = '';
      if (focusPath == null) buildOverview(); else buildSequence();
    }
    function fit() {
      const r = canvas.getBoundingClientRect();
      if (!r.width || !contentW) return;
      // keep cards readable: only shrink to fit the WIDTH (never the height —
      // tall diagrams scroll down instead), and never below 0.55× or above 1:1
      scale = Math.max(0.55, Math.min(1, (r.width - PAD * 2) / contentW));
      ox = Math.max(PAD, (r.width - contentW * scale) / 2);
      oy = PAD; applyT(); syncSize();
      canvas.scrollTop = 0; canvas.scrollLeft = Math.max(0, (contentW * scale + PAD * 2 - r.width) / 2);
    }
    function renderCrumb() {
      crumb.textContent = '';
      crumb.append(h('span', { class: 'crumb-link', onclick: () => focus(null) }, 'model'));
      if (focusPath) {
        let acc = '';
        focusPath.split('.').forEach((seg, i, arr) => {
          acc = acc ? acc + '.' + seg : seg; const p = acc;
          const label = /^\d+$/.test(seg) ? '[' + seg + ']' : seg;
          crumb.append(h('span', { class: 'crumb-sep' }, ' ▸ '));
          crumb.append(i === arr.length - 1
            ? h('span', { class: 'crumb-cur' }, label)
            : h('span', { class: 'crumb-link', onclick: () => focus(p) }, label));
        });
        // expand/collapse control when focused on a ModuleList
        const fn = findByPath(root, focusPath);
        const li = fn && listInfo(fn);
        if (li) {
          crumb.append(h('span', { class: 'crumb-sep' }, '  ·  '));
          crumb.append(listExpanded === focusPath
            ? h('span', { class: 'crumb-link', onclick: () => { listExpanded = null; renderCrumb(); build(); requestAnimationFrame(fit); } }, `collapse to ×${li.count}`)
            : h('span', { class: 'crumb-link', onclick: () => { listExpanded = focusPath; renderCrumb(); build(); requestAnimationFrame(fit); } }, `show all ${li.count} blocks`));
        }
      } else {
        crumb.append(h('span', { class: 'crumb-note' }, ' — overview · input/output types read from weights · dashed = inferred wiring'));
      }
    }
    function showDetails(node) {
      if (onScope) onScope(node || root);
      details.textContent = '';
      if (!node) { details.append(h('div', { class: 'gd-empty' }, 'Click a node to see its tensors, shapes and dtypes. Double-click (or ⤢) to drill in.')); return; }
      const io = inferIO(node);
      const im = inModality(node), ht = headType(node);
      const bb = detectBackbone(node, model.config);
      details.append(h('div', { class: 'gd-head' },
        h('span', { class: 'gd-name' }, node.path || node.key),
        bb ? h('span', { class: 'gd-badge bb' }, bb) : null,
        im ? h('span', { class: 'gd-badge in' }, 'input: ' + im.label) : null,
        ht ? h('span', { class: 'gd-badge out' }, 'output: ' + ht.label) : null,
        h('span', { class: 'gd-io' }, ioLabel(io) || ''),
        node.children.size ? h('button', { class: 'gbtn gd-focus', onclick: () => focus(node.path) }, 'Drill in ⤢') : null));
      details.append(h('details', { class: 'gd-code' },
        h('summary', null, 'module definition (reconstructed from weights)'),
        codeBlockEl(node.path || node.key, codeFor(node))));
      const ts = allTensors(node).sort((a, b) => b.params - a.params);
      // what kind of layer this weight belongs to, read from its shape
      const layerOf = (t) => {
        if (t.leaf !== 'weight') return '';
        const lp = (t.mod + '.' + t.leaf).toLowerCase();
        if (t.shape.length === 2) return /embed/.test(lp) ? 'Embedding' : `Linear ${t.shape[1]}→${t.shape[0]}`;
        if (t.shape.length === 1) return 'Norm';
        if (t.shape.length >= 3) return `Conv (${t.shape.slice(2).join('×')})`;
        return '';
      };
      const table = h('table', { class: 'gd-table' },
        h('tr', null, h('th', null, 'tensor'), h('th', null, 'layer'), h('th', null, 'shape'), h('th', null, 'dtype'), h('th', null, 'params')));
      for (const t of ts.slice(0, 200)) {
        table.append(h('tr', null,
          h('td', { class: 'gd-tname' }, t.mod === node.path ? t.leaf : t.mod.slice(node.path.length + 1) + '.' + t.leaf),
          h('td', { class: 'gd-ltype mono' }, layerOf(t)),
          h('td', { class: 'mono' }, '[' + t.shape.join(', ') + ']'),
          h('td', null, t.dtype),
          h('td', { class: 'mono' }, fmtNum(t.params))));
      }
      details.append(table);
      if (ts.length > 200) details.append(h('div', { class: 'gd-empty' }, `… ${ts.length - 200} more tensors`));
    }
    function focus(path, opts) {
      focusPath = path; selected = null;
      if (!opts || !opts.keepList) listExpanded = null;
      renderCrumb(); build(); showDetails(path ? findByPath(root, path) : null); requestAnimationFrame(fit);
    }

    // interaction: the canvas scrolls natively (trackpad/scrollbar); dragging
    // anywhere also scrolls it. A small movement threshold distinguishes a
    // drag from a click, so dragging over a node moves the view, not selects.
    let down = false, sx, sy, sox, soy;
    svg.addEventListener('mousedown', (e) => {
      down = true; moved = false; sx = e.clientX; sy = e.clientY;
      sox = canvas.scrollLeft; soy = canvas.scrollTop;
    });
    window.addEventListener('mousemove', (e) => {
      if (!down) return;
      if (!moved && Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) > 4) { moved = true; svg.classList.add('panning'); }
      if (moved) { canvas.scrollLeft = sox - (e.clientX - sx); canvas.scrollTop = soy - (e.clientY - sy); }
    });
    window.addEventListener('mouseup', () => { down = false; svg.classList.remove('panning'); });
    // keep the flow fitted when the panel is resized
    if (window.ResizeObserver) new ResizeObserver(() => fit()).observe(canvas);

    const up = () => { if (focusPath == null) return; const segs = focusPath.split('.'); segs.pop(); focus(segs.length ? segs.join('.') : null); };
    const toolbar = h('div', { class: 'graph-toolbar' },
      h('button', { class: 'gbtn', onclick: up }, '⬆ Up'),
      h('span', { class: 'gspacer' }),
      h('span', { class: 'graph-hint' }, 'double-click a box to drill in · drag or scroll to move'),
      h('button', { class: 'gbtn', onclick: () => zoomBy(1 / 1.15) }, '–'),
      h('button', { class: 'gbtn', onclick: () => zoomBy(1.15) }, '+'),
      h('button', { class: 'gbtn', onclick: fit }, 'Fit'));

    renderCrumb(); showDetails(null); build(); applyT();
    wrap.append(toolbar, crumb, canvas, details);
    requestAnimationFrame(fit);
    return { el: wrap, focus };
  }

  // ---------- indented tree view (kept as a toggle) ----------
  function treeView(model, root, total) {
    const tops = [...root.children.values()].sort((a, b) => b.params - a.params);
    const tree = h('div', { class: 'tree' });
    function renderNode(node, depth, container) {
      const kids = [...node.children.values()].sort((a, b) => b.params - a.params);
      const numeric = kids.length > 0 && kids.every((k) => /^\d+$/.test(k.key));
      const type = inferType(node);
      const pct = (node.params / total) * 100;
      const hasKids = kids.length > 0 || node.tensors.length > 0;
      const row = h('div', { class: 'node', style: `padding-left:${depth * 16}px` });
      const caret = h('span', { class: 'caret' + (hasKids ? '' : ' leaf') }, hasKids ? '▸' : '·');
      const body = h('div', { class: 'node-children' });
      body.style.display = 'none';
      let open = false;
      row.append(caret,
        h('span', { class: 'node-name' }, node.key || 'root'),
        h('span', { class: 'type ' + type.cls }, type.label),
        h('span', { class: 'node-params' }, `${fmtNum(node.params)} · ${pct.toFixed(1)}%`));
      if (hasKids) row.addEventListener('click', () => {
        open = !open; caret.textContent = open ? '▾' : '▸'; body.style.display = open ? 'block' : 'none';
      });
      container.append(row, body);
      for (const t of node.tensors) {
        body.append(h('div', { class: 'tensor-leaf', style: `padding-left:${(depth + 1) * 16 + 14}px` },
          h('span', { class: 'tl-name' }, t.leaf),
          h('span', { class: 'tl-shape' }, '[' + t.shape.join(', ') + ']'),
          h('span', { class: 'tl-dtype' }, t.dtype)));
      }
      if (numeric) {
        body.append(h('div', { class: 'repeat-note', style: `padding-left:${(depth + 1) * 16}px` },
          `${kids.length}× repeated block — showing block [0]`));
        renderNode(kids.find((k) => k.key === '0') || kids[0], depth + 1, body);
      } else {
        for (const c of kids) renderNode(c, depth + 1, body);
      }
    }
    for (const c of tops) renderNode(c, 0, tree);
    return tree;
  }

  function archView(model) {
    const root = buildTree(model.tensors);
    const total = root.params || 1;
    const wrap = h('div', { class: 'arch' });

    // Composition card + Code view — scope follows the current graph selection/focus.
    const compScope = h('div', { class: 'comp-scope' });
    const codeView = h('div', { class: 'code-view' });
    let scopeNode = root;
    function buildCodeView() {
      codeView.textContent = '';
      codeView.append(codeBlockEl(scopeNode.path || 'whole model', codeFor(scopeNode)));
    }
    function setComp(node) {
      scopeNode = node || root;
      compScope.textContent = '';
      compScope.append(
        h('div', { class: 'comp-scope-label' }, node && node.path ? node.path : 'whole model'),
        compositionEl(scopeNode, model.config, model));
      if (codeView.style.display !== 'none') buildCodeView();
    }

    // Build the graph once; the Components strip drives it via graph.focus().
    // onScope fires whenever a node/component is selected or focused.
    const graph = graphView(model, root, total, setComp);
    const tree = treeView(model, root, total);

    // top-level component strip (clickable → focus that path in the graph)
    const tops = [...root.children.values()].sort((a, b) => b.params - a.params);
    const strip = h('div', { class: 'components' });
    for (const c of tops) {
      const pct = (c.params / total) * 100;
      const kind = kindOf(c.key);
      const io = ioLabel(inferIO(c));
      const bb = detectBackbone(c, model.config);
      strip.append(h('div', { class: 'component clickable', title: 'Click to zoom into this path',
        onclick: () => { showGraph(); graph.focus(c.key); } },
        h('div', { class: 'comp-head' }, h('span', { class: 'comp-name' }, c.key),
          h('span', { class: 'badge ' + kind.cls, title: kind.label }, bb || kind.label)),
        h('div', { class: 'bar' }, h('div', { class: 'bar-fill', style: `width:${pct.toFixed(1)}%` })),
        h('div', { class: 'comp-meta' },
          `${fmtNum(c.params)} params · ${pct.toFixed(1)}%`, io ? h('span', { class: 'comp-io' }, io) : null)));
    }
    wrap.append(h('div', { class: 'section-label' }, 'Components'), strip);

    // composition (layer counts, blocks, config hyper-params) — updates on selection
    wrap.append(h('div', { class: 'section-label' }, 'Composition'), compScope);

    // trainer-facing facts: I/O sizes, vocab, memory & finetune footprint
    wrap.append(h('div', { class: 'section-label' }, 'I/O & training footprint'), ioTrainCard(model, root));

    // Graph / Tree / Code toggle (built once, shown/hidden)
    const viewBox = h('div', { class: 'arch-view' }, graph.el, tree, codeView);
    tree.style.display = 'none'; codeView.style.display = 'none';
    const bG = h('button', { class: 'seg-btn active' }, 'Graph');
    const bT = h('button', { class: 'seg-btn' }, 'Tree');
    const bC = h('button', { class: 'seg-btn' }, 'Code');
    function activate(btn, el) {
      for (const b of [bG, bT, bC]) b.classList.toggle('active', b === btn);
      for (const e of [graph.el, tree, codeView]) e.style.display = e === el ? '' : 'none';
    }
    function showGraph() { activate(bG, graph.el); }
    bG.onclick = showGraph;
    bT.onclick = () => activate(bT, tree);
    bC.onclick = () => { buildCodeView(); activate(bC, codeView); };
    wrap.append(h('div', { class: 'arch-head' },
      h('div', { class: 'section-label', style: 'margin:0' }, 'Architecture'),
      h('div', { class: 'seg' }, bG, bT, bC)));
    wrap.append(viewBox);
    return wrap;
  }

  function tensorsView(model) {
    const wrap = h('div', { class: 'tensors' });
    const search = h('input', { class: 'search', type: 'text', placeholder: 'Filter tensors by name…' });
    const table = h('table', { class: 'ttable' });
    const rows = model.tensors.slice().sort((a, b) => b.params - a.params);
    const layerOf = (t) => {
      if (!/\.weight$/.test(t.name)) return '';
      if (t.shape.length === 2) return /embed/.test(t.name.toLowerCase()) ? 'Embedding' : `Linear ${t.shape[1]}→${t.shape[0]}`;
      if (t.shape.length === 1) return 'Norm';
      if (t.shape.length >= 3) return `Conv (${t.shape.slice(2).join('×')})`;
      return '';
    };
    function draw(filter) {
      table.innerHTML = '';
      table.append(h('tr', null,
        h('th', null, 'Name'), h('th', null, 'Layer'), h('th', null, 'Shape'), h('th', null, 'dtype'),
        h('th', null, 'Params'), h('th', null, 'Size'), model.sharded ? h('th', null, 'Shard') : null));
      let shown = 0;
      for (const t of rows) {
        if (filter && !t.name.toLowerCase().includes(filter)) continue;
        shown++;
        if (shown > 4000) break; // guard huge tables
        table.append(h('tr', null,
          h('td', { class: 'tname' }, t.name),
          h('td', { class: 'gd-ltype mono' }, layerOf(t)),
          h('td', { class: 'mono' }, '[' + t.shape.join(', ') + ']'),
          h('td', null, t.dtype),
          h('td', { class: 'mono' }, fmtNum(t.params)),
          h('td', { class: 'mono' }, fmtBytes(t.bytes)),
          model.sharded ? h('td', { class: 'shard' }, t.shard) : null));
      }
      count.textContent = `${shown} of ${rows.length} tensors`;
    }
    const count = h('div', { class: 'count' });
    search.addEventListener('input', () => draw(search.value.trim().toLowerCase()));
    draw('');
    wrap.append(h('div', { class: 'search-row' }, search, count), table);
    return wrap;
  }

  function jsonView(obj, empty, title) {
    if (!obj) return h('div', { class: 'empty' }, empty);
    return codeBlockEl(title || 'json', JSON.stringify(obj, null, 2), 'json');
  }

  function render(model) {
    app.innerHTML = '';
    app.append(summaryBar(model));
    const body = h('div', { class: 'body' });
    const views = {
      Architecture: () => archView(model),
      Tensors: () => tensorsView(model),
      Config: () => jsonView(model.config, 'No config.json next to this file.', 'config.json'),
    };
    // Auto-discovered sidecar configs (training args, generation, adapter, etc.)
    for (const [label, entry] of Object.entries(model.extras || {})) {
      views[label] = () => jsonView(entry.data, '', entry.file);
    }
    views.Metadata = () => jsonView(model.meta, 'No __metadata__ in the header.', '__metadata__');
    const bar = tabs(Object.keys(views), (name) => { body.innerHTML = ''; body.append(views[name]()); });
    app.append(bar, body);
    body.append(views.Architecture());
  }

  window.addEventListener('message', (e) => {
    const msg = e.data;
    if (msg.type === 'model') { MODEL = msg; render(msg); }
    else if (msg.type === 'error') {
      app.innerHTML = '';
      app.append(h('div', { class: 'error' }, h('div', { class: 'error-t' }, 'Could not read this file'), h('div', null, msg.message)));
    }
  });
})();

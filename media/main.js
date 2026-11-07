/* Safetensors Viewer — webview. Builds the module-hierarchy "architecture" from
 * tensor names, collapses repeated blocks, infers layer types, and renders the UI.
 * Everything here is derived from the header (names/shapes/dtypes) — see README for
 * why a true forward-pass graph needs the model code, not just the weights. */
(function () {
  'use strict';
  const app = document.getElementById('app');
  let MODEL = null;
  let graphId = 0;
  const viewDisposers = [];

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
    for (const original of tensors) {
      const t = { ...original, params: window.WLA ? window.WLA.realParams(original) : original.params };
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
  const TYPE_GLYPH = { attn: '◎', mlp: '▤', embed: '⊞', norm: '≡', conv: '⌗', linear: '⊗', list: '≣', mod: '▣' };
  // ---------- rendering ----------
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
    return e;
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
  function graphView(model, root, total, onScope, options = {}) {
    let focusPath = null;     // null = model root (top-level components)
    let selected = null;      // path of selected node (details panel)
    let hierarchyPage = 0;
    let listExpanded = null;  // container path whose ×N blocks are shown individually

    const NODE_W = 300, NODE_H = 70, VGAP = 60, PAD = 28, CX = NODE_W / 2;
    const wrap = h('div', { class: 'graph-wrap' });
    const crumb = h('div', { class: 'graph-crumb' });
    // HTML cards live in an absolutely-positioned layer OVER the SVG, which only
    // draws edges. foreignObject is avoided entirely — engines mis-paint its
    // children under ancestor transforms (text escaping the cards).
    const svg = sv('svg', { class: 'graph-edges' });
    const arrowId = 'stv-arrow-' + (++graphId);
    const marker = sv('marker', { id: arrowId, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
    marker.append(sv('path', { class: 'garrow', d: 'M 0 0 L 10 5 L 0 10 z' }));
    svg.append(sv('defs', null, marker));
    const gEdges = sv('g');
    svg.append(gEdges);
    const nodesLayer = h('div', { class: 'graph-nodes' });
    const scene = h('div', { class: 'graph-scene' }, svg, nodesLayer);
    const sizer = h('div', { class: 'graph-sizer' }, scene);
    const canvas = h('div', { class: 'graph-canvas' }, sizer);
    const details = h('div', { class: 'graph-details' });

    let scale = 1, ox = PAD, oy = PAD, contentW = 0, contentH = 0, moved = false;
    const applyT = () => { scene.style.transform = `translate(${ox}px,${oy}px) scale(${scale})`; };
    // size the scroll area to the (scaled) content so the canvas scrolls
    function syncSize() {
      scene.style.width = contentW + 'px';
      scene.style.height = contentH + 'px';
      svg.setAttribute('width', contentW || 1);
      svg.setAttribute('height', contentH || 1);
      sizer.style.width = Math.max(canvas.clientWidth - 2, contentW * scale + PAD * 2) + 'px';
      sizer.style.height = Math.max(canvas.clientHeight - 2, contentH * scale + PAD * 2) + 'px';
    }
    const zoomBy = (f) => { scale = Math.min(2.5, Math.max(0.25, scale * f)); applyT(); syncSize(); };
    const focusNode = () => findByPath(root, focusPath) || root;

    const OW = 250, HGAP = 40; // overview layout (nodeSep)
    // "smoothstep" connectors: orthogonal with rounded elbows
    function edgeBetween(fx, fy, tx, ty, opts) {
      opts = opts || {};
      const my = (fy + ty) / 2;
      const dx = tx - fx;
      let d;
      if (Math.abs(dx) < 1) {
        d = `M ${fx} ${fy} L ${tx} ${ty}`;
      } else {
        const r = Math.min(8, Math.abs(dx) / 2, Math.abs(ty - fy) / 2);
        const s = dx > 0 ? 1 : -1;
        d = `M ${fx} ${fy} L ${fx} ${my - r} Q ${fx} ${my} ${fx + s * r} ${my}`
          + ` L ${tx - s * r} ${my} Q ${tx} ${my} ${tx} ${my + r} L ${tx} ${ty}`;
      }
      gEdges.append(sv('path', { class: 'gedge ' + (opts.dashed ? 'inferred' : 'flow'), 'marker-end': `url(#${arrowId})`, d }));
      if (opts.label) { const t = sv('text', { class: 'gedge-lbl', x: (fx + tx) / 2 + 6, y: my + 3 }); t.textContent = opts.label; gEdges.append(t); }
    }
    function arrow(prevY, nextY, label) { edgeBetween(CX, prevY + NODE_H, CX, nextY, { label }); }
    // pill width fits the text (estimated from length), so labels never cut off
    function pillWidth(text) { return Math.min(300, Math.max(90, Math.round(text.length * 6.4 + 28))); }
    function pill(cx, edgeY, dir, label, kind) {
      const text = (dir === 'in' ? 'input: ' : 'output: ') + label;
      const PW = pillWidth(text), PH = 26, GAP = 24;
      const py = dir === 'in' ? edgeY - GAP - PH : edgeY + GAP;
      nodesLayer.append(h('div', {
        class: 'gpill ' + (dir === 'in' ? 'gpill-in k-' + kind : 'gpill-out'),
        style: `left:${cx - PW / 2}px;top:${py}px;width:${PW}px;height:${PH}px`,
      }, text));
      if (dir === 'in') edgeBetween(cx, py + PH, cx, edgeY, {});
      else edgeBetween(cx, edgeY, cx, py, {});
    }
    function nodeBox(item) {
      const { node, repeat } = item;
      const x = item.x || 0, y = item.y, W = item.w || NODE_W;
      const type = options.hierarchy ? { cls: node.children.size ? 'mod' : 'linear', label: node.children.size ? `${node.children.size} modules` : `${node.tensors.length} tensors` } : inferType(node);
      const pct = (node.params / total) * 100;
      const kids = node.children.size > 0;
      const io = options.hierarchy ? null : inferIO(node);
      const bb = item.showBB ? detectBackbone(node, model.config) : null;
      const card = h('div', {
        role: 'button', tabindex: '0', 'aria-label': node.path || 'Model', 'aria-pressed': String(selected === node.path),
        class: 'gnode t-' + type.cls + (kids ? ' drillable' : '') + (repeat > 1 ? ' stacked' : '') + (selected === node.path ? ' selected' : ''),
        style: `left:${x}px;top:${y}px;width:${W}px;height:${NODE_H}px`,
      },
        h('div', { class: 'gnode-top' },
          h('span', { class: 'gnode-ic' }, TYPE_GLYPH[type.cls] || '▣'),
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
      card.addEventListener('keydown', (e) => {
        if (e.target !== card) return;
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selected = node.path; build(); showDetails(node); }
        if (e.key === 'ArrowRight' && kids) { e.preventDefault(); focus(node.path); }
      });
      card.addEventListener('dblclick', (e) => { e.stopPropagation(); if (kids) focus(node.path); });
      const openBtn = card.querySelector('.gnode-open');
      if (openBtn) openBtn.addEventListener('click', (e) => { if (moved) return; e.stopPropagation(); focus(node.path); });
      nodesLayer.append(card);
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
        gEdges.append(sv('circle', { class: 'gjunc', cx: x, cy: y, r: JR }));
        const t = sv('text', { class: 'gjunc-lbl', x: x + 13, y: y + 4 }); t.textContent = label; gEdges.append(t);
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
        nodesLayer.append(h('div', { class: 'gpill gpill-in k-' + trunkIn.kind,
          style: `left:${tx - PW / 2}px;top:${laneY0 - 24 - PH}px;width:${PW}px;height:${PH}px` }, txt));
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
        nodesLayer.append(h('div', { class: 'gnode t-mod', style: `left:0;top:0;width:${NODE_W}px;height:${NODE_H}px` },
          h('div', { class: 'gnode-top' },
            h('span', { class: 'gnode-ic' }, TYPE_GLYPH.mod),
            h('span', { class: 'gnode-name' }, 'Leaf module — no sub-modules')),
          h('div', { class: 'gnode-bot' }, h('span', { class: 'gnode-io' }, 'see tensors below'))));
        cursorY = NODE_H;
      }
      contentW = NODE_W; contentH = Math.max(cursorY - VGAP, NODE_H);
    }
    function buildHierarchy() {
      const parent = focusNode();
      const expandedList = listExpanded === focusPath && listInfo(parent);
      const children = expandedList
        ? [...parent.children.values()].sort((a, b) => Number(a.key) - Number(b.key)).map((node) => ({ node, repeat: 1 }))
        : graphChildren(parent);
      const pageSize = 12;
      const pageCount = Math.max(1, Math.ceil(children.length / pageSize));
      hierarchyPage = Math.min(hierarchyPage, pageCount - 1);
      const visible = children.slice(hierarchyPage * pageSize, (hierarchyPage + 1) * pageSize);
      contentW = visible.length ? NODE_W * 2 + 100 : NODE_W;
      contentH = Math.max(NODE_H, visible.length * 100 - 30);
      nodeBox({ node: parent, repeat: 1, x: 0, y: 0 });
      visible.forEach((item, index) => {
        const y = index * 100;
        gEdges.append(sv('path', { class: 'gedge flow', 'marker-end': `url(#${arrowId})`,
          d: `M ${NODE_W} ${NODE_H / 2} H ${NODE_W + 50} V ${y + NODE_H / 2} H ${NODE_W + 100}` }));
        nodeBox({ ...item, x: NODE_W + 100, y });
      });
      pageLabel.textContent = `${hierarchyPage + 1} / ${pageCount}`;
      previousPage.disabled = hierarchyPage === 0;
      nextPage.disabled = hierarchyPage === pageCount - 1;
      pageControls.hidden = pageCount === 1;
    }
    function build() {
      gEdges.textContent = ''; nodesLayer.textContent = '';
      if (options.hierarchy) buildHierarchy();
      else if (focusPath == null) buildOverview(); else buildSequence();
      syncSize();
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
        crumb.append(h('span', { class: 'crumb-note' }, ' — overview'));
      }
    }
    function showDetails(node) {
      if (onScope) onScope(node || root);
      if (options.externalDetails) return;
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
      focusPath = path; selected = path; hierarchyPage = 0;
      if (!opts || !opts.keepList) listExpanded = null;
      renderCrumb(); build(); showDetails(path ? findByPath(root, path) : null); requestAnimationFrame(fit);
    }

    // interaction: the canvas scrolls natively (trackpad/scrollbar); dragging
    // anywhere also scrolls it. A small movement threshold distinguishes a
    // drag from a click, so dragging over a node moves the view, not selects.
    let down = false, sx, sy, sox, soy;
    canvas.addEventListener('mousedown', (e) => {
      down = true; moved = false; sx = e.clientX; sy = e.clientY;
      sox = canvas.scrollLeft; soy = canvas.scrollTop;
    });
    const moveCanvas = (e) => {
      if (!down) return;
      if (!moved && Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) > 4) { moved = true; canvas.classList.add('panning'); }
      if (moved) { canvas.scrollLeft = sox - (e.clientX - sx); canvas.scrollTop = soy - (e.clientY - sy); }
    };
    const releaseCanvas = () => { down = false; canvas.classList.remove('panning'); };
    window.addEventListener('mousemove', moveCanvas);
    window.addEventListener('mouseup', releaseCanvas);
    const resizeObserver = window.ResizeObserver ? new ResizeObserver(() => fit()) : null;
    if (resizeObserver) resizeObserver.observe(canvas);
    viewDisposers.push(() => {
      window.removeEventListener('mousemove', moveCanvas);
      window.removeEventListener('mouseup', releaseCanvas);
      if (resizeObserver) resizeObserver.disconnect();
    });

    const up = () => { if (focusPath == null) return; const segs = focusPath.split('.'); segs.pop(); focus(segs.length ? segs.join('.') : null); };
    const toolbar = h('div', { class: 'graph-toolbar' },
      h('button', { class: 'gbtn', 'aria-label': 'Parent module', onclick: up }, '↑ Up'),
      h('span', { class: 'gspacer' }),
      h('span', { class: 'graph-hint' }, 'double-click a box to drill in · drag or scroll to move'),
      h('button', { class: 'gbtn', 'aria-label': 'Zoom out', onclick: () => zoomBy(1 / 1.15) }, '–'),
      h('button', { class: 'gbtn', 'aria-label': 'Zoom in', onclick: () => zoomBy(1.15) }, '+'),
      h('button', { class: 'gbtn', onclick: fit }, 'Fit'));

    const pageLabel = h('span', { class: 'wa-count' });
    const previousPage = h('button', { class: 'gbtn', 'aria-label': 'Previous graph page', onclick: () => { hierarchyPage--; build(); fit(); } }, 'Previous');
    const nextPage = h('button', { class: 'gbtn', 'aria-label': 'Next graph page', onclick: () => { hierarchyPage++; build(); fit(); } }, 'Next');
    const pageControls = h('div', { class: 'wa-graph-pages', hidden: true }, previousPage, pageLabel, nextPage);
    renderCrumb(); showDetails(null); build(); applyT();
    wrap.append(toolbar, crumb, canvas, pageControls);
    if (!options.externalDetails) wrap.append(details);
    requestAnimationFrame(fit);
    const select = (path) => {
      const node = findByPath(root, path);
      if (!node) return;
      focusPath = path ? path.split('.').slice(0, -1).join('.') || null : null;
      selected = node.path;
      hierarchyPage = 0; listExpanded = focusPath;
      // Reveal the selected child on its graph page, including numeric blocks.
      const parent = focusNode();
      const children = listInfo(parent) ? [...parent.children.values()].sort((a, b) => Number(a.key) - Number(b.key)) : graphChildren(parent).map((item) => item.node);
      hierarchyPage = Math.max(0, Math.floor(children.findIndex((child) => child.path === path) / 12));
      renderCrumb(); build(); showDetails(node); requestAnimationFrame(fit);
    };
    return { el: wrap, focus, select };

  }

  function tensorsView(model, options = {}) {
    const wrap = h('div', { class: 'tensors' });
    const search = h('input', { class: 'search', type: 'search', 'aria-label': 'Filter scoped tensors', placeholder: 'Filter by name, shape, or dtype' });
    const table = h('table', { class: 'ttable' });
    const count = h('span', { class: 'count', role: 'status' });
    const pageLabel = h('span', { class: 'wa-count' });
    let scope = '', page = 0, sortKey = 'params', descending = true, selectedTensor = null;
    const pageSize = 100;
    const collator = new Intl.Collator('en', { numeric: true });
    const rows = model.tensors.map((tensor) => ({ ...tensor, params: window.WLA ? window.WLA.realParams(tensor) : tensor.params,
      searchText: `${tensor.name} ${tensor.shape.join(' ')} ${tensor.dtype} ${tensor.shard || ''}`.toLowerCase() }));
    const columns = [['name', 'Tensor'], ['shape', 'Stored shape'], ['dtype', 'Dtype'], ['params', 'Parameters'], ['bytes', 'Storage']];
    if (model.sharded) columns.push(['shard', 'Shard']);
    const sortRows = () => rows.sort((a, b) => {
      const left = Array.isArray(a[sortKey]) ? a[sortKey].join(' × ') : a[sortKey];
      const right = Array.isArray(b[sortKey]) ? b[sortKey].join(' × ') : b[sortKey];
      const order = typeof left === 'number' ? left - right : collator.compare(String(left || ''), String(right || ''));
      return descending ? -order : order;
    });
    function draw() {
      const query = search.value.trim().toLowerCase();
      const filtered = rows.filter((tensor) => (!scope || tensor.name.startsWith(scope + '.')) &&
        (!query || tensor.searchText.includes(query)));
      const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
      page = Math.min(page, pages - 1);
      const header = h('tr', null, columns.map(([key, label]) => h('th', {
        scope: 'col', 'aria-sort': key === sortKey ? (descending ? 'descending' : 'ascending') : 'none',
      }, h('button', { class: 'wa-sort', onclick: () => {
        descending = sortKey === key ? !descending : key === 'params' || key === 'bytes';
        sortKey = key; page = 0; sortRows(); draw();
      } }, label))));
      table.replaceChildren(h('thead', null, header));
      const body = h('tbody');
      for (const tensor of filtered.slice(page * pageSize, (page + 1) * pageSize)) {
        const row = h('tr', { class: selectedTensor === tensor.name ? 'wa-tensor-selected' : '' });
        row.append(h('td', { class: 'tname' }, h('button', { class: 'wa-tensor-name', title: tensor.name, onclick: () => {
          selectedTensor = tensor.name;
          if (options.onSelect) options.onSelect(tensor);
          draw();
        } }, tensor.name)), h('td', { class: 'mono' }, '[' + tensor.shape.join(', ') + ']'),
        h('td', null, tensor.dtype), h('td', { class: 'mono' }, tensor.params.toLocaleString('en-US')),
        h('td', { class: 'mono' }, fmtBytes(tensor.bytes)));
        if (model.sharded) row.append(h('td', { class: 'shard' }, tensor.shard || '—'));
        body.append(row);
      }
      table.append(body);
      count.textContent = filtered.length ? `${page * pageSize + 1}–${Math.min((page + 1) * pageSize, filtered.length)} of ${filtered.length.toLocaleString('en-US')} tensors` : '0 tensors';
      pageLabel.textContent = `${page + 1} / ${pages}`;
      previous.disabled = page === 0; next.disabled = page === pages - 1;
      pagination.hidden = pages === 1;
    }
    const previous = h('button', { class: 'gbtn', onclick: () => { page--; draw(); } }, 'Previous');
    const next = h('button', { class: 'gbtn', onclick: () => { page++; draw(); } }, 'Next');
    const pagination = h('div', { class: 'wa-table-pages' }, previous, pageLabel, next);
    search.addEventListener('input', () => { page = 0; draw(); });
    wrap.setScope = (path) => { scope = path || ''; page = 0; draw(); };
    sortRows(); draw();
    wrap.append(h('div', { class: 'search-row' }, search, count), h('div', { class: 'wa-table-scroll' }, table),
      pagination);
    return wrap;
  }

  function architectureWorkspace(model) {
    const root = buildTree(model.tensors);
    const anatomy = window.WLA.analyze(model);
    const workspace = h('div', { class: 'wa-workspace' });
    const moduleTree = h('div', { class: 'wa-module-tree', role: 'tree', 'aria-label': 'Model modules' });
    const moduleCount = h('span', { class: 'wa-count', role: 'status' });
    const moduleSearch = h('input', { class: 'search', type: 'search', placeholder: 'Find a module or tensor…', 'aria-label': 'Find module or tensor' });
    const inspector = h('aside', { class: 'wa-inspector', 'aria-label': 'Selection inspector' });
    const scopeLabel = h('span', { class: 'wa-scope' });
    const expanded = new Set(['']);
    const records = [];
    const recordByPath = new Map();
    const renderedRows = new Map();
    let selected = root, graph, visibleRecords = [], treeLimit = 200;
    const scopeKey = 'wl.architecture.scope.' + model.fileName;
    const savedPath = store.get(scopeKey);
    const collator = new Intl.Collator('en', { numeric: true });
    function indexNode(node, depth) {
      const children = [...node.children.values()].sort((a, b) => collator.compare(a.key, b.key));
      const record = { node, depth, children,
        search: [node.path, ...node.tensors.map((tensor) => node.path ? node.path + '.' + tensor.leaf : tensor.leaf)].join(' ').toLowerCase() };
      records.push(record); recordByPath.set(node.path, record);
      children.forEach((child) => indexNode(child, depth + 1));
    }
    indexNode(root, 1);
    function reveal(path) {
      const parts = path.split('.');
      while (parts.length) { parts.pop(); expanded.add(parts.join('.')); }
    }
    function selectTree(record, focus) {
      graph.select(record.node.path);
      if (focus) renderedRows.get(record.node.path)?.focus({ preventScroll: true });
    }
    function drawTree(focusPath) {
      const query = moduleSearch.value.trim().toLowerCase();
      const matches = [];
      if (query) {
        for (const record of records) if (record.search.includes(query)) matches.push(record);
      } else {
        const visit = (record) => {
          matches.push(record);
          if (expanded.has(record.node.path)) record.children.forEach((child) => visit(recordByPath.get(child.path)));
        };
        visit(records[0]);
      }
      visibleRecords = matches.slice(0, treeLimit);
      moduleCount.textContent = query ? `${matches.length.toLocaleString('en-US')} matches` : `${records.length.toLocaleString('en-US')} modules`;
      moduleTree.replaceChildren(); renderedRows.clear();
      const focusable = visibleRecords.some((record) => record.node.path === selected.path) ? selected.path : visibleRecords[0]?.node.path;
      for (const record of visibleRecords) {
        const { node, children } = record;
        const open = expanded.has(node.path);
        const row = h('div', { class: 'wa-module-row' + (selected.path === node.path ? ' selected' : ''),
          role: 'treeitem', 'aria-level': query ? 1 : record.depth, 'aria-selected': String(selected.path === node.path),
          'aria-expanded': children.length ? String(open) : null, tabindex: focusable === node.path ? '0' : '-1',
          title: node.path || 'Model', 'data-module': node.path, style: `--depth:${query ? 0 : record.depth - 1}`,
          onclick: () => selectTree(record, true),
          onkeydown: (event) => {
            if (event.target !== row) return;
            const index = visibleRecords.indexOf(record);
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
              event.preventDefault();
              const next = visibleRecords[index + (event.key === 'ArrowDown' ? 1 : -1)];
              if (next) selectTree(next, true);
            } else if (event.key === 'ArrowRight' && children.length) {
              event.preventDefault();
              if (!open) { expanded.add(node.path); drawTree(node.path); }
              else selectTree(recordByPath.get(children[0].path), true);
            } else if (event.key === 'ArrowLeft') {
              event.preventDefault();
              if (open && children.length) { expanded.delete(node.path); drawTree(node.path); }
              else if (node.path) selectTree(recordByPath.get(node.path.split('.').slice(0, -1).join('.')), true);
            } else if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectTree(record, true); }
            else if (event.key === 'Home' || event.key === 'End') {
              event.preventDefault(); selectTree(visibleRecords[event.key === 'Home' ? 0 : visibleRecords.length - 1], true);
            }
          } },
          children.length ? h('button', { class: 'wa-tree-caret', tabindex: '-1', 'aria-label': (open ? 'Collapse ' : 'Expand ') + (node.path || 'Model'), onclick: (event) => {
            event.stopPropagation();
            if (open) expanded.delete(node.path); else expanded.add(node.path);
            drawTree(node.path);
          } }, open ? '▾' : '▸') : h('span', { class: 'wa-tree-leaf', 'aria-hidden': 'true' }, '·'),
          h('span', { class: 'wa-module-name' }, node.key || 'Model'), h('span', { class: 'wa-module-size' }, fmtNum(node.params)));
        renderedRows.set(node.path, row); moduleTree.append(row);
      }
      if (!matches.length) moduleTree.append(h('div', { class: 'wa-empty' }, 'No matching modules'));
      more.hidden = matches.length <= treeLimit;
      if (focusPath !== undefined) renderedRows.get(focusPath)?.focus({ preventScroll: true });
    }
    const more = h('button', { class: 'gbtn wa-more', onclick: () => { treeLimit += 200; drawTree(); } }, 'Show more modules');
    const paperStackIndexes = anatomy.stacks.map((_, index) => index).filter((index) =>
      anatomy.stacks[index].blocks.every((block) => block.attn > 0 && block.mlp > 0));
    const paper = h('div', { class: 'wa-paper', hidden: true });
    const stackPicker = h('select', { class: 'wl-select', 'aria-label': 'Diagram stack' }, paperStackIndexes.map((index) => h('option', { value: index }, `${anatomy.stacks[index].role} · ${anatomy.stacks[index].n} blocks`)));
    stackPicker.hidden = true;
    function drawPaper() {
      if (paper.hidden || !anatomy.stacks.length) return;
      const stackIndex = Number(stackPicker.value), stack = anatomy.stacks[stackIndex];
      const blockId = Number(selected.path.slice(stack.prefix.length + 1).split('.')[0]);
      const blockIndex = Math.max(0, stack.blocks.findIndex((block) => block.i === blockId));
      paper.replaceChildren(window.WLX.paperDiagram(anatomy, stackIndex, blockIndex, (key) => {
        let path = stack.prefix + '.' + stack.blocks[blockIndex].i;
        if (key === 'embed' || key === 'head') {
          const pattern = key === 'head' ? /lm_head|(^|\.)output\.weight/ : /embed_tokens|\.wte\.|token_embd|word_embeddings|token_embedding|(^|\.)shared\.weight/;
          const tensor = model.tensors.find((tensor) => pattern.test(tensor.name));
          if (tensor) path = tensor.name.split('.').slice(0, -1).join('.');
        }
        graph.select(path);
      }));
    }
    function showModule(node) {
      const dtypes = new Map();
      for (const tensor of allTensors(node)) dtypes.set(tensor.dtype, (dtypes.get(tensor.dtype) || 0) + tensor.bytes);
      const definition = h('details', { class: 'wl-details wa-definition' }, h('summary', null, 'Shape-derived definition'));
      definition.addEventListener('toggle', () => {
        if (definition.open && definition.children.length === 1) definition.append(codeBlockEl(node.path || 'Model', codeFor(node)));
      });
      inspector.replaceChildren(h('h3', null, 'Selection'), h('div', { class: 'wa-selected-path' }, node.path || 'Model'),
        h('dl', { class: 'wa-facts' },
          h('div', null, h('dt', null, 'Parameters'), h('dd', null, node.params.toLocaleString('en-US'))),
          h('div', null, h('dt', null, 'Storage'), h('dd', null, fmtBytes(node.bytes))),
          h('div', null, h('dt', null, 'Tensors'), h('dd', null, node.count.toLocaleString('en-US'))),
          h('div', null, h('dt', null, 'Child modules'), h('dd', null, String(node.children.size)))),
        h('details', { class: 'wl-details' }, h('summary', null, 'Storage by dtype'), h('div', { class: 'wa-dtypes' }, [...dtypes].sort((a, b) => b[1] - a[1]).map(([dtype, bytes]) => h('div', null, h('span', { class: 'mono' }, dtype), h('span', null, fmtBytes(bytes)))))),
        ...(node.children.size ? [h('button', { class: 'wl-btn', onclick: () => graph.focus(node.path || null) }, 'Open module')] : []),
        definition);
    }
    function showTensor(tensor) {
      const modulePath = tensor.name.split('.').slice(0, -1).join('.');
      graph.select(modulePath);
      inspector.hidden = false; workspace.classList.add('has-selection');
      const fields = [['Stored shape', '[' + tensor.shape.join(', ') + ']'], ['Dtype', tensor.dtype],
        ['Parameters', tensor.params.toLocaleString('en-US')], ['Storage', fmtBytes(tensor.bytes)]];
      if (tensor.shard) fields.push(['File', tensor.shard]);
      inspector.replaceChildren(h('h3', null, 'Tensor inspector'), h('div', { class: 'wa-selected-path' }, tensor.name),
        h('dl', { class: 'wa-facts' }, fields.map(([label, value]) => h('div', null, h('dt', null, label), h('dd', null, value)))),
        h('button', { class: 'wl-btn', onclick: () => showModule(selected) }, 'Module summary'));
    }
    const tensors = tensorsView(model, { onSelect: showTensor });
    const tensorSummary = h('summary', { class: 'wa-panel-title' }, 'Tensors');
    const applyScope = (node) => {
      selected = node; reveal(node.path);
      inspector.hidden = !node.path; workspace.classList.toggle('has-selection', !!node.path);
      tensorSummary.textContent = 'Tensors · ' + node.count.toLocaleString('en-US');
      store.set(scopeKey, node.path);
      scopeLabel.textContent = node.path || 'Model';
      tensors.setScope(node.path);
      showModule(node); drawTree();
      const stackIndex = anatomy.stacks.findIndex((stack) => node.path === stack.prefix || node.path.startsWith(stack.prefix + '.'));
      if (paperStackIndexes.includes(stackIndex)) stackPicker.value = String(stackIndex);
      drawPaper();
    };
    graph = graphView(model, root, root.params || 1, applyScope, { hierarchy: true, externalDetails: true });
    graph.el.classList.add('wa-module-graph');
    const diagramMode = h('select', { class: 'wl-select', 'aria-label': 'Architecture view', onchange: () => {
      const isPaper = diagramMode.value === 'paper';
      graph.el.hidden = isPaper; paper.hidden = !isPaper; stackPicker.hidden = !isPaper;
      drawPaper();
    } }, h('option', { value: 'modules' }, 'Module graph'), paperStackIndexes.length ? h('option', { value: 'paper' }, 'Paper diagram') : null);
    stackPicker.addEventListener('change', () => graph.select(anatomy.stacks[Number(stackPicker.value)].prefix));
    moduleSearch.addEventListener('input', () => { treeLimit = 200; drawTree(); workspace.classList.add('show-modules'); });
    moduleSearch.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && visibleRecords.length) selectTree(visibleRecords[0], true);
      if (event.key === 'Escape') { moduleSearch.value = ''; treeLimit = 200; drawTree(); }
    });
    const treeToggle = h('button', { class: 'wl-btn wa-tree-toggle', 'aria-expanded': 'false', onclick: () => {
      const open = workspace.classList.toggle('show-modules'); treeToggle.setAttribute('aria-expanded', String(open));
    } }, 'Modules');
    workspace.addEventListener('keydown', (event) => {
      if (event.key === '/' && !/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) {
        event.preventDefault(); workspace.classList.add('show-modules'); treeToggle.setAttribute('aria-expanded', 'true'); moduleSearch.focus();
      }
    });
    workspace.append(h('div', { class: 'wa-toolbar' }, treeToggle, moduleSearch, diagramMode, stackPicker),
      h('div', { class: 'wa-layout' },
        h('aside', { class: 'wa-tree-panel', 'aria-label': 'Module browser' }, h('div', { class: 'wa-panel-title' }, 'Modules', moduleCount), moduleTree, more),
        h('div', { class: 'wa-center' }, h('div', { class: 'wa-panel-title' }, scopeLabel), graph.el, paper),
        inspector,
        h('details', { class: 'wa-tensor-panel', 'aria-label': 'Scoped tensor inventory' }, tensorSummary, tensors)));
    if (savedPath && recordByPath.has(savedPath)) graph.select(savedPath);
    return workspace;
  }

  function jsonView(obj, empty, title) {
    if (!obj) return h('div', { class: 'empty' }, empty);
    return codeBlockEl(title || 'json', JSON.stringify(obj, null, 2), 'json');
  }

  // VS Code webview ↔ host channel; null on the web, where navigation does the job
  const vsc = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : null;
  let galleryShown = false;
  const store = {
    get: (k) => { try { return localStorage.getItem(k); } catch (_) { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch (_) { /* private mode */ } },
  };
  function openModel(id) {
    if (vsc) vsc.postMessage({ type: 'open', id });
    else location.search = '?model=' + encodeURIComponent(id);
  }
  function showGallery() {
    viewDisposers.splice(0).forEach((dispose) => dispose());
    galleryShown = true;
    window.scrollTo(0, 0);
    window.WLX.gallery(app, { onOpen: openModel });
  }
  function goBack() { if (vsc) showGallery(); else location.href = location.pathname; }
  function backButton() {
    if (!galleryShown && vsc) return null;
    return h('button', { class: 'back', onclick: goBack }, '← All models');
  }

  function modelDataMenu(model) {
    const entries = [];
    if (model.config) entries.push({ label: 'Config', file: 'config.json', data: model.config });
    for (const [label, entry] of Object.entries(model.extras || {})) entries.push({ label, file: entry.file || label, data: entry.data });
    if (model.meta) entries.push({ label: 'Metadata', file: '__metadata__', data: model.meta });
    if (!entries.length) return null;
    const menu = h('details', { class: 'wl-model-data' });
    const panel = h('div', { class: 'wl-model-data-panel' });
    menu.append(h('summary', null, 'Model data'), panel);
    menu.addEventListener('keydown', (event) => { if (event.key === 'Escape') { menu.open = false; menu.querySelector('summary').focus(); } });
    menu.addEventListener('toggle', () => {
      if (!menu.open || panel.children.length) return;
      const content = h('div', { class: 'wl-model-data-content' });
      const selector = h('select', { class: 'wl-select', 'aria-label': 'Model data file' }, entries.map((entry, index) => h('option', { value: index }, entry.label)));
      const draw = () => {
        const entry = entries[Number(selector.value)];
        content.replaceChildren(jsonView(entry.data, '', entry.file));
      };
      selector.addEventListener('change', draw);
      panel.append(h('div', { class: 'wl-model-data-toolbar' }, selector,
        h('button', { class: 'gbtn', 'aria-label': 'Close model data', onclick: () => { menu.open = false; menu.querySelector('summary').focus(); } }, '×')), content);
      draw();
    });
    return menu;
  }

  function render(model) {
    viewDisposers.splice(0).forEach((dispose) => dispose());
    app.replaceChildren();
    window.scrollTo(0, 0);
    let view;
    try { view = window.WLX.explain(model, window.WLA.analyze(model)); }
    catch (error) { console.error(error); view = h('div', { class: 'wl wlx' }, tensorsView(model)); }
    app.append(h('div', { class: 'wl-top' }, backButton(),
      h('span', { class: 'brand' }, 'Weightless'), h('span', { class: 'crumb' }, model.fileName),
      h('span', { class: 'spacer' }), modelDataMenu(model)), view);
  }

  window.WLArchitecture = architectureWorkspace;
  document.body.classList.add('wl-body');
  window.addEventListener('message', (e) => {
    const msg = e.data;
    if (!msg || !msg.type) return;
    if (msg.type === 'model') { MODEL = msg; render(msg); }
    else if (msg.type === 'gallery') showGallery();
    else if (msg.type === 'loading') {
      viewDisposers.splice(0).forEach((dispose) => dispose());
      window.WLX.loading(app, msg.id);
    }
    else if (msg.type === 'error') {
      app.innerHTML = '';
      const back = backButton();
      app.append(h('div', { class: 'wl wl-loading' },
        h('div', { class: 'wl-loading-t' }, 'Could not open this model'),
        h('div', { class: 'wl-loading-s' }, msg.message),
        back ? h('button', { class: 'wl-btn', onclick: goBack }, 'Back to all models') : null));
    }
  });
})();

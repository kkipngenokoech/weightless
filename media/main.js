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
    if (has(/q_proj|k_proj|v_proj|o_proj|out_proj|\battn\b|attention/)) return { label: 'Attention', cls: 'attn' };
    if (has(/gate_proj|up_proj|down_proj|\bmlp\b|fc1|fc2|feed_forward|\bffn\b/)) return { label: 'MLP', cls: 'mlp' };
    if (has(/patch_embed|patch_embedding/)) return { label: 'PatchEmbed', cls: 'conv' };
    if (has(/embed/)) return { label: 'Embedding', cls: 'embed' };
    if (has(/layernorm|rmsnorm|\bnorm\b|ln_/)) return { label: 'Norm', cls: 'norm' };
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

  function archView(model) {
    const root = buildTree(model.tensors);
    const total = root.params || 1;
    const wrap = h('div', { class: 'arch' });

    // top-level component strip
    const tops = [...root.children.values()].sort((a, b) => b.params - a.params);
    const strip = h('div', { class: 'components' });
    for (const c of tops) {
      const pct = (c.params / total) * 100;
      const kind = kindOf(c.key);
      strip.append(h('div', { class: 'component' },
        h('div', { class: 'comp-head' }, h('span', { class: 'comp-name' }, c.key),
          h('span', { class: 'badge ' + kind.cls }, kind.label)),
        h('div', { class: 'bar' }, h('div', { class: 'bar-fill', style: `width:${pct.toFixed(1)}%` })),
        h('div', { class: 'comp-meta' }, `${fmtNum(c.params)} params · ${pct.toFixed(1)}%`)));
    }
    wrap.append(h('div', { class: 'section-label' }, 'Components'), strip);

    // collapsible module tree
    wrap.append(h('div', { class: 'section-label' }, 'Module tree'));
    const tree = h('div', { class: 'tree' });
    function renderNode(node, depth, container) {
      const kids = [...node.children.values()].sort((a, b) => b.params - a.params);
      // collapse ModuleList: children all-numeric -> show ×N + one representative
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
      // tensors directly on this module
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
    wrap.append(tree);
    return wrap;
  }

  function tensorsView(model) {
    const wrap = h('div', { class: 'tensors' });
    const search = h('input', { class: 'search', type: 'text', placeholder: 'Filter tensors by name…' });
    const table = h('table', { class: 'ttable' });
    const rows = model.tensors.slice().sort((a, b) => b.params - a.params);
    function draw(filter) {
      table.innerHTML = '';
      table.append(h('tr', null,
        h('th', null, 'Name'), h('th', null, 'Shape'), h('th', null, 'dtype'),
        h('th', null, 'Params'), h('th', null, 'Size'), model.sharded ? h('th', null, 'Shard') : null));
      let shown = 0;
      for (const t of rows) {
        if (filter && !t.name.toLowerCase().includes(filter)) continue;
        shown++;
        if (shown > 4000) break; // guard huge tables
        table.append(h('tr', null,
          h('td', { class: 'tname' }, t.name),
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

  function jsonView(obj, empty) {
    if (!obj) return h('div', { class: 'empty' }, empty);
    return h('pre', { class: 'json' }, JSON.stringify(obj, null, 2));
  }

  function render(model) {
    app.innerHTML = '';
    app.append(summaryBar(model));
    const body = h('div', { class: 'body' });
    const views = {
      Architecture: () => archView(model),
      Tensors: () => tensorsView(model),
      Config: () => jsonView(model.config, 'No config.json next to this file.'),
      Metadata: () => jsonView(model.meta, 'No __metadata__ in the header.'),
    };
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

/* Weightless Explain — the beginner-facing views: the open-model gallery and the
 * "Explain" page for one model (silhouette plate, the journey of a token, where
 * the parameters sit, what it takes to run). Pure DOM, fed by WLA.analyze().
 * Exposed as WLX. Every number on these pages comes from the header or config. */
(function () {
  'use strict';

  // ---------- tiny DOM helpers ----------
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(kid));
    return e;
  }
  const SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs, ...kids) {
    const e = document.createElementNS(SVGNS, tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    for (const kid of kids) if (kid != null) e.append(kid);
    return e;
  }

  // ---------- numbers people can read ----------
  const sig = (x) => String(parseFloat(x.toPrecision(3)));
  function words(n) {
    if (n >= 1e12) return sig(n / 1e12) + ' trillion';
    if (n >= 1e9) return sig(n / 1e9) + ' billion';
    if (n >= 1e6) return sig(n / 1e6) + ' million';
    if (n >= 1e3) return sig(n / 1e3) + ' thousand';
    return String(n);
  }
  function short(n) {
    if (n >= 1e12) return sig(n / 1e12) + 'T';
    if (n >= 1e9) return sig(n / 1e9) + 'B';
    if (n >= 1e6) return sig(n / 1e6) + 'M';
    if (n >= 1e3) return sig(n / 1e3) + 'K';
    return String(n);
  }
  const int = (n) => Number(n).toLocaleString('en-US');
  function gb(b) {
    const g = b / 1e9;
    if (g >= 100) return Math.round(g) + ' GB';
    if (g >= 1) return sig(g) + ' GB';
    return Math.max(1, Math.round(b / 1e6)) + ' MB';
  }

  // ---------- glossary: dotted terms with a plain definition on hover/focus ----------
  const GLOSSARY = {
    parameter: 'One learned number. Training adjusts every parameter a tiny bit until the model’s outputs come out right.',
    token: 'A chunk of text the model treats as one unit: a word, part of a word, or a single character.',
    vocabulary: 'Every token the model knows. Each one has an ID number and its own list of numbers.',
    embedding: 'The list of numbers that stands for a token. Tokens used in similar ways end up with similar lists.',
    block: 'One repeating unit of the model. Every block has the same design but its own learned numbers.',
    attention: 'The step where each token looks at the other tokens and takes in what is relevant to it.',
    'feed-forward': 'The step where each token is processed on its own through a wider layer. Much of what a model “knows” lives here.',
    expert: 'One of many small feed-forward layers in a block. Each token is sent to only a few of them.',
    router: 'A small layer that decides which experts each token goes to.',
    encoder: 'The part that reads an input (text, an image, audio) and turns it into numbers describing it.',
    decoder: 'The part that writes output one token at a time, looking only at what came before.',
    patch: 'A small square of an image. Vision models read pictures patch by patch, like words in a sentence.',
    quantization: 'Storing each number with fewer bits (say 4 instead of 16) to save memory, at a small cost in quality.',
    context: 'How many tokens the model can take in at once: its working memory.',
  };
  const used = new Set();
  function T(word, shown) {
    const key = word.toLowerCase();
    used.add(key);
    return h('span', { class: 'wl-term', tabindex: '0', 'data-def': GLOSSARY[key] || '' }, shown || word);
  }

  // ---------- the plate: a model's silhouette, drawn to scale from its weights ----------
  // Each row is one block; its width is √(parameters) so a trillion-parameter model
  // and a tiny one both stay legible. Attention = cyan, feed-forward = paper,
  // hatched feed-forward = experts. The sun band is the vocabulary table.
  let PID = 0;
  function plate(sil, opts) {
    opts = opts || {};
    const mini = !!opts.mini, a = opts.anatomy || null;
    const W = mini ? 100 : 320, pad = mini ? 4 : 12;
    const stacks = sil.stacks;
    const parallel = sil.kind === 'match' && stacks.length === 2;
    const maxP = Math.max(sil.embed || 0, sil.head || 0, ...stacks.flatMap((s) => s.blocks.map((b) => b[0] + b[1])), 1);
    const colW = parallel ? (W - pad * 3) / 2 : W - pad * 2;
    const wOf = (p) => Math.max(mini ? 2 : 4, Math.sqrt(p / maxP) * colW);
    const nRows = parallel ? Math.max(...stacks.map((s) => s.blocks.length)) : stacks.reduce((n, s) => n + s.blocks.length, 0);
    const rowH = Math.max(mini ? 1.2 : 3, Math.min(mini ? 7 : 12, (mini ? 80 : 380) / Math.max(nRows, 1)));
    const gap = mini ? (rowH >= 3 ? 0.8 : 0.4) : rowH >= 5 ? 2 : rowH >= 3 ? 1.2 : 0.6;
    const LBL = mini ? 4 : 26;
    const id = 'wlp' + (++PID);

    const svg = sv('svg', { class: 'wl-plate' + (mini ? ' mini' : ''), role: mini || !a ? 'img' : 'group' });
    svg.append(sv('defs', null,
      sv('pattern', { id: id + 'h', width: mini ? 2.5 : 5, height: 6, patternUnits: 'userSpaceOnUse' },
        sv('rect', { width: 5, height: 6, class: 'hatch-bg' }),
        sv('rect', { width: mini ? 0.8 : 1.6, height: 6, class: 'hatch-ln' }))));
    const label = (text, x, y, anchor) => { if (!mini) { const t = sv('text', { class: 'pl-lbl', x, y, 'text-anchor': anchor || 'start' }); t.textContent = text; svg.append(t); } };
    let y = pad, row = 0;
    const delay = () => Math.round((row++ / Math.max(nRows, 1)) * 900) + 'ms';

    // the vocabulary table sits where text enters: above the first text stack
    const textIdx = stacks.findIndex((s) => s.role === 'decoder' || s.role === 'encoder');
    const embedBand = (x0, cw, yy) => {
      if (!sil.embed) return yy;
      label('vocabulary table', x0, yy + 11);
      yy += LBL;
      const w = wOf(sil.embed), eh = mini ? 6 : 16;
      svg.append(sv('rect', { class: 'pl-embed pr', 'data-k': 'embed', x: x0 + (cw - w) / 2, y: yy, width: w, height: eh, rx: mini ? 1 : 2, style: '--d:0ms' }));
      return yy + eh + (mini ? 4 : 10);
    };
    const drawStack = (s, si, x0, cw, y0) => {
      let yy = si === textIdx || (textIdx < 0 && si === 0) ? embedBand(x0, cw, y0) : y0;
      if (!mini) {
        const st = a && a.stacks[si];
        const what = { vision: 'vision blocks', audio: 'audio blocks', encoder: 'reading blocks', decoder: 'blocks' }[s.role];
        label(`${s.blocks.length} ${what}${st && st.experts ? ` · ${st.experts} experts each` : ''}`, x0, yy + 11);
      }
      yy += LBL;
      const g = sv('g', { class: 'pl-stack role-' + s.role, 'data-k': 'stack:' + si });
      for (const [bi, b] of s.blocks.entries()) {
        const w = wOf(b[0] + b[1]);
        const wa = w * (b[0] / Math.max(b[0] + b[1], 1));
        const x = x0 + (cw - w) / 2;
        const r = sv('g', { class: 'pr', 'data-block': si + ':' + bi, style: '--d:' + delay() });
        r.append(sv('rect', { class: 'seg-attn', x, y: yy, width: Math.max(wa, 0.5), height: rowH }));
        r.append(sv('rect', { class: 'seg-mlp', x: x + wa, y: yy, width: Math.max(w - wa, 0.5), height: rowH, style: b[2] ? `fill:url(#${id}h)` : null }));
        g.append(r);
        yy += rowH + gap;
      }
      svg.append(g);
      return yy;
    };
    if (parallel) {
      const ys = stacks.map((s, i) => drawStack(s, i, pad + i * (colW + pad), colW, y));
      y = Math.max(...ys);
    } else {
      stacks.forEach((s, i) => { y = drawStack(s, i, pad, colW, y) + (mini ? 3 : 6); });
    }
    if (sil.head) {
      y += mini ? 1 : 4;
      label('output table', pad, y + 11);
      y += LBL;
      const w = wOf(sil.head), eh = mini ? 6 : 16;
      svg.append(sv('rect', { class: 'pl-head pr', 'data-k': 'head', x: (W - w) / 2, y, width: w, height: eh, rx: mini ? 1 : 2, style: '--d:900ms' }));
      y += eh;
    }
    const H = Math.ceil(y + pad);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('aria-label', `Silhouette: ${stacks.map((s) => s.blocks.length + ' ' + s.role + ' blocks').join(', ')}`);
    return svg;
  }

  // ---------- gallery ----------
  function gallery(mount, opts) {
    const groups = window.WL_GALLERY || [];
    const input = h('input', { class: 'wlg-input', type: 'text', spellcheck: 'false', autocomplete: 'off',
      'aria-label': 'Hugging Face model id or URL', placeholder: 'Or paste any Hugging Face model, e.g. Qwen/Qwen3-8B' });
    const go = () => { const v = input.value.trim(); if (v) opts.onOpen(v); };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });

    const head = h('header', { class: 'wlg-head' },
      h('div', { class: 'wl-eyebrow' }, 'Weightless · Model explorer'),
      h('h1', { class: 'wlg-title' }, 'Explore open models'),
      h('p', { class: 'wlg-lede' },
        'Trace the path from input to output. Explore the layers, see where the parameters live, and understand what training changes. ',
        'Start with a model below. Only its file index is read; the weights stay where they are.'),
      h('div', { class: 'wlg-search' }, input, h('button', { class: 'wl-btn primary', onclick: go }, 'Open model')),
      h('div', { class: 'wlg-slot' }));

    const all = groups.flatMap((g) => g.items);
    const feat = all.find((m) => m.id === 'deepseek-ai/DeepSeek-V3') || all[0];
    const hero = h('div', { class: 'wlg-hero' }, head, feat ? h('button', { class: 'wlg-feature', onclick: () => opts.onOpen(feat.id), 'aria-label': 'Open ' + feat.name },
      h('div', { class: 'wlx-plate-frame' }, plate(feat, {})),
      h('div', { class: 'wlg-feature-cap' }, h('b', null, feat.name), ` · ${short(feat.total)} parameters. Three ordinary blocks, then ${feat.stacks[0].blocks.length - 3} blocks of hatched experts.`)) : null);
    const body = h('div', { class: 'wlg-groups' });
    const search = h('input', { type: 'search', class: 'wlg-input', placeholder: 'Find a model, maker, or capability…', 'aria-label': 'Filter model gallery' });
    const sort = h('select', { class: 'wl-select', 'aria-label': 'Sort models' },
      h('option', { value: 'curated' }, 'Curated order'), h('option', { value: 'small' }, 'Smallest first'), h('option', { value: 'large' }, 'Largest first'));
    const count = h('span', { class: 'wl-result-count', role: 'status' });
    const filters = h('div', { class: 'wl-filter-chips', role: 'group', 'aria-label': 'Model capability' });
    let category = 'All models';
    const apply = () => {
      let visible = 0;
      body.querySelectorAll('.wlg-group').forEach((section) => {
        const cards = [...section.querySelectorAll('.wlg-card')];
        cards.sort((a, b) => sort.value === 'curated' ? +a.dataset.order - +b.dataset.order :
          (+a.dataset.total - +b.dataset.total) * (sort.value === 'large' ? -1 : 1));
        let n = 0;
        cards.forEach((card) => {
          card.hidden = (category !== 'All models' && section.dataset.group !== category) || !card.dataset.search.includes(search.value.trim().toLowerCase());
          if (!card.hidden) { n++; visible++; }
          card.parentNode.append(card);
        });
        section.hidden = !n;
      });
      count.textContent = `${visible} of ${all.length} models`;
      empty.hidden = visible > 0;
    };
    ['All models', ...groups.map((g) => g.title)].forEach((label) => {
      const button = h('button', { class: 'wl-chip', 'aria-pressed': String(label === category), onclick: () => {
        category = label;
        [...filters.children].forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
        apply();
      } }, label);
      filters.append(button);
    });
    const empty = h('div', { class: 'wl-empty', hidden: true }, h('h3', null, 'No matching models'),
      h('p', null, 'Try another name or reset the filters.'), h('button', { class: 'wl-btn', onclick: () => {
        search.value = ''; filters.firstElementChild.click();
      } }, 'Reset filters'));
    search.addEventListener('input', apply);
    sort.addEventListener('change', apply);
    const tools = h('section', { class: 'wlg-tools', 'aria-label': 'Explore models' },
      h('div', { class: 'wl-section-heading' }, h('h2', null, 'Model catalog'), count),
      h('div', { class: 'wlg-toolbar' }, search, sort), filters);
    for (const g of groups) {
      const grid = h('div', { class: 'wlg-grid' });
      for (const [order, m] of g.items.entries()) {
        grid.append(h('button', { class: 'wlg-card', 'data-order': order, 'data-total': m.total, 'data-search': `${m.name} ${m.maker} ${m.blurb} ${g.title}`.toLowerCase(), onclick: () => opts.onOpen(m.id), title: m.id },
          h('div', { class: 'wlg-sil' }, plate(m, { mini: true })),
          h('div', { class: 'wlg-meta' },
            h('div', { class: 'wlg-name' }, m.name),
            h('div', { class: 'wlg-maker' }, m.maker),
            h('p', { class: 'wlg-blurb' }, m.blurb),
            h('div', { class: 'wlg-size' }, m.active ? `${short(m.total)} parameters · ${short(m.active)} active` : `${short(m.total)} parameters`))));
      }
      body.append(h('section', { class: 'wlg-group', 'data-group': g.title }, h('h2', { class: 'wlg-gtitle' }, g.title), grid));
    }
    const foot = h('footer', { class: 'wlg-foot' },
      'Silhouettes come from each model’s tensor shapes. Width uses a square-root scale, normalized separately for each model; compare parameter counts for absolute size.');
    mount.textContent = '';
    mount.append(h('div', { class: 'wl wlg' }, hero, tools, body, empty, foot));
    apply();
  }

  function loading(mount, id) {
    mount.textContent = '';
    mount.append(h('div', { class: 'wl wl-loading' },
      h('div', { class: 'wl-spinner', 'aria-hidden': 'true' }),
      h('div', { class: 'wl-loading-t' }, 'Reading ', h('b', null, id)),
      h('div', { class: 'wl-loading-s' }, 'Fetching the index at the front of each weight file. The weights themselves stay on Hugging Face.')));
  }

  // ---------- explain view ----------
  function lede(a, name) {
    const st = a.stacks, k = a.kind.key;
    const by = (r) => st.find((s) => s.role === r);
    const n = (s) => s ? s.n : 0;
    const total = words(a.total);
    if (k === 'moe') {
      const s = st.find((x) => x.experts);
      return [name, ` holds ${total} `, T('parameter', 'parameters'), ', but each ', T('token'), ` only passes through about ${words(a.active)} of them. `,
        `Each of its ${s.n} `, T('block', 'blocks'), ` holds ${s.experts} `, T('expert', 'experts'), ` and uses ${s.activeExperts || 'a few'} for any one token.`];
    }
    if (k === 'vlm') return [name, ` has ${total} `, T('parameter', 'parameters'), `. A ${n(by('vision'))}-block vision `, T('encoder'),
      ` turns pictures into something its ${n(by('decoder'))}-block language model can read, and the language model writes the answer.`];
    if (k === 'speech') return [name, ` has ${total} `, T('parameter', 'parameters'), `. A ${n(by('audio'))}-block `, T('encoder'),
      ` listens to the audio, and a ${n(by('decoder'))}-block `, T('decoder'), ' writes down what it heard, one token at a time.'];
    if (k === 'reader') return [name, ` has ${total} `, T('parameter', 'parameters'), ` in ${n(by('encoder'))} `, T('block', 'blocks'),
      ' that read a whole passage at once, in both directions. It is built to understand text, not to write it.'];
    if (k === 'match') return [name, ` has ${total} `, T('parameter', 'parameters'), ' in two towers side by side: one reads images, one reads text. ',
      'A picture and a caption that fit together come out as matching lists of numbers.'];
    if (k === 'llm') return [name, ` is ${total} learned numbers, called `, T('parameter', 'parameters'), `, arranged as ${n(by('decoder'))} identical `,
      T('block', 'blocks'), '. It reads text and writes the next ', T('token'), ', one at a time.'];
    return [name, ` has ${total} `, T('parameter', 'parameters'), '. ', a.kind.does];
  }

  function facts(a) {
    const main = a.stacks.find((s) => s.role === 'decoder') || a.stacks[a.stacks.length - 1];
    const out = [['parameters', words(a.total)]];
    if (a.active) out.push(['used per token', words(a.active)]);
    if (main) out.push([a.stacks.length > 1 ? 'blocks in total' : 'blocks', String(a.stacks.reduce((n, s) => n + s.n, 0))]);
    if (a.vocab) out.push(['tokens it knows', int(a.vocab)]);
    if (a.context) out.push(['context (tokens at once)', int(a.context)]);
    out.push(['download size', gb(a.fileSize)]);
    return h('dl', { class: 'wlx-facts' }, out.map(([k, v]) => h('div', null, h('dd', null, v), h('dt', null, k))));
  }

  function shareBar(parts) {
    const tot = parts.reduce((n, p) => n + p[1], 0) || 1;
    return h('div', { class: 'wlx-share', role: 'img', 'aria-label': parts.map((p) => `${p[0]} ${Math.round((100 * p[1]) / tot)}%`).join(', ') },
      parts.map(([name, v, cls]) => h('span', { class: 'sh ' + cls, style: `flex-grow:${v / tot}` })));
  }

  function stackStage(a, s, si) {
    const dec = s.role === 'decoder';
    const pctA = Math.round((100 * s.attn) / Math.max(s.attn + s.mlp, 1));
    const title = {
      vision: `${s.n} vision blocks study the patches`,
      audio: `${s.n} blocks listen to the sound`,
      encoder: `${s.n} blocks read the whole text`,
      decoder: `${s.n} blocks, one after another`,
    }[s.role];
    const attnText = [
      'Each ', s.role === 'vision' ? T('patch') : T('token'), ' looks at the others and takes in what matters to it.',
      s.heads ? ` It does this ${s.heads} ways at once (${s.heads} “heads”)` : '',
      s.heads && s.kvHeads && s.kvHeads < s.heads ? `, sharing ${s.kvHeads} sets of notes between them to save memory.` : (s.heads ? '.' : ''),
      dec ? ' It can only look at earlier tokens, which is what lets it write left to right.' : ' It looks in both directions, at the whole input at once.',
      s.cross ? ' Each block also looks back at the encoder’s notes on the input.' : '',
    ];
    let mlpText;
    if (s.experts) {
      mlpText = ['Instead of one feed-forward layer, every block holds ', h('b', null, `${s.experts} `), T('expert', 'experts'),
        s.sharedExperts ? ` plus ${s.sharedExperts} shared one${s.sharedExperts > 1 ? 's' : ''} every token uses` : '',
        '. A small ', T('router'), ` sends each token to ${s.activeExperts || 'a few'} of them, so most experts sit idle for any one token.`,
        s.denseBlocks ? ` The first ${s.denseBlocks} block${s.denseBlocks > 1 ? 's use' : ' uses'} one ordinary feed-forward layer instead.` : ''];
    } else {
      mlpText = ['Then each token is processed on its own',
        s.ffn && s.hidden ? `: widened from ${int(s.hidden)} to ${int(s.ffn)} numbers, transformed, and narrowed back` : '',
        '. Much of what the model has learned is stored here.'];
    }
    const stats = [s.hidden && `width ${int(s.hidden)}`, s.ffn && !s.experts && `feed-forward ${int(s.ffn)}`,
      s.expertFfn && `each expert ${int(s.expertFfn)}`, s.heads && `${s.heads} heads`, `${short(s.params)} parameters`].filter(Boolean);
    return {
      k: ['stack:' + si], title, tech: `${s.prefix}.[0…${s.n - 1}]`,
      body: [
        h('p', null, 'Every ', T('block'), ' has the same design but its own learned numbers. Each one refines every ',
          s.role === 'vision' ? 'patch’s' : 'token’s', ' list of numbers a little more, in two steps:'),
        h('div', { class: 'wlx-pair' },
          h('div', { class: 'wlx-sub attn' }, h('div', { class: 'wlx-sub-h' }, h('i'), T('attention', 'Attention'), h('span', null, pctA + '%')), h('p', null, attnText)),
          h('div', { class: 'wlx-sub mlp' + (s.experts ? ' moe' : '') }, h('div', { class: 'wlx-sub-h' }, h('i'), s.experts ? T('expert', 'Experts') : T('feed-forward', 'Feed-forward'), h('span', null, (100 - pctA) + '%')), h('p', null, mlpText))),
        shareBar([['attention', s.attn, 'attn'], [s.experts ? 'experts' : 'feed-forward', s.mlp, 'mlp' + (s.experts ? ' moe' : '')]]),
        h('div', { class: 'wlx-stats' }, stats.join('  ·  ')),
      ],
    };
  }

  function stages(a) {
    const out = [];
    const st = a.stacks;
    const textStack = st.find((s) => s.role === 'decoder') || st.find((s) => s.role === 'encoder');
    const tied = a.output.tied;
    const embedParams = a.vocab && a.embedDim ? a.vocab * a.embedDim : 0;
    const textIn = () => {
      out.push({ k: [], title: textStack && textStack.role === 'decoder' && st.some((s) => s.role === 'audio') ? 'The words written so far' : 'Text comes in',
        tech: 'tokenizer',
        body: [h('p', null, 'Text is cut into ', T('token', 'tokens'), ': whole words, pieces of words, or single characters. ',
          a.vocab ? ['This model’s ', T('vocabulary'), ` has `, h('b', null, int(a.vocab)), ' of them, and each has an ID number.'] : '')] });
      if (a.embedDim) out.push({ k: ['embed'], title: 'Each token becomes a list of numbers', tech: `embedding · ${int(a.vocab)} × ${int(a.embedDim)}`,
        body: [h('p', null, 'The model looks up each token ID in its vocabulary table and swaps it for that token’s ', T('embedding'), ': ',
          h('b', null, `${int(a.embedDim)} numbers`), ' learned in training. From here on, the model only works with numbers.'),
        vector(a.embedDim),
        h('p', { class: 'wlx-note' }, `The table alone holds ${words(embedParams)} parameters, ${Math.round((100 * a.where.embed) / a.total)}% of the model.`)] });
    };

    if (a.kind.key === 'match') {
      const vs = st.findIndex((s) => s.role === 'vision');
      if (a.image) out.push(imageIn(a, st[vs]));
      out.push(stackStage(a, st[vs], vs));
      textIn();
      const ts = st.findIndex((s) => s.role !== 'vision');
      if (ts >= 0) out.push(stackStage(a, st[ts], ts));
    } else {
      st.forEach((s, i) => {
        if (s.role === 'vision' && a.image) out.push(imageIn(a, s));
        if (s.role === 'audio') out.push({ k: [], title: 'Sound comes in', tech: a.audio ? `log-mel spectrogram · ${a.audio.mels} bands` : 'audio',
          body: [h('p', null, 'The audio is turned into a spectrogram, a picture of the sound with ',
            h('b', null, `${a.audio ? a.audio.mels : 'many'} frequency bands`), ' over time. Two small filters squeeze each moment into a list of ',
            s.hidden ? int(s.hidden) : '', ' numbers.')] });
        if (s === textStack) textIn();
        out.push(stackStage(a, s, i));
        const next = st[i + 1];
        if (s.role === 'vision' && next && next.role === 'decoder') out.push({ k: [], title: 'Image notes are handed to the language model',
          tech: 'connector · projector',
          body: [h('p', null, 'A small connector translates the vision blocks’ output',
            s.hidden && next.hidden ? ` (${int(s.hidden)} numbers per patch) into the language model’s format (${int(next.hidden)} numbers)` : '',
            '. The image now sits in the text like a run of extra tokens, and the language model reads both together.')] });
      });
    }

    const o = a.output;
    if (o.kind === 'next-token') out.push({ k: tied ? ['embed'] : ['head'], title: 'Scoring the next token', tech: tied ? 'output · tied to the embedding table' : `lm_head · ${int(o.vocab)} scores`,
      body: [h('p', null, 'The last token’s numbers are compared against all ', h('b', null, int(o.vocab)), ' tokens in the vocabulary, giving each one a score. ',
        'One likely token is picked and added to the text. Then the whole trip runs again for the next one.'),
      h('p', { class: 'wlx-note' }, tied ? `It reuses the vocabulary table from the start for this, which saves ${words(embedParams)} parameters.`
        : `It has its own separate output table: ${words(embedParams)} more parameters.`),
      loop()] });
    else if (o.kind === 'masked') out.push({ k: [], title: 'What comes out', tech: 'masked-word head',
      body: [h('p', null, 'In training, some words were hidden and the model learned to guess them from both sides. ',
        'In practice, people use the numbers it produces for each token to search, sort and compare text.')] });
    else if (o.kind === 'match') out.push({ k: [], title: 'Comparing picture and caption', tech: 'similarity score',
      body: [h('p', null, 'Each tower ends in one list of numbers. If an image and a caption belong together, their lists point the same way, and the model scores how close they are.')] });
    else if (o.kind === 'classes') out.push({ k: [], title: 'Picking a category', tech: `classifier · ${o.n} classes`,
      body: [h('p', null, `The final numbers are scored against ${o.n} categories, and the highest score wins.`)] });
    else out.push({ k: [], title: 'What comes out', tech: 'features',
      body: [h('p', null, 'A list of numbers that describes the input. Other programs use it for search, comparison, or as the first stage of a bigger model.')] });
    return out;
  }
  function imageIn(a, s) {
    const im = a.image;
    return { k: [], title: 'The picture comes in', tech: `patch embedding · ${im.patch}×${im.patch} px`,
      body: [h('p', null, 'The image is ', im.size ? `resized to ${im.size}×${im.size} pixels and ` : '', 'cut into ',
        h('b', null, `${im.patch}×${im.patch}-pixel `), T('patch', 'patches'),
        im.size ? ` (${(im.size / im.patch) ** 2} of them)` : '', '. Each patch becomes a list of ', s && s.hidden ? int(s.hidden) + ' ' : '',
        'numbers, the same way a word does.'), patches(im)] };
  }
  // a token's embedding, drawn as a strip of cells (shade is decorative, not data)
  function vector(dim) {
    const n = Math.min(48, dim);
    const cells = [];
    for (let i = 0; i < n; i++) cells.push(h('span', { style: `--v:${(0.25 + 0.75 * Math.abs(Math.sin(i * 12.9898) % 1)).toFixed(2)}` }));
    return h('div', { class: 'wlx-vec', 'aria-hidden': 'true' }, h('div', { class: 'wlx-vec-tok' }, 'token'), h('div', { class: 'wlx-vec-arrow' }, '→'),
      h('div', { class: 'wlx-vec-cells' }, cells), h('div', { class: 'wlx-vec-n' }, dim > n ? `… ${int(dim)} numbers` : `${dim} numbers`));
  }
  function patches(im) {
    const n = im.size ? Math.min(8, im.size / im.patch) : 6;
    const g = h('div', { class: 'wlx-patches', 'aria-hidden': 'true', style: `--n:${n}` });
    for (let i = 0; i < n * n; i++) g.append(h('span'));
    return g;
  }
  function loop() {
    return h('div', { class: 'wlx-loop', 'aria-hidden': 'true' },
      ['The', 'cat', 'sat', 'on'].map((w) => h('span', { class: 'tok' }, w)), h('span', { class: 'tok next' }, 'the'), h('span', { class: 'wlx-loop-l' }, '↺ repeat'));
  }

  // where the parameters sit: 100 cells, largest-remainder split
  const ROLES = [
    ['embed', 'Vocabulary table', 'the list of numbers for every token'],
    ['vision', 'Vision encoder', 'turns images into numbers'],
    ['audio', 'Audio encoder', 'turns sound into numbers'],
    ['connector', 'Connector', 'translates image notes for the language model'],
    ['attn', 'Attention', 'how tokens look at each other'],
    ['mlp', 'Feed-forward', 'where much of the knowledge is stored'],
    ['head', 'Output table', 'scores every possible next token'],
    ['other', 'Norms and small parts', 'keep the numbers in a steady range'],
  ];
  function paperDiagram(a, stackIndex, blockIndex, onSelect) {
    const stack = a.stacks[stackIndex], block = stack.blocks[blockIndex];
    const width = 380, center = 190, left = 92, boxWidth = 196;
    const markerId = 'wl-paper-arrow-' + (++PID);
    const svg = sv('svg', { class: 'wl-paper-diagram', viewBox: '0 0 380 690', role: 'group', 'aria-label': `${stack.role} architecture diagram` });
    const marker = sv('marker', { id: markerId, viewBox: '0 0 10 10', refX: '8', refY: '5', markerWidth: '5', markerHeight: '5', orient: 'auto-start-reverse' },
      sv('path', { d: 'M 0 0 L 10 5 L 0 10 z', class: 'pd-arrowhead' }));
    svg.append(sv('defs', null, marker));
    const text = (value, x, y, className) => {
      const node = sv('text', { x, y, class: className || 'pd-text', 'text-anchor': 'middle' });
      node.textContent = value;
      return node;
    };
    const line = (path, arrow = true) => svg.append(sv('path', { d: path, class: 'pd-line', 'marker-end': arrow ? `url(#${markerId})` : null }));
    const node = (title, subtitle, y, kind, key) => {
      const group = sv('g', { class: 'pd-node pd-' + kind, 'data-k': key, role: key ? 'button' : null, tabindex: key ? '0' : null, 'aria-label': subtitle ? title + ', ' + subtitle : title });
      group.append(sv('rect', { x: left, y, width: boxWidth, height: 46, rx: 3 }), text(title, center, y + (subtitle ? 19 : 28)));
      if (subtitle) group.append(text(subtitle, center, y + 35, 'pd-dim'));
      if (key) {
        group.addEventListener('click', () => onSelect(key));
        group.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(key); }
        });
      }
      svg.append(group);
    };
    const sum = (y) => svg.append(sv('circle', { cx: center, cy: y, r: 10, class: 'pd-sum' }), text('+', center, y + 4, 'pd-plus'));
    const residual = (before, after) => {
      line(`M ${center} ${before} H 68 V ${after} H ${center - 11}`);
      svg.append(sv('circle', { cx: center, cy: before, r: 2.5, class: 'pd-dot' }));
      sum(after);
    };
    const key = 'stack:' + stackIndex;
    const textStack = stack.role === 'decoder' || stack.role === 'encoder';
    const input = stack.role === 'vision' ? 'Image patches' : stack.role === 'audio' ? 'Audio features' : 'Token IDs';
    const dimensions = stack.hidden ? 'd = ' + int(stack.hidden) : '';
    svg.append(text(input, center, 21));
    line(`M ${center} 30 V 50`);
    node(textStack ? 'Token embedding' : 'Input projection',
      textStack && a.vocab && stack.hidden ? `${int(a.vocab)} × ${int(stack.hidden)}` : dimensions, 52, 'embedding', textStack && a.where.embed ? 'embed' : key);
    line(`M ${center} 99 V 175`);
    svg.append(text(dimensions, center + 55, 133, 'pd-dim'));
    const groupBottom = stack.cross ? 514 : 410;
    svg.append(sv('rect', { x: 44, y: 145, width: 284, height: groupBottom - 130, rx: 4, class: 'pd-block' }));
    const repeatLabel = text(`× ${stack.n}`, 314, 165, 'pd-caption');
    repeatLabel.setAttribute('text-anchor', 'end');
    svg.append(repeatLabel);
    node(stack.role === 'decoder' && a.output.kind === 'next-token' ? 'Causal self-attention' : 'Self-attention',
      stack.heads ? `${stack.heads} heads${stack.kvHeads && stack.kvHeads !== stack.heads ? ` · ${stack.kvHeads} KV heads` : ''}` : dimensions, 194, 'attention', key);
    line(`M ${center} 177 V 192`);
    line(`M ${center} 241 V 258`);
    residual(180, 270);
    let feedY = 308;
    if (stack.cross) {
      line(`M ${center} 281 V 296`);
      node('Cross-attention', 'Encoder features → decoder', 298, 'attention', key);
      line(`M ${center} 345 V 362`);
      residual(284, 374);
      line('M 368 321 H 290');
      svg.append(text('encoder', 346, 307, 'pd-dim'));
      feedY += 104;
    }
    line(`M ${center} ${stack.cross ? 385 : 281} V ${feedY - 2}`);
    const experts = block.experts > 0;
    const feedLabel = experts ? 'Mixture of experts' : 'Feed-forward';
    const feedDim = experts ? `${stack.experts} experts${stack.activeExperts ? ` · ${stack.activeExperts} active` : ''}` :
      stack.hidden && stack.ffn ? `${int(stack.hidden)} → ${int(stack.ffn)} → ${int(stack.hidden)}` : dimensions;
    node(feedLabel, feedDim, feedY, 'feedforward', key);
    line(`M ${center} ${feedY + 47} V ${feedY + 64}`);
    residual(feedY - 15, feedY + 76);
    line(`M ${center} ${feedY + 87} V ${groupBottom + 40}`);
    const outputY = groupBottom + 42;
    const isFinalText = stack.role === 'decoder' || (stack.role === 'encoder' && !a.stacks.some((s) => s.role === 'decoder'));
    const tokenOutput = isFinalText && (a.output.kind === 'next-token' || a.output.kind === 'masked');
    const outputKey = tokenOutput ? (a.output.tied ? 'embed' : 'head') : key;
    node(tokenOutput ? 'Vocabulary projection' : 'Output representation', tokenOutput && a.output.vocab ? `${int(a.output.vocab)} scores` : dimensions, outputY, 'output', outputKey);
    line(`M ${center} ${outputY + 47} V ${outputY + 70}`);
    svg.append(text(tokenOutput ? 'Token scores' : stack.role === 'vision' ? 'Image features' : stack.role === 'audio' ? 'Audio features' : 'Hidden states', center, outputY + 88));
    svg.setAttribute('viewBox', `0 0 ${width} ${outputY + 108}`);
    return svg;
  }

  function trainabilityView(a, parts) {
    const setup = h('select', { class: 'wl-select', 'aria-label': 'Training setup' },
      h('option', { value: 'full' }, 'Full fine-tuning'),
      h('option', { value: 'frozen' }, 'Frozen for inference'),
      h('option', { value: 'custom' }, 'Choose trainable parts'));
    const stats = h('dl', { class: 'wl-trainability-stats', role: 'status' });
    const bar = h('div', { class: 'wl-trainability-bar', role: 'img' });
    const checks = [];
    const update = () => {
      const trainable = parts.reduce((sum, part, i) => sum + (checks[i].checked ? part[3] : 0), 0);
      const frozen = a.total - trainable;
      const percent = 100 * trainable / a.total;
      stats.replaceChildren(
        h('div', null, h('dt', null, 'Total parameters'), h('dd', { 'data-count': 'total' }, int(a.total))),
        h('div', null, h('dt', null, 'Trainable'), h('dd', { 'data-count': 'trainable' }, int(trainable))),
        h('div', null, h('dt', null, 'Non-trainable · frozen'), h('dd', { 'data-count': 'frozen' }, int(frozen))));
      bar.setAttribute('aria-label', `${percent.toFixed(1)}% trainable, ${(100 - percent).toFixed(1)}% frozen`);
      bar.replaceChildren(h('span', { class: 'trainable', style: `width:${percent}%` }), h('span', { class: 'frozen', style: `width:${100 - percent}%` }));
    };
    const choices = h('div', { class: 'wl-trainability-parts', role: 'group', 'aria-label': 'Trainable parts', hidden: true }, parts.map(([key, label, , count]) => {
      const check = h('input', { type: 'checkbox', checked: true, 'data-train-role': key, onchange: update });
      checks.push(check);
      return h('label', null, check, h('span', null, label), h('span', { class: 'wl-part-count' }, short(count)));
    }));
    setup.addEventListener('change', () => {
      choices.hidden = setup.value !== 'custom';
      if (setup.value !== 'custom') checks.forEach((check) => { check.checked = setup.value === 'full'; });
      update();
    });
    update();
    return h('div', { class: 'wl-trainability' },
      h('div', { class: 'wl-trainability-heading' }, h('label', null, 'Training setup', setup)),
      stats, bar, choices);
  }

  function whereView(a) {
    const moe = a.stacks.some((s) => s.experts);
    const roles = ROLES.map(([k, l, d]) => [k, k === 'mlp' && moe ? 'Experts' : l, k === 'mlp' && moe ? 'many feed-forward layers; each token uses a few' : d, a.where[k] || 0])
      .filter((r) => r[3] > 0);
    const exact = roles.map((r) => (100 * r[3]) / a.total);
    const cells = exact.map(Math.floor);
    let left = 100 - cells.reduce((x, y) => x + y, 0);
    exact.map((v, i) => [v - Math.floor(v), i]).sort((x, y) => y[0] - x[0]).forEach(([, i]) => { if (left > 0) { cells[i]++; left--; } });
    const grid = h('div', { class: 'wlx-waffle', role: 'group', 'aria-label': roles.map((r, i) => `${r[1]} ${exact[i].toFixed(1)}%`).join(', ') });
    roles.forEach((r, i) => { for (let c = 0; c < cells[i]; c++) grid.append(h('span', { class: 'c r-' + r[0] + (r[0] === 'mlp' && moe ? ' moe' : ''), 'data-r': r[0] })); });
    const readout = h('div', { class: 'wl-insight', role: 'status' });
    let selected = null;
    const buttons = [];
    const select = (key) => {
      selected = key === selected ? null : key;
      grid.classList.toggle('focusing', !!selected);
      grid.querySelectorAll('.c').forEach((c) => c.classList.toggle('lit', c.dataset.r === selected));
      buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.r === selected)));
      const i = roles.findIndex((r) => r[0] === selected);
      readout.textContent = i < 0 ? `${short(a.total)} parameters in total. Select a part to isolate its share.` :
        `${roles[i][1]}: ${int(roles[i][3])} parameters · ${exact[i].toFixed(2)}% of this model. ${cells[i] ? cells[i] + ' of the 100 squares (rounded).' : 'Less than one square; too small to draw at this scale.'}`;
    };
    const legend = h('ul', { class: 'wlx-legend' }, roles.map((r, i) => {
      const button = h('button', { class: 'wl-role-button', 'data-r': r[0], 'aria-pressed': 'false', onclick: () => select(r[0]) },
        h('i', { class: 'r-' + r[0] + (r[0] === 'mlp' && moe ? ' moe' : '') }),
        h('span', { class: 'lg-n' }, r[1]),
        h('span', { class: 'lg-p' }, exact[i] < 1 ? '<1%' : Math.round(exact[i]) + '%'),
        h('span', { class: 'lg-d' }, `${short(r[3])} · ${r[2]}`));
      buttons.push(button);
      return h('li', null, button);
    }));
    grid.addEventListener('click', (event) => { const cell = event.target.closest('[data-r]'); if (cell) select(cell.dataset.r); });
    select(null);
    return h('section', { class: 'wlx-sec', id: 'wl-parameters' },
      h('div', { class: 'wl-eyebrow' }, 'Parameter explorer'),
      h('h2', null, 'Parameters'), h('details', { class: 'wl-details' }, h('summary', null, 'Plan fine-tuning'), trainabilityView(a, roles)),
      h('p', { class: 'wlx-sec-lede' }, 'Each square represents about 1%. Select a category or a square to isolate it.'),
      h('div', { class: 'wlx-where' }, h('div', null, grid, readout), legend));
  }

  // what it takes to run: weights must fit in memory (×1.2 for the runtime around them)
  const DEVICES = [['Phone', 8], ['Laptop', 16], ['Gaming GPU', 24], ['Data-center GPU', 80], ['8-GPU server', 640]];
  function runView(a) {
    const P = a.total;
    const rows = [['16-bit', 'as trained', 2], ['8-bit', 'quantized', 1], ['4-bit', 'quantized', 0.5]];
    const table = h('table', { class: 'wlx-run' },
      h('thead', null, h('tr', null, h('th', null, 'Each number stored as'), h('th', null, 'Memory needed'),
        DEVICES.map(([d, g]) => h('th', { class: 'dev' }, d, h('small', null, `${g} GB`))))),
      h('tbody', null, rows.map(([bits, note, bpp]) => {
        const need = P * bpp * 1.2;
        return h('tr', null, h('th', { scope: 'row' }, bits, h('small', null, note)), h('td', { class: 'need' }, gb(need)),
          DEVICES.map(([d, g]) => { const ok = need <= g * 1e9; return h('td', { class: ok ? 'ok' : 'no', 'aria-label': ok ? 'fits' : 'does not fit' }, ok ? '●' : '·'); }));
      })));
    const precision = h('select', { class: 'wl-select', 'aria-label': 'Weight precision' }, rows.map(([bits, note, bpp]) => h('option', { value: bpp }, bits + ' · ' + note)));
    const device = h('select', { class: 'wl-select', 'aria-label': 'Available hardware' }, DEVICES.map(([name, size]) => h('option', { value: size }, `${name} · ${size} GB`)));
    device.value = '16';
    const result = h('div', { class: 'wl-memory-result', role: 'status' });
    const meter = h('div', { class: 'wl-memory-meter', role: 'meter', 'aria-label': 'Estimated memory use', 'aria-valuemin': '0' }, h('span'));
    const update = () => {
      const need = P * Number(precision.value) * 1.2, capacity = Number(device.value) * 1e9;
      const fits = need <= capacity;
      result.classList.toggle('fits', fits);
      result.replaceChildren(h('strong', null, gb(need)), h('span', null, fits ? `Fits · ${(capacity === need ? '0 GB' : gb(capacity - need))} remaining` : `${gb(need - capacity)} over available memory`));
      meter.setAttribute('aria-valuemax', String(capacity / 1e9));
      meter.setAttribute('aria-valuenow', String(Math.min(need, capacity) / 1e9));
      meter.setAttribute('aria-valuetext', `${gb(need)} needed out of ${gb(capacity)} available`);
      meter.classList.toggle('over', !fits);
      meter.firstElementChild.style.width = Math.min(100, need / capacity * 100) + '%';
    };
    precision.addEventListener('change', update); device.addEventListener('change', update); update();
    const sandbox = h('div', { class: 'wl-memory' },
      h('div', { class: 'wl-control-row' }, h('label', null, 'Store weights as', precision), h('label', null, 'Try on', device)), result, meter);
    const moe = a.active ? h('p', { class: 'wlx-note' }, 'Every expert has to be in memory, even though each token only uses a few. Speed, though, is closer to a ',
      words(a.active), '-parameter model, because only that much work happens per token.') : null;
    return h('section', { class: 'wlx-sec', id: 'wl-run' },
      h('div', { class: 'wl-eyebrow' }, 'Memory playground'), h('h2', null, 'Memory estimate'),
      h('p', { class: 'wlx-sec-lede' }, 'To run a model, all its parameters have to fit in memory. ', T('quantization', 'Quantization'),
        ' stores each number with fewer bits, trading a little quality for a lot of space.'),
      sandbox, h('details', { class: 'wl-details' }, h('summary', null, 'Compare all hardware and precision options'), h('div', { class: 'wlx-run-wrap' }, table)),
      moe,
      h('p', { class: 'wlx-note' }, `Teaching it something new costs more. Full fine-tuning needs about ${gb(16 * P)}. LoRA, which trains a small add-on and freezes the rest, needs about ${gb(2.2 * P)}.`));
  }

  function transparencyView(model, a) {
    const has = (role) => a.stacks.some((s) => s.role === role);
    const inputs = [has('vision') && 'Images', has('audio') && 'Audio', (a.vocab || has('encoder') || has('decoder')) && 'Text'].filter(Boolean);
    const output = {
      'next-token': 'Scores for the next token → generated text', masked: 'Scores for masked tokens',
      match: 'Image–text similarity', classes: `${a.output.n} category scores`, features: 'Feature vectors',
    }[a.output.kind] || 'Feature vectors';
    const detail = h('div', { class: 'wl-io-detail', role: 'status' });
    const entries = [
      ['Input', inputs.join(' + ') || 'Input',
        `${a.vocab ? `${int(a.vocab)} vocabulary entries. ` : ''}${a.image ? `${a.image.patch} × ${a.image.patch} pixel patches${a.image.size ? ` at ${a.image.size} px` : ''}. ` : ''}${a.audio ? `${a.audio.mels} audio frequency bands. ` : ''}The input is converted into lists of numbers for the model to process.`],
      ['Hidden layers', `${a.stacks.reduce((n, s) => n + s.n, 0)} blocks · ${short(a.total)} parameters`,
        'Each block transforms the input using its learned parameters. Attention combines information across tokens, while feed-forward layers process each token.'],
      ['Output', output,
        a.output.kind === 'next-token'
          ? `The output has ${a.output.vocab ? int(a.output.vocab) : 'vocabulary-sized'} token scores. These scores guide the choice of the next token. ${a.output.tied ? 'The vocabulary table is shared with the input embeddings.' : 'An output table maps the hidden layers to token scores.'}`
          : 'The output head converts the hidden layers into the scores or features used by this task.'],
    ];
    const buttons = [];
    const choose = (index) => {
      buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(i === index)));
      const e = entries[index];
      detail.replaceChildren(h('h3', null, e[0] + ': ' + e[1]), h('p', null, e[2]));
    };
    const flow = h('div', { class: 'wl-io-flow', role: 'group', 'aria-label': 'Inspect input, model, or output' }, entries.map((e, i) => {
      const b = h('button', { class: 'wl-io-node', 'aria-pressed': 'false', onclick: () => choose(i) },
        h('span', { class: 'wl-eyebrow' }, e[0]), h('strong', null, e[1]), h('span', { class: 'wl-io-action' }, 'Inspect ' + e[0].toLowerCase() + ' ↗'));
      buttons.push(b); return b;
    }));
    choose(0);
    return h('section', { class: 'wlx-sec wl-transparency', id: 'wl-io' },
      h('div', { class: 'wl-eyebrow' }, 'Input → model → output'), h('h2', null, 'Input & output'),
      h('p', { class: 'wlx-sec-lede' }, 'Select input, hidden layers, or output to explore how the model processes information.'),
      flow, detail);
  }

  function learningLab() {
    let weight = 0.2, updates = 0;
    const rate = h('select', { class: 'wl-select', 'aria-label': 'Example learning rate' },
      [0.01, 0.1, 0.5].map((value) => h('option', { value }, String(value))));
    rate.value = '0.1';
    const values = h('div', { class: 'wl-learning-values', role: 'status' });
    const bars = h('div', { class: 'wl-learning-bars', 'aria-hidden': 'true' });
    const update = () => {
      const error = (weight - 1) ** 2;
      values.replaceChildren(
        h('div', null, h('span', null, 'Prediction · weight × 1'), h('strong', { 'data-value': 'prediction' }, weight.toFixed(4))),
        h('div', null, h('span', null, 'Target'), h('strong', null, '1.0000')),
        h('div', null, h('span', null, 'Squared error'), h('strong', { 'data-value': 'loss' }, error.toFixed(6))));
      bars.replaceChildren(
        h('div', null, h('span', null, 'Prediction'), h('div', { class: 'wl-learning-track' }, h('i', { style: `width:${weight * 100}%` }))),
        h('div', null, h('span', null, 'Error'), h('div', { class: 'wl-learning-track loss' }, h('i', { style: `width:${error * 100}%` }))));
      counter.textContent = `${updates} weight update${updates === 1 ? '' : 's'}`;
    };
    const counter = h('span', { class: 'wl-journey-status' });
    const train = h('button', { class: 'wl-btn primary', onclick: () => {
      weight -= Number(rate.value) * 2 * (weight - 1);
      updates++; update();
    } }, 'Train one step →');
    const reset = h('button', { class: 'wl-btn', onclick: () => { weight = 0.2; updates = 0; update(); } }, 'Reset example');
    update();
    return h('div', { class: 'wl-learning-lab' },
      h('span', { class: 'wl-evidence' }, 'One-weight example'),
      h('h3', null, 'Learning experiment'),
      h('p', { class: 'wlx-note' }, 'Input 1, target 1. Each update moves one weight using the gradient of squared error. Change the learning rate to see how quickly it learns.'),
      values, bars,
      h('div', { class: 'wl-control-row' }, h('label', null, 'Learning rate', rate), train, reset, counter));
  }

  function trainingView(model, a) {
    const masked = a.output.kind === 'masked', matching = a.output.kind === 'match';
    const next = a.output.kind === 'next-token';
    const steps = [
      ['Example', matching ? 'Pair an image with a caption' : masked ? 'Hide part of an example' : next ? 'Provide a sequence and its next token' : 'Provide an input and a training target',
        matching ? 'A typical contrastive objective brings matching image and text representations closer while separating mismatched pairs.' : masked ? 'A typical masked-language objective hides tokens and asks the model to reconstruct them from surrounding text.' : next ? 'In a typical next-token objective, earlier tokens provide the input and the following token supplies the target. Multiple positions can be trained together.' : 'The example and target depend on the task and training objective.'],
      ['Prediction', 'Run the forward pass', 'The input moves through the model to produce scores or features. During training these intermediate values are used to compute gradients. During inference the learned weights usually stay fixed.'],
      ['Loss', 'Measure the error', 'A loss function compares the prediction with the target. It reduces the mismatch to a number the optimizer can minimize.'],
      ['Update', 'Adjust the trainable weights', 'Backpropagation computes gradients. An optimizer uses those gradients and a learning rate to update the trainable parameters. Full fine-tuning can update the whole model; LoRA trains smaller adapter matrices instead.'],
    ];
    let index = 0;
    const scene = h('div', { class: 'wl-training-scene', role: 'status' });
    const buttons = [];
    const draw = () => {
      buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(i === index)));
      scene.replaceChildren(h('span', { class: 'wl-evidence' }, `Step ${index + 1} / ${steps.length}`),
        h('h3', null, steps[index][1]), h('p', null, steps[index][2]),
        h('div', { class: 'wl-training-path', 'aria-hidden': 'true' }, steps.map((s, i) => h('span', { class: i === index ? 'current' : '' }, s[0]))));
      back.disabled = index === 0;
      forward.textContent = index === steps.length - 1 ? 'Repeat loop ↺' : 'Next step →';
    };
    const back = h('button', { class: 'wl-btn', onclick: () => { index--; draw(); } }, '← Previous');
    const forward = h('button', { class: 'wl-btn primary', onclick: () => { index = (index + 1) % steps.length; draw(); } }, 'Next step →');
    const tabs = h('div', { class: 'wl-training-tabs', role: 'group', 'aria-label': 'Training steps' }, steps.map((s, i) => {
      const b = h('button', { class: 'wl-chip', onclick: () => { index = i; draw(); } }, `${i + 1}. ${s[0]}`);
      buttons.push(b); return b;
    }));
    draw();
    const records = h('div', { class: 'wl-training-records' });
    for (const label of ['Training', 'Adapter']) {
      const entry = model.extras && model.extras[label];
      if (!entry || !entry.data) continue;
      records.append(h('details', { class: 'wl-details' }, h('summary', null, `Recorded ${label.toLowerCase()} settings · ${entry.file || 'sidecar JSON'}`),
        h('pre', { class: 'wl-record-json' }, JSON.stringify(entry.data, null, 2))));
    }
    return h('section', { class: 'wlx-sec', id: 'wl-training' },
      h('div', { class: 'wl-eyebrow' }, 'Learning, made visible'), h('h2', null, 'Training'),
      h('p', { class: 'wlx-sec-lede' }, 'Follow the learning loop from a training example to a weight update.'),
      tabs, scene, h('div', { class: 'wl-control-row' }, back, forward), learningLab(),
      records.children.length ? h('h3', { class: 'wl-record-heading' }, 'Training settings') : null, records);
  }

  function wordsView() {
    const list = [...used].filter((k) => GLOSSARY[k]);
    return h('section', { class: 'wlx-sec wlx-words' },
      h('h2', null, 'Words on this page'),
      h('dl', null, list.map((k) => h('div', null, h('dt', null, k), h('dd', null, GLOSSARY[k])))));
  }

  function explain(model, a) {
    used.clear();
    const name = String(model.fileName || 'This model').split('/').pop().replace(/\.(safetensors|gguf)$/i, '');
    const owner = String(model.fileName || '').includes('/') ? String(model.fileName).split('/')[0] : null;
    if (!a.stacks.length && window.WLArchitecture) {
      return h('div', { class: 'wl wlx' },
        h('section', { class: 'wlx-hero' }, h('h1', { class: 'wlx-name' }, name), facts(a)),
        h('section', { id: 'wl-anatomy', class: 'wl-engineering' }, window.WLArchitecture(model)));
    }
    if (!a.stacks.length) {
      return h('div', { class: 'wl wlx' }, h('section', { class: 'wlx-hero' },
        h('div', { class: 'wl-eyebrow' }, 'Weights only'), h('h1', { class: 'wlx-name' }, name),
        h('p', { class: 'wlx-lede' }, `${words(a.total)} parameters, but no repeating blocks were found, so there is no journey to draw. `,
          'Use the module browser to inspect the tensors.'), facts(a)), transparencyView(model, a), trainingView(model, a), whereView(a), runView(a), wordsView());
    }
    const sil = WLA.silhouette(a);
    const pl = plate(sil, { anatomy: a });
    const st = stages(a);
    const paper = h('div', { class: 'wl-paper-frame' });
    let current = 0;
    const journey = h('ol', { class: 'wlx-journey' });
    const status = h('span', { class: 'wl-journey-status', role: 'status' });
    const progress = h('div', { class: 'wl-journey-progress', 'aria-hidden': 'true' }, h('span'));
    const select = (i) => {
      current = i;
      [...journey.children].forEach((li, j) => {
        li.classList.toggle('on', i === j);
        li.querySelector('button').setAttribute('aria-expanded', String(i === j));
        li.querySelector('.wlx-stage-b').hidden = i !== j;
      });
      pl.classList.toggle('focusing', st[i].k.length > 0);
      pl.querySelectorAll('[data-k]').forEach((el) => el.classList.toggle('lit', st[i].k.includes(el.getAttribute('data-k'))));
      paper.querySelectorAll('[data-k]').forEach((el) => el.classList.toggle('selected', st[i].k.includes(el.getAttribute('data-k'))));
      status.textContent = `Step ${i + 1} of ${st.length}`;
      progress.firstElementChild.style.width = ((i + 1) / st.length * 100) + '%';
      previous.disabled = i === 0;
      next.disabled = i === st.length - 1;
    };
    st.forEach((stage, i) => {
      const panelId = 'wl-stage-' + i;
      const trigger = h('button', { class: 'wl-stage-trigger', 'aria-expanded': 'false', 'aria-controls': panelId, onclick: () => select(i) },
        h('span', { class: 'wlx-step' }, String(i + 1)), h('span', null, stage.title), h('span', { class: 'wl-stage-chevron', 'aria-hidden': 'true' }, '+'));
      journey.append(h('li', { class: 'wlx-stage' }, trigger,
        h('div', { class: 'wlx-stage-b', id: panelId, hidden: true }, h('div', { class: 'wlx-tech' }, stage.tech), stage.body)));
    });
    const previous = h('button', { class: 'wl-btn', onclick: () => select(current - 1) }, '← Previous');
    const next = h('button', { class: 'wl-btn primary', onclick: () => select(current + 1) }, 'Next →');
    const transport = h('div', { class: 'wl-journey-controls' }, previous, status, next);
    // The diagram and stepper share one selection, including keyboard activation.
    pl.querySelectorAll('[data-k]').forEach((el) => {
      const i = st.findIndex((stage) => stage.k.includes(el.getAttribute('data-k')));
      if (i < 0) return;
      el.setAttribute('role', 'button'); el.setAttribute('tabindex', '0'); el.setAttribute('aria-label', st[i].title);
      el.addEventListener('click', () => select(i));
      el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(i); } });
    });
    select(0);
    const stackPicker = h('select', { class: 'wl-select', 'aria-label': 'Stack to inspect' }, a.stacks.map((s, i) => h('option', { value: i }, `${s.role} · ${s.n} blocks`)));
    const blockSlider = h('input', { type: 'range', min: '0', max: a.stacks[0].n - 1, value: '0', 'aria-label': 'Block to inspect' });
    const blockInfo = h('div', { class: 'wl-block-info', role: 'status' });
    const inspectBlock = () => {
      const si = Number(stackPicker.value), bi = Number(blockSlider.value), stack = a.stacks[si], block = stack.blocks[bi];
      paper.replaceChildren(paperDiagram(a, si, bi, (key) => {
        const step = st.findIndex((stage) => stage.k.includes(key));
        if (step >= 0) select(step);
      }));
      blockSlider.setAttribute('aria-valuetext', `${stack.role} block ${block.i}`);
      pl.querySelectorAll('[data-block]').forEach((el) => el.classList.toggle('block-selected', el.getAttribute('data-block') === `${si}:${bi}`));
      blockInfo.replaceChildren(h('b', null, `Block ${block.i} · ${short(block.attn + block.mlp + block.norm)} parameters`),
        h('span', null, `${short(block.attn)} attention · ${short(block.mlp)} ${block.experts ? 'experts' : 'feed-forward'}`),
        h('span', null, stack.prefix + '.' + block.i));
    };
    stackPicker.addEventListener('change', () => { blockSlider.max = a.stacks[Number(stackPicker.value)].n - 1; blockSlider.value = '0'; inspectBlock(); select(st.findIndex((stage) => stage.k.includes('stack:' + stackPicker.value))); });
    blockSlider.addEventListener('input', () => { inspectBlock(); select(st.findIndex((stage) => stage.k.includes('stack:' + stackPicker.value))); });
    inspectBlock();
    const legend = h('ul', { class: 'wlx-plate-key' },
      sil.embed ? h('li', null, h('i', { class: 'k-embed' }), 'vocabulary table') : null,
      sil.head ? h('li', null, h('i', { class: 'k-head' }), 'output table') : null,
      h('li', null, h('i', { class: 'k-attn' }), 'attention'),
      h('li', null, h('i', { class: 'k-mlp' }), 'feed-forward'),
      a.stacks.some((s) => s.experts) ? h('li', null, h('i', { class: 'k-moe' }), 'experts') : null,
      a.stacks.some((s) => s.role === 'vision') ? h('li', null, h('i', { class: 'k-vision' }), 'vision') : null,
      a.stacks.some((s) => s.role === 'audio') ? h('li', null, h('i', { class: 'k-audio' }), 'audio') : null);

    const hero = h('section', { class: 'wlx-hero' },
      h('div', { class: 'wl-eyebrow' }, a.kind.label, owner ? h('span', { class: 'by' }, ` · by ${owner}`) : null),
      h('h1', { class: 'wlx-name' }, name),
      h('p', { class: 'wlx-lede' }, lede(a, name)),
      facts(a));
    const layerFrame = h('div', { class: 'wlx-plate-frame', hidden: true }, pl);
    const diagramMode = h('select', { class: 'wl-select', 'aria-label': 'Architecture style', onchange: () => {
      paper.hidden = diagramMode.value !== 'paper';
      layerFrame.hidden = diagramMode.value === 'paper';
    } }, h('option', { value: 'paper' }, 'Paper diagram'), h('option', { value: 'layers' }, 'Parameter map'));
    const body = h('section', { class: 'wlx-body wl-architecture-workspace', id: 'wl-anatomy' },
      h('figure', { class: 'wlx-plate' },
        h('div', { class: 'wl-plate-heading' }, diagramMode), paper, layerFrame,
        h('div', { class: 'wl-block-controls' }, h('label', null, 'Inspect a stack', stackPicker), h('label', null, 'Scrub through its blocks', blockSlider), blockInfo),
        h('figcaption', null, legend)),
      h('div', { class: 'wlx-journey-col' },
        h('h2', null, 'The journey of a ', a.kind.key === 'match' ? 'picture and a caption' : a.image ? 'question about an image' : a.audio ? 'sound clip' : 'token'),
        h('p', { class: 'wlx-note' }, 'Select a layer or step through the model.'), progress, transport, journey));
    const sections = [
      ['Input & output', transparencyView(model, a)], ['Architecture', window.WLArchitecture ? h('section', { id: 'wl-anatomy', class: 'wl-engineering' }, window.WLArchitecture(model)) : body],
      ['Training', trainingView(model, a)], ['Parameters', whereView(a)], ['Memory', runView(a)],
    ];
    const nav = h('div', { class: 'wl-section-nav', role: 'tablist', 'aria-label': 'Model exploration' });
    const selectSection = (index, focus) => {
      sections.forEach(([, panel], i) => {
        panel.hidden = i !== index;
        const tab = nav.children[i];
        tab.setAttribute('aria-selected', String(i === index));
        tab.tabIndex = i === index ? 0 : -1;
      });
      if (nav.isConnected) window.scrollTo(0, 0);
      if (focus) nav.children[index].focus();
    };
    sections.forEach(([label, panel], i) => {
      const id = 'wl-tab-' + i;
      panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', id); panel.tabIndex = 0;
      nav.append(h('button', { id, role: 'tab', 'aria-controls': panel.id, onclick: () => selectSection(i), onkeydown: (event) => {
        let target;
        if (event.key === 'ArrowRight') target = (i + 1) % sections.length;
        if (event.key === 'ArrowLeft') target = (i + sections.length - 1) % sections.length;
        if (event.key === 'Home') target = 0;
        if (event.key === 'End') target = sections.length - 1;
        if (target !== undefined) { event.preventDefault(); selectSection(target, true); }
      } }, label));
    });
    selectSection(0);
    const glossary = h('details', { class: 'wl-details wl-glossary' }, h('summary', null, 'Terminology reference'), wordsView());
    return h('div', { class: 'wl wlx' }, hero, nav, sections.map(([, panel]) => panel), glossary);
  }

  window.WLX = { gallery, explain, loading, plate, paperDiagram, words, short };
})();

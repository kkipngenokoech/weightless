/* Weightless anatomy — turns a header-only model ({tensors, config, …}) into the
 * plain-language facts the Explain view draws: which block stacks exist, what each
 * block spends its parameters on, how the model takes input and produces output.
 * DOM-free; exposed as WLA on window (webview) or globalThis (node: tests, gallery). */
(function (root) {
  'use strict';

  // MXFP4 experts (gpt-oss) pack two 4-bit weights per U8 byte; causal-mask
  // buffers (GPT-2's attn.bias [1,1,n,n]) are stored but are not parameters.
  function realParams(t) {
    const s = t.shape || [];
    if (/(^|\.)(running_mean|running_var|num_batches_tracked)$/.test(t.name)) return 0;
    if (s.length === 4 && s[0] === 1 && s[1] === 1) return 0;
    if (t.dtype === 'U8' && /_blocks$/.test(t.name)) return t.params * 2;
    return t.params;
  }

  // What a tensor inside a block is for. Order matters: 1-D norms first (GGUF
  // "attn_norm" is a norm, not attention), then attention, then feed-forward.
  function partOf(rest, shape) {
    const r = rest.toLowerCase();
    if (shape.length === 1 && /norm|(^|[._])ln|gamma|beta/.test(r)) return 'norm';
    if (/attn|attention|q_proj|k_proj|v_proj|o_proj|qkv|query|key|value|sinks/.test(r)) return 'attn';
    if (/mlp|ffn|feed_forward|expert|router|gate|fc1|fc2|c_fc|intermediate|output\.dense|moe|w1|w2|w3/.test(r)) return 'mlp';
    return 'norm';
  }

  const isTokEmbed = (n) => /embed_tokens|word_embeddings|(^|\.)wte\.|tok_embeddings|token_embd|token_embedding|(^|\.)shared\.weight/.test(n);
  const isPosEmbed = (n) => /(^|\.)wpe\.|embed_positions|position_embedding/.test(n);

  function modalityOf(prefix) {
    const p = prefix.toLowerCase();
    if (/vision|visual|image|vit\b|siglip|img/.test(p)) return 'vision';
    if (/audio|speech|whisper|mel/.test(p)) return 'audio';
    return null;
  }

  function mode(arr) {
    const f = new Map(); let best = null, n = 0;
    for (const v of arr) { const c = (f.get(v) || 0) + 1; f.set(v, c); if (c > n) { n = c; best = v; } }
    return best;
  }

  // config scoped to a stack: vision_config for the vision tower, text_config
  // (or the root) for the language part; Whisper keeps encoder_*/decoder_* keys.
  function cfgFor(config, role) {
    if (!config) return {};
    if (role === 'vision') return config.vision_config || {};
    if (role === 'audio') return config.audio_config || config;
    return config.text_config || config.language_config || config.llm_config || config;
  }
  const pick = (c, ...keys) => { for (const k of keys) if (c && typeof c[k] === 'number') return c[k]; return null; };

  function analyze(model) {
    const tensors = model.tensors.map((t) => ({ ...t, p: realParams(t) }));
    const total = tensors.reduce((a, t) => a + t.p, 0) || 1;
    const config = model.config || null;

    // ---- block stacks: "<prefix>.<i>.<sub>.<leaf>" with ≥2 indices ----
    const groups = new Map();
    for (const t of tensors) {
      const m = t.name.match(/^(.*?)\.(\d+)\.(.+\..+)$/);
      if (!m) continue;
      if (!groups.has(m[1])) groups.set(m[1], new Map());
      const g = groups.get(m[1]);
      if (!g.has(+m[2])) g.set(+m[2], []);
      g.get(+m[2]).push({ ...t, rest: m[3] });
    }
    const inStack = new Set();
    let stacks = [];
    for (const [prefix, blocks] of groups) {
      if (blocks.size < 2) continue;
      const idx = [...blocks.keys()].sort((a, b) => a - b);
      // a "block" must contain sub-modules (attention / feed-forward), not bare weights
      const kinds = new Set(blocks.get(idx[0]).map((t) => partOf(t.rest, t.shape)));
      if (!kinds.has('attn') && !kinds.has('mlp')) continue;
      stacks.push({ prefix, idx, blocks });
    }
    // nested numeric groups (experts.N inside layers.N) belong to their outer stack
    stacks = stacks.filter((s) => !stacks.some((o) => o !== s && s.prefix.startsWith(o.prefix + '.')));

    const audioConv = tensors.find((t) => t.shape.length === 3 && (t.shape[1] === 80 || t.shape[1] === 128) && /conv/.test(t.name));
    const out = stacks.map((s) => {
      const p = s.prefix.toLowerCase();
      let role = modalityOf(p);
      if (!role && audioConv && /encoder/.test(p) && audioConv.name.startsWith(s.prefix.replace(/\.layers?$|\.blocks$/, ''))) role = 'audio';
      if (!role) role = /encoder|(^|\.)bert\b/.test(p) ? 'encoder' : 'decoder';
      const c = cfgFor(config, role);
      const isWhisperEnc = /encoder/.test(p) && config && config.encoder_layers;
      const blocks = s.idx.map((i) => {
        const b = { i, attn: 0, mlp: 0, norm: 0, experts: 0, routed: 0, cross: false };
        const expertIds = new Set();
        for (const t of s.blocks.get(i)) {
          inStack.add(t.name);
          const part = partOf(t.rest, t.shape);
          b[part] += t.p;
          if (/encoder_attn|crossattention|cross_attn/.test(t.rest)) b.cross = true;
          if (part === 'mlp' && /expert/.test(t.rest) && !/shared/.test(t.rest)) {
            b.routed += t.p;
            const em = t.rest.match(/experts\.(\d+)\./);
            if (em) expertIds.add(em[1]);
            else if (t.shape.length >= 3) b.experts = Math.max(b.experts, t.shape[0]); // fused [E, …]
          }
        }
        if (expertIds.size) b.experts = expertIds.size;
        return b;
      });
      const b0 = s.blocks.get(s.idx[0]);
      const hidden = pick(c, isWhisperEnc ? 'd_model' : 'hidden_size', 'd_model', 'n_embd', 'hidden_dim', 'dim')
        || mode(b0.filter((t) => t.shape.length === 1 && partOf(t.rest, t.shape) === 'norm').map((t) => t.shape[0]))
        || null;
      // feed-forward width: config if present, else the non-hidden dim of the up projection
      let ffn = pick(c, isWhisperEnc ? 'encoder_ffn_dim' : 'intermediate_size', 'intermediate_size', 'ffn_dim', 'n_inner', 'mlp_dim', 'decoder_ffn_dim');
      if (!ffn && hidden) {
        const dense = s.blocks.get(s.idx.find((i) => !blocks[s.idx.indexOf(i)].experts) ?? s.idx[0]);
        const dims = dense.filter((t) => partOf(t.rest, t.shape) === 'mlp' && t.shape.length === 2)
          .map((t) => { const d = Math.max(...t.shape.filter((x) => x !== hidden), 0); return /gate_up/.test(t.rest) ? d / 2 : d; });
        ffn = dims.length ? Math.max(...dims) : null;
      }
      const E = Math.max(...blocks.map((b) => b.experts), 0) || null;
      return {
        prefix: s.prefix, role, n: blocks.length, blocks,
        params: blocks.reduce((a, b) => a + b.attn + b.mlp + b.norm, 0),
        attn: blocks.reduce((a, b) => a + b.attn, 0),
        mlp: blocks.reduce((a, b) => a + b.mlp, 0),
        hidden, ffn,
        heads: pick(c, isWhisperEnc ? 'encoder_attention_heads' : 'num_attention_heads', 'num_attention_heads', 'n_head', 'decoder_attention_heads', 'num_heads'),
        kvHeads: pick(c, 'num_key_value_heads', 'num_kv_heads'),
        experts: E,
        activeExperts: E ? pick(c, 'num_experts_per_tok', 'experts_per_token', 'num_selected_experts', 'moe_top_k') : null,
        expertFfn: E ? pick(c, 'moe_intermediate_size', 'expert_intermediate_size') || (c && c.model_type === 'gpt_oss' ? ffn : null) : null,
        sharedExperts: E ? pick(c, 'n_shared_experts', 'num_shared_experts') || (s.blocks.get(s.idx[s.idx.length - 1]).some((t) => /shared_expert/.test(t.rest)) ? 1 : 0) : 0,
        denseBlocks: E ? blocks.filter((b) => !b.experts).length : 0,
        cross: blocks.some((b) => b.cross),
      };
    });
    const ORDER = { vision: 0, audio: 0, encoder: 1, decoder: 2 };
    out.sort((a, b) => ORDER[a.role] - ORDER[b.role]);

    // ---- inputs ----
    const owner = (name) => out.slice().sort((a, b) => common(b.prefix, name) - common(a.prefix, name))[0];
    const tokEmb = tensors.filter((t) => t.shape.length === 2 && isTokEmbed(t.name) && !inStack.has(t.name))
      .sort((a, b) => b.p - a.p);
    const emb = tokEmb[0] || null;
    const textCfg = cfgFor(config, 'decoder');
    const vocab = emb ? emb.shape[0] : pick(textCfg, 'vocab_size');
    const patchConv = tensors.find((t) => t.shape.length >= 4 && t.shape[1] === 3 && /patch|embed|conv|proj|stem/.test(t.name));
    let image = null;
    if (patchConv) {
      const k = patchConv.shape[patchConv.shape.length - 1];
      const comp = patchConv.name.split('.').slice(0, -2).join('.');
      const pos = tensors.find((t) => t.shape.length === 2 && isPosEmbed(t.name) && t.name.startsWith(comp.split('.embeddings')[0]));
      let size = null;
      if (pos) for (const n of [pos.shape[0], pos.shape[0] - 1]) { const g = Math.sqrt(n); if (Number.isInteger(g)) { size = g * k; break; } }
      const vc = cfgFor(config, 'vision');
      image = { patch: k, size: pick(vc, 'image_size') || size };
    }
    const audio = audioConv ? { mels: audioConv.shape[1] } : null;
    const textPos = tensors.find((t) => t.shape.length === 2 && isPosEmbed(t.name) && !modalityOf(t.name) && !/encoder\.embed_positions/.test(t.name));
    const context = pick(textCfg, 'max_position_embeddings', 'n_positions', 'max_target_positions', 'seq_length', 'max_sequence_length')
      || (textPos ? textPos.shape[0] : null);

    // ---- output ----
    const headT = tensors.find((t) => t.shape.length === 2 && vocab && t.shape[0] === vocab && t !== emb &&
      /lm_head|(^|\.)output\.weight|proj_out|logits|embed_out/.test(t.name));
    const hasDecoder = out.some((s) => s.role === 'decoder');
    const hasVision = out.some((s) => s.role === 'vision');
    const textEncoder = out.some((s) => s.role === 'encoder');
    let output;
    if (headT) output = { kind: 'next-token', tied: false, vocab };
    else if (tensors.some((t) => /cls\.predictions|mlm|lm_head/.test(t.name)) && !hasDecoder) output = { kind: 'masked', vocab };
    else if (hasDecoder && emb) output = { kind: 'next-token', tied: true, vocab };
    else if (hasVision && textEncoder) output = { kind: 'match' };
    else {
      const cls = tensors.find((t) => t.shape.length === 2 && /classifier|score|(^|\.)head\.weight/.test(t.name) && !inStack.has(t.name));
      output = cls && cls.shape[0] !== cls.shape[1] ? { kind: 'classes', n: cls.shape[0] } : { kind: 'features' };
    }
    if (output.kind === 'next-token' && config && config.tie_word_embeddings === false && !headT) output.tied = false;

    // ---- where the parameters live ----
    const where = { embed: 0, attn: 0, mlp: 0, head: 0, vision: 0, audio: 0, connector: 0, other: 0 };
    const stackOf = new Map();
    for (const s of out) for (const b of s.blocks) for (const t of groups.get(s.prefix).get(b.i)) stackOf.set(t.name, s);
    // vision_model.encoder.layers → vision_model; Whisper's model.encoder.layers → model.encoder
    const towerRoot = (s) => { const r = s.prefix.replace(/\.(layers?|blocks|h)$/, ''); return modalityOf(r.replace(/\.encoder$/, '')) ? r.replace(/\.encoder$/, '') : r; };
    const towers = out.filter((s) => s.role === 'vision' || s.role === 'audio').map((s) => [towerRoot(s), s.role]);
    for (const t of tensors) {
      const s = stackOf.get(t.name);
      const tower = towers.find(([r]) => t.name.startsWith(r + '.'));
      if (s && (s.role === 'vision' || s.role === 'audio')) where[s.role] += t.p;
      else if (tower) where[tower[1]] += t.p;
      else if (s) { const m = t.name.slice(s.prefix.length).match(/^\.\d+\.(.+)$/); const part = partOf(m ? m[1] : t.name, t.shape); where[part === 'norm' ? 'other' : part] += t.p; }
      else if (/connector|projector|merger|mm_proj|multi_modal/.test(t.name)) where.connector += t.p;
      else if (modalityOf(t.name)) where[modalityOf(t.name)] += t.p;
      else if (t === headT || /cls\.predictions|lm_head/.test(t.name)) where.head += t.p;
      else if (isTokEmbed(t.name) || isPosEmbed(t.name) || /embed/.test(t.name)) where.embed += t.p;
      else where.other += t.p;
    }

    // ---- MoE: parameters that actually run for one token ----
    let active = null;
    for (const s of out) if (s.experts && s.activeExperts) {
      if (active == null) active = total;
      for (const b of s.blocks) if (b.experts) active -= b.routed * (b.experts - s.activeExperts) / b.experts;
    }

    // ---- storage precision, by bytes ----
    const byBytes = {};
    for (const t of tensors) byBytes[t.dtype] = (byBytes[t.dtype] || 0) + t.bytes;
    const dtype = Object.entries(byBytes).sort((a, b) => b[1] - a[1])[0];

    return {
      total, active: active ? Math.round(active) : null, stacks: out, vocab, embedDim: emb ? emb.shape[1] : null,
      image, audio, context, output, where, dtype: dtype ? dtype[0] : null,
      // summed tensor bytes beat an index's metadata.total_size (DeepSeek-V3's is 2× off)
      fileSize: tensors.reduce((a, t) => a + t.bytes, 0) || model.fileSize,
      kind: kindOf(out, output, image, audio),
      config,
    };
  }
  function common(a, b) { let i = 0; while (i < a.length && a[i] === b[i]) i++; return i; }

  function kindOf(stacks, output, image, audio) {
    const moe = stacks.some((s) => s.experts);
    if (output.kind === 'next-token' && audio) return { key: 'speech', label: 'Speech recognizer', does: 'Listens to audio and writes down what was said.' };
    if (output.kind === 'next-token' && image) return { key: 'vlm', label: 'Vision-language model', does: 'Looks at images, reads text, and writes text back.' };
    if (output.kind === 'next-token') return { key: moe ? 'moe' : 'llm', label: moe ? 'Text generator · mixture of experts' : 'Text generator', does: 'Reads text and writes more of it, one token at a time.' };
    if (output.kind === 'masked') return { key: 'reader', label: 'Text reader', does: 'Reads a whole passage at once to understand it. It fills in blanks rather than writing.' };
    if (output.kind === 'match') return { key: 'match', label: 'Image–text matcher', does: 'Scores how well a picture and a caption go together.' };
    if (image) return { key: 'vision', label: 'Vision model', does: 'Turns images into numbers that describe what is in them.' };
    if (output.kind === 'classes') return { key: 'classifier', label: 'Classifier', does: `Sorts its input into ${output.n} categories.` };
    return { key: 'net', label: 'Neural network', does: 'Turns its input into a list of numbers that describes it.' };
  }

  // Compact silhouette for the gallery: per stack, each block's [attention, feed-forward]
  // parameter counts. Enough to draw the plate without the full header.
  function silhouette(a) {
    const r = (x) => Number(x.toPrecision(3));
    return {
      total: a.total, active: a.active, kind: a.kind.key, embed: r(a.where.embed), head: r(a.where.head),
      stacks: a.stacks.map((s) => ({ role: s.role, blocks: s.blocks.map((b) => (b.experts ? [r(b.attn), r(b.mlp), 1] : [r(b.attn), r(b.mlp)])) })),
    };
  }

  root.WLA = { analyze, silhouette, realParams };
})(typeof window !== 'undefined' ? window : globalThis);

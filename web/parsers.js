/* Weightless web — DOM-free parser ports of lib/{safetensors,gguf,hub}.js
 * (browser ArrayBuffer/DataView instead of node Buffer). Exposed as WLP on
 * window (browser) or globalThis (node, for the smoke test). */
(function (root) {
  'use strict';
  const dec = new TextDecoder();
  const MAX_HEADER_BYTES = 512 * 1024 * 1024;

  // ---------- safetensors ----------
  function tensorsFromHeader(header, shard) {
    const out = [];
    for (const [name, info] of Object.entries(header)) {
      if (name === '__metadata__' || !info || !info.shape) continue;
      const shape = info.shape;
      const params = shape.reduce((a, b) => a * b, 1);
      const bytes = info.data_offsets ? info.data_offsets[1] - info.data_offsets[0] : 0;
      out.push({ name, dtype: info.dtype, shape, params, bytes, shard });
    }
    return out;
  }

  // ---------- GGUF ----------
  const GGML_TYPES = ['F32', 'F16', 'Q4_0', 'Q4_1', 'Q4_2', 'Q4_3', 'Q5_0', 'Q5_1', 'Q8_0', 'Q8_1',
    'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K', 'IQ2_XXS', 'IQ2_XS', 'IQ3_XXS', 'IQ1_S',
    'IQ4_NL', 'IQ3_S', 'IQ2_S', 'IQ4_XS', 'I8', 'I16', 'I32', 'I64', 'F64', 'IQ1_M', 'BF16'];

  function parseGGUF(buf, fileSize) {
    const dv = new DataView(buf);
    let off = 0;
    const need = (n) => { if (off + n > dv.byteLength) { const e = new Error('need more bytes'); e.code = 'EOFBUF'; throw e; } };
    const u8 = () => { need(1); return dv.getUint8(off++); };
    const i8 = () => { need(1); return dv.getInt8(off++); };
    const u16 = () => { need(2); const v = dv.getUint16(off, true); off += 2; return v; };
    const i16 = () => { need(2); const v = dv.getInt16(off, true); off += 2; return v; };
    const u32 = () => { need(4); const v = dv.getUint32(off, true); off += 4; return v; };
    const i32 = () => { need(4); const v = dv.getInt32(off, true); off += 4; return v; };
    const f32 = () => { need(4); const v = dv.getFloat32(off, true); off += 4; return v; };
    const u64 = () => { need(8); const v = Number(dv.getBigUint64(off, true)); off += 8; return v; };
    const i64 = () => { need(8); const v = Number(dv.getBigInt64(off, true)); off += 8; return v; };
    const f64 = () => { need(8); const v = dv.getFloat64(off, true); off += 8; return v; };
    const str = () => { const n = u64(); need(n); const s = dec.decode(new Uint8Array(buf, off, n)); off += n; return s; };
    function value(type) {
      switch (type) {
        case 0: return u8(); case 1: return i8();
        case 2: return u16(); case 3: return i16();
        case 4: return u32(); case 5: return i32();
        case 6: return f32(); case 7: return !!u8();
        case 8: return str();
        case 9: {
          const t = u32(); const n = u64();
          const out = [];
          for (let i = 0; i < n; i++) { const v = value(t); if (out.length < 64) out.push(v); }
          return n > 64 ? { __array: true, length: n, sample: out } : out;
        }
        case 10: return u64(); case 11: return i64(); case 12: return f64();
        default: throw new Error(`Unknown GGUF value type ${type}`);
      }
    }

    if (u32() !== 0x46554747) throw new Error('Not a GGUF file (bad magic)');
    const version = u32();
    if (version < 2 || version > 3) throw new Error(`Unsupported GGUF version ${version}`);
    const tensorCount = u64();
    const kvCount = u64();

    const kv = {};
    for (let i = 0; i < kvCount; i++) { const key = str(); const type = u32(); kv[key] = value(type); }

    const infos = [];
    for (let i = 0; i < tensorCount; i++) {
      const name = str();
      const nDims = u32();
      const dims = [];
      for (let d = 0; d < nDims; d++) dims.push(u64());
      const type = u32();
      const offset = u64();
      infos.push({ name, shape: dims.slice().reverse(), type, offset });
    }

    const alignment = kv['general.alignment'] || 32;
    const dataStart = Math.ceil(off / alignment) * alignment;
    const byOff = infos.slice().sort((a, b) => a.offset - b.offset);
    for (let i = 0; i < byOff.length; i++) {
      const end = i + 1 < byOff.length ? byOff[i + 1].offset : Math.max(fileSize - dataStart, byOff[i].offset);
      byOff[i].bytes = Math.max(0, end - byOff[i].offset);
    }

    const tensors = infos.map((t) => ({
      name: t.name,
      dtype: GGML_TYPES[t.type] || `GGML(${t.type})`,
      shape: t.shape,
      params: t.shape.reduce((a, b) => a * b, 1),
      bytes: t.bytes || 0,
      shard: null,
    }));
    return { kv, tensors };
  }

  function kvToConfig(kv) {
    const arch = kv['general.architecture'];
    if (!arch) return null;
    const g = (k) => kv[`${arch}.${k}`];
    const toks = kv['tokenizer.ggml.tokens'];
    const cfg = {
      model_type: arch,
      _name_or_path: kv['general.name'] || undefined,
      num_hidden_layers: g('block_count'),
      hidden_size: g('embedding_length'),
      intermediate_size: g('feed_forward_length'),
      num_attention_heads: g('attention.head_count'),
      num_key_value_heads: g('attention.head_count_kv'),
      max_position_embeddings: g('context_length'),
      vocab_size: g('vocab_size') || (toks && toks.length) || undefined,
      quantization: kv['general.file_type'] != null ? `ggml file_type ${kv['general.file_type']}` : undefined,
    };
    for (const k of Object.keys(cfg)) if (cfg[k] == null) delete cfg[k];
    return cfg;
  }

  function kvToMeta(kv) {
    const out = {};
    for (const [k, v] of Object.entries(kv)) {
      if (v && v.__array) out[k] = `[${v.length} items] ${JSON.stringify(v.sample.slice(0, 4))}…`;
      else if (Array.isArray(v)) out[k] = v.length > 16 ? `[${v.length} items]` : v;
      else if (typeof v === 'string' && v.length > 400) out[k] = v.slice(0, 400) + '…';
      else out[k] = v;
    }
    return out;
  }

  // ---------- Hugging Face Hub (browser fetch; CORS-friendly endpoints) ----------
  const BASE = 'https://huggingface.co';

  function normalizeId(input) {
    let s = String(input).trim();
    s = s.replace(/^https?:\/\/(www\.)?huggingface\.co\//i, '');
    s = s.replace(/[?#].*$/, '');
    const parts = s.split('/').filter(Boolean);
    if (parts.length < 2) throw new Error(`"${input}" doesn't look like a model id (expected owner/model)`);
    return parts.slice(0, 2).join('/');
  }

  function httpError(res, url) {
    const hint = res.status === 401 || res.status === 403
      ? ' — this model is gated/private. Accept its license on huggingface.co, then open it in the Weightless VS Code extension with HF_TOKEN set.'
      : res.status === 404 ? ' — repo or file not found; check the model id.' : '';
    return new Error(`HTTP ${res.status}${hint}`);
  }

  async function fetchRange(url, start, end) {
    const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
    if (!res.ok) throw httpError(res, url);
    return res.arrayBuffer();
  }
  async function fetchJson(url) {
    const res = await fetch(url);
    if (!res.ok) throw httpError(res, url);
    return res.json();
  }

  async function readRemoteHeader(url) {
    const len = new DataView(await fetchRange(url, 0, 7));
    const n = Number(len.getBigUint64(0, true));
    if (!(n > 0 && n <= MAX_HEADER_BYTES)) throw new Error(`Invalid safetensors header length (${n})`);
    const hdr = await fetchRange(url, 8, 8 + n - 1);
    return JSON.parse(dec.decode(hdr));
  }

  async function loadFromHub(rawId) {
    const id = normalizeId(rawId);
    const resolve = (f) => `${BASE}/${id}/resolve/main/${f}`;
    const info = await fetchJson(`${BASE}/api/models/${id}?revision=main`);
    const files = (info.siblings || []).map((s) => s.rfilename);

    const indexFile = files.find((f) => f === 'model.safetensors.index.json')
      || files.find((f) => f.endsWith('.safetensors.index.json') && !f.includes('/'));
    const stFiles = files.filter((f) => f.endsWith('.safetensors') && !f.includes('/'));

    let tensors = [], meta = null, shardFiles = [], fileSize = 0, sharded = false;
    if (indexFile) {
      const idx = await fetchJson(resolve(indexFile));
      shardFiles = Array.from(new Set(Object.values(idx.weight_map || {})));
      sharded = true;
      if (idx.metadata && idx.metadata.total_size) fileSize = idx.metadata.total_size;
      const headers = await Promise.all(shardFiles.map((f) => readRemoteHeader(resolve(f))));
      headers.forEach((hd, i) => {
        if (!meta && hd.__metadata__) meta = hd.__metadata__;
        tensors = tensors.concat(tensorsFromHeader(hd, shardFiles[i]));
      });
    } else {
      const f = stFiles.find((n) => n === 'model.safetensors') || stFiles[0];
      if (!f) throw new Error(`No .safetensors file found in ${id} — it may only ship .bin/.gguf weights`);
      shardFiles = [f];
      const hd = await readRemoteHeader(resolve(f));
      meta = hd.__metadata__ || null;
      tensors = tensorsFromHeader(hd, f);
    }
    if (!fileSize) fileSize = tensors.reduce((a, t) => a + t.bytes, 0);

    let config = null;
    if (files.includes('config.json')) {
      try { config = await fetchJson(resolve('config.json')); } catch (_) { /* optional */ }
    }
    const extras = {};
    const SIDE = [['Generation', 'generation_config.json'], ['Adapter', 'adapter_config.json'], ['Tokenizer', 'tokenizer_config.json']];
    await Promise.all(SIDE.map(async ([label, name]) => {
      if (!files.includes(name)) return;
      try { extras[label] = { file: name, data: await fetchJson(resolve(name)) }; } catch (_) { /* optional */ }
    }));

    return { tensors, fileSize, meta, config, extras, sharded, shardFiles };
  }

  root.WLP = { tensorsFromHeader, parseGGUF, kvToConfig, kvToMeta, normalizeId, loadFromHub, GGML_TYPES };
})(typeof window !== 'undefined' ? window : globalThis);

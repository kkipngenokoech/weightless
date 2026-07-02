/* GGUF (llama.cpp) header parsing — header-only, like the safetensors path.
 * Reads the metadata KVs + tensor infos; never touches the quantized blocks.
 * vscode-free, unit-testable. Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md */
const fs = require('fs');

const GGUF_MAGIC = 0x46554747; // 'GGUF' little-endian
const GGML_TYPES = ['F32', 'F16', 'Q4_0', 'Q4_1', 'Q4_2', 'Q4_3', 'Q5_0', 'Q5_1', 'Q8_0', 'Q8_1',
  'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K', 'IQ2_XXS', 'IQ2_XS', 'IQ3_XXS', 'IQ1_S',
  'IQ4_NL', 'IQ3_S', 'IQ2_S', 'IQ4_XS', 'I8', 'I16', 'I32', 'I64', 'F64', 'IQ1_M', 'BF16'];

class NeedMoreBytes extends Error { constructor() { super('need more bytes'); this.code = 'EOFBUF'; } }

class Reader {
  constructor(buf) { this.buf = buf; this.off = 0; }
  need(n) { if (this.off + n > this.buf.length) throw new NeedMoreBytes(); }
  u8() { this.need(1); return this.buf.readUInt8(this.off++); }
  i8() { this.need(1); return this.buf.readInt8(this.off++); }
  u16() { this.need(2); const v = this.buf.readUInt16LE(this.off); this.off += 2; return v; }
  i16() { this.need(2); const v = this.buf.readInt16LE(this.off); this.off += 2; return v; }
  u32() { this.need(4); const v = this.buf.readUInt32LE(this.off); this.off += 4; return v; }
  i32() { this.need(4); const v = this.buf.readInt32LE(this.off); this.off += 4; return v; }
  f32() { this.need(4); const v = this.buf.readFloatLE(this.off); this.off += 4; return v; }
  u64() { this.need(8); const v = this.buf.readBigUInt64LE(this.off); this.off += 8; return Number(v); }
  i64() { this.need(8); const v = this.buf.readBigInt64LE(this.off); this.off += 8; return Number(v); }
  f64() { this.need(8); const v = this.buf.readDoubleLE(this.off); this.off += 8; return v; }
  str() { const n = this.u64(); this.need(n); const s = this.buf.toString('utf8', this.off, this.off + n); this.off += n; return s; }
  value(type) {
    switch (type) {
      case 0: return this.u8(); case 1: return this.i8();
      case 2: return this.u16(); case 3: return this.i16();
      case 4: return this.u32(); case 5: return this.i32();
      case 6: return this.f32(); case 7: return !!this.u8();
      case 8: return this.str();
      case 9: { // array: item type + count + items
        const t = this.u32(); const n = this.u64();
        const out = [];
        for (let i = 0; i < n; i++) {
          const v = this.value(t);
          if (out.length < 64) out.push(v); // keep a sample; vocab arrays can be huge
        }
        return n > 64 ? { __array: true, length: n, sample: out } : out;
      }
      case 10: return this.u64(); case 11: return this.i64(); case 12: return this.f64();
      default: throw new Error(`Unknown GGUF value type ${type}`);
    }
  }
}

function parseGGUF(buf, fileSize) {
  const r = new Reader(buf);
  if (r.u32() !== GGUF_MAGIC) throw new Error('Not a GGUF file (bad magic)');
  const version = r.u32();
  if (version < 2 || version > 3) throw new Error(`Unsupported GGUF version ${version}`);
  const tensorCount = r.u64();
  const kvCount = r.u64();

  const kv = {};
  for (let i = 0; i < kvCount; i++) {
    const key = r.str();
    const type = r.u32();
    kv[key] = r.value(type);
  }

  const infos = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = r.str();
    const nDims = r.u32();
    const dims = [];
    for (let d = 0; d < nDims; d++) dims.push(r.u64());
    const type = r.u32();
    const offset = r.u64();
    // GGML stores dims fastest-first; reverse for torch-style [rows, cols]
    infos.push({ name, shape: dims.slice().reverse(), type, offset });
  }

  const alignment = kv['general.alignment'] || 32;
  const dataStart = Math.ceil(r.off / alignment) * alignment;

  // exact per-tensor bytes from offset deltas within the data section
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
  return { kv, tensors, dataStart };
}

/** Map GGUF metadata onto an HF-config-like object so the viewer's
 *  config-driven facts (layers, heads, context, KV cache) light up. */
function kvToConfig(kv) {
  const arch = kv['general.architecture'];
  if (!arch) return null;
  const g = (k) => kv[`${arch}.${k}`];
  const cfg = {
    model_type: arch,
    _name_or_path: kv['general.name'] || undefined,
    num_hidden_layers: g('block_count'),
    hidden_size: g('embedding_length'),
    intermediate_size: g('feed_forward_length'),
    num_attention_heads: g('attention.head_count'),
    num_key_value_heads: g('attention.head_count_kv'),
    max_position_embeddings: g('context_length'),
    vocab_size: g('vocab_size') || kv['tokenizer.ggml.tokens']?.length || kv['tokenizer.ggml.tokens']?.__array && kv['tokenizer.ggml.tokens'].length,
    quantization: kv['general.file_type'] != null ? `ggml file_type ${kv['general.file_type']}` : undefined,
  };
  for (const k of Object.keys(cfg)) if (cfg[k] == null) delete cfg[k];
  return cfg;
}

/** Compact, display-safe metadata (huge arrays summarized). */
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

/** Load a local .gguf — grows the read window until the header parses. */
async function loadGGUFModel(fsPath) {
  const { size } = await fs.promises.stat(fsPath);
  let cap = Math.min(size, 4 * 1024 * 1024);
  for (;;) {
    const fh = await fs.promises.open(fsPath, 'r');
    let buf;
    try { buf = Buffer.alloc(cap); await fh.read(buf, 0, cap, 0); } finally { await fh.close(); }
    try {
      const { kv, tensors } = parseGGUF(buf, size);
      return {
        tensors, fileSize: size, meta: kvToMeta(kv), config: kvToConfig(kv),
        extras: {}, sharded: false, shardFiles: [require('path').basename(fsPath)],
      };
    } catch (e) {
      if (e.code === 'EOFBUF' && cap < size) { cap = Math.min(size, cap * 4); continue; }
      throw e;
    }
  }
}

module.exports = { parseGGUF, loadGGUFModel, kvToConfig, GGML_TYPES };

# Safetensors Viewer

A VSCode extension to **visually inspect `.safetensors` models** — architecture tree,
layer-type inference, and a full tensor inventory — right inside the editor. It reads
**only the file header**, so a 1.4 GB (or 100 GB sharded) model opens instantly.

Double-click any `.safetensors` file → it opens in a rich viewer instead of failing as
"binary file not shown."

## Features

- **Architecture view** — reconstructs the **module hierarchy** from tensor names
  (`vision_model.encoder.layers.0.self_attn.q_proj` → a real nested tree), collapses
  **repeated blocks** (`layers.0…31` → `ModuleList ×32`, shown once), and **infers layer
  types** (Attention / MLP / Norm / Embedding / Conv / Linear) from names + shapes.
- **Component breakdown** — top-level modules sized by parameter share, with a
  heuristic *backbone vs. head* tag.
- **Tensor inventory** — searchable, sortable table of every tensor: name, shape, dtype,
  params, byte size (+ shard).
- **Summary** — total params, tensor count, file size, shard count, dtype distribution.
- **Sharded models** — aggregates `*.safetensors.index.json` weight maps across shards.
- **Config + metadata tabs** — pretty-prints an adjacent `config.json` and the header's
  `__metadata__`.
- **Header-only** — never loads weight bytes, so it's instant on huge files and uses no
  meaningful memory.
- Themed with VSCode variables (light/dark aware), no external runtime dependencies.

## How it works (and its one honest limit)

A `.safetensors` file is **just weights** — a header (8-byte length + JSON of every
tensor's `name`, `shape`, `dtype`, byte-offsets) followed by raw bytes. There is **no
computational graph** stored in it. This extension parses that header and reconstructs
everything you *can* get from it:

- the **module hierarchy** (parent/child nesting) — the dotted names encode it exactly;
- **repeated structure** (transformer stacks);
- **layer types** (inferred from naming/shape conventions).

What it **cannot** show from the file alone is the **true forward-pass graph** (which
module feeds which — Netron-style edges). That data flow lives in the model's *code*, not
its weights. See the roadmap for the optional Python-tracing path.

## Roadmap

- **Tier 2 — true data-flow graph (opt-in):** a Python companion that loads the model
  (`config.json` + the weights + the model class) and traces a forward pass with
  `torch.fx` / torchview to render real edges. Requires the user's Python env + the model
  definition; strictly optional.
- SVG/graph-layout rendering of the module tree (dagre/elk).
- Diff two checkpoints (shape/param deltas).
- Per-tensor stats (min/max/mean) via a bounded ranged read of the weight bytes.

## Install / develop

```bash
git clone <this repo> && cd safetensors-viewer
# no build step — plain JS. Press F5 in VSCode to launch an Extension Development Host,
# then open any .safetensors file.
# to package a .vsix:
npx @vscode/vsce package
```

## License

MIT — see [LICENSE](LICENSE).

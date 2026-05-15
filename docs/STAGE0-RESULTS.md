# Cellpose.js — Stage 0 Spike Results

Companion to `PLAN.md`. Verdict on the two hard-dependency gates that
gate Phase 1 implementation.

**Date run:** 2026-05-14
**Verdict:** Both gates **PASS**. Phase 1 implementation is unblocked.

---

## Spike A — ONNX export parity (PASS)

Goal: confirm `cellpose.vit_sam.Transformer` exports to ONNX cleanly and matches
PyTorch within FP32 noise (gate: max abs error < 1e-3).

**Setup**

- Python 3.11.9 venv
- torch 2.12.0, cellpose 4.1.1, onnx 1.21.0, onnxruntime 1.26.0
- CPSAM weights: `mouseland/cellpose-sam` (1.23 GB), 304.6M params
- Tile: `(1, 3, 256, 256)` FP32, 10 deterministic random tiles (seed 0)

**Export notes**

- torch 2.12 uses the dynamo-based exporter by default. `dynamo=True` succeeded
  on the second attempt (`strict=False` failed, `strict=True` succeeded).
- Required extra dep: `onnxscript`.
- Opset 17 requested, auto-upgraded to **opset 18** (version converter couldn't
  downgrade some axis-attribute ops). Acceptable.
- Weights externalized to `cpsam_fp32.onnx.data` (1.1 GB) — graph file is 2.8 MB.
  Normal for >2 GB protobuf payloads.
- `dynamic_axes` H/W on output got hardcoded to 256 by the new exporter. Not
  blocking — Phase 1 always tiles at 256.

**Parity results**

| tile | max abs err | pt (ms) | ort (ms) |
| ---: | ----------- | ------- | -------- |
|    0 | 1.24e-05    | 1063    | 2122     |
|    1 | 8.58e-06    | 1087    | 2011     |
|    2 | 8.58e-06    | 1031    | 2424     |
|    3 | 7.63e-06    | 1698    | 2632     |
|    4 | 8.58e-06    | 2039    | 2078     |
|    5 | 7.63e-06    | 991     | 1989     |
|    6 | 1.00e-05    | 975     | 1935     |
|    7 | 9.06e-06    | 987     | 2139     |
|    8 | 8.11e-06    | 966     | 1928     |
|    9 | 9.06e-06    | 948     | 1922     |

- **Worst max abs err:** 1.24e-05 (gate: 1e-3 — passes by ~80×)
- **Mean max abs err:** 8.96e-06
- PyTorch CPU baseline: ~1.18 s/tile
- ONNX Runtime CPU EP: ~2.12 s/tile (slower than PyTorch on CPU — irrelevant; CPU
  is not the deployment target)

**FP16 variant**

- Post-export FP16 conversion via `onnxconverter-common` and
  `onnxruntime.transformers.float16` both produced broken graphs (dangling type
  mismatches and duplicate node names respectively) — known issues with the
  dynamo-exported graph topology.
- **Workaround that worked:** instantiate the Transformer directly with
  `dtype=torch.float16` and re-export from scratch.
- After merging externalized weights back into the graph (under 2 GB, fits in
  single protobuf), final file is a self-contained **588 MB** `cpsam_fp16.onnx`.
- Graph IO is **FP16** (input `image: tensor(float16)`, output
  `flows_cellprob: tensor(float16)`). Browser harness must use native
  `Float16Array` (Chrome 135+ / Safari 17.4+) or convert manually.

---

## Spike B — WebGPU tile latency (PASS)

Goal: confirm the FP16 ONNX runs on `onnxruntime-web`'s WebGPU EP at acceptable
per-tile latency (gate: median < 2 s/tile; hard fail >= 5 s/tile).

**Setup**

- Hardware: Apple M1 Max, macOS, Chrome (Metal-3 WebGPU adapter)
- `onnxruntime-web@1.20.1` UMD build (`ort.webgpu.min.js`), WASM fallback assets
  from jsDelivr
- Model: `cpsam_fp16.onnx` (588 MB, self-contained), served from
  `python3 -m http.server`
- 3 warmup + 10 timed forward passes on `(1, 3, 256, 256)` FP16 input

**Results**

```
WebGPU adapter: vendor=apple arch=metal-3
model fetched: 588 MB in 695 ms (loopback)
session ready in 1.31 s
warmup 0: 2.29 s  (cold)
warmup 1: 627 ms
warmup 2: 632 ms
iter 0–9: 621–643 ms (tight)

min:    621 ms
median: 628 ms
p95:    643 ms

GATE: PASS (median 628 ms < 2 s target)
```

**Headline numbers**

- **Steady-state: ~628 ms/tile** — 3.2× under the 2 s target.
- **Cold start:** ~2.3 s (one-time shader/JIT compile on first run).
- **Session create:** 1.31 s.
- **Distribution:** tight (~3% spread min → p95) — no instability or thermal
  throttling over a short run.

**Friction encountered (resolved)**

1. `jsdelivr` `+esm` wrapper 404'd on `onnxruntime-web` — switched to the UMD
   build `dist/ort.webgpu.min.js`.
2. `onnxconverter-common` and ORT's `transformers.float16` both produced broken
   FP16 graphs from the dynamo export — fixed by re-exporting directly with
   `Transformer(dtype=torch.float16)`.
3. Manual Uint16 half-float bit-pattern construction was rejected by ORT-web
   1.20 — fixed by using native `Float16Array` (Chrome 135+).

---

## Implications for the Phase 1 plan

- **Model size assumption confirmed:** FP16 single-file ONNX is **588 MB**
  (plan said 620 MB — close enough). INT8 path remains unnecessary for v1.
- **WebGPU is mandatory.** Steady-state 628 ms with the GPU; CPU EP was ~2.1 s
  for FP32 (and WASM EP on the M1 will be in the same ballpark or worse). The
  "WebGPU required" decision holds.
- **Cold start is a real UX concern.** First-run penalty is ~2.3 s of shader
  compilation **plus** the 588 MB cold-cache fetch from CDN. Mitigations to plan
  for in Phase 1:
  - IndexedDB cache for the model bytes (hit on second visit).
  - Eager session creation while the user is browsing the param panel, before
    they hit Run.
  - Optional "preload model" affordance in the engine registry.
- **Output IO is FP16.** The `cellpose-js` package must accept `Float16Array`
  outputs and convert to FP32 for downstream flow-dynamics postprocessing (or
  upgrade dynamics to work in FP16 directly — small wins, not worth the risk for
  v1).
- **Browser version floor:** Chrome 135+ / Safari 17.4+ for native
  `Float16Array`. If we need to support older browsers, switch to FP32 IO with
  `keep_io_types=True` _but_ that needs a working FP16 conversion path — defer.

## Scratch workspace

All spike artifacts live outside the jit-ui repo at:

```
~/cellpose-js-spike/
├── .venv/                       # Python 3.11 venv
├── export_cpsam.py              # FP32 export
├── export_cpsam_fp16.py         # native FP16 export (the one we use)
├── parity_check.py              # PyTorch vs ONNX FP32 parity
├── fp16_convert.py              # post-hoc FP16 conversion (broken, kept for record)
├── cpsam_fp32.onnx              # FP32 graph (2.8 MB)
├── cpsam_fp32.onnx.data         # FP32 weights (1.1 GB)
├── cpsam_fp16.onnx              # FP16 single-file (588 MB) — the deployment artifact
└── webgpu_bench/
    ├── index.html               # browser harness
    └── bench.js                 # benchmark logic
```

Keep this directory around as the reference rig for the next milestone (it's
also the source of truth for the model artifact we'll upload to HF Hub when the
`cellpose-js` repo is ready to publish).

# Cellpose.js — Milestone 1 Results

Companion to `PLAN.md` and `STAGE0-RESULTS.md`.
Verdict on the Milestone 1 exit criterion: *loads the model, runs an identity
forward pass on a tile*.

**Date run:** 2026-05-14
**Verdict:** **PASS**. Phase 1 work continues to Milestone 2 (preprocessing).
**Repo:** `~/git/Cellpose.js/` — initial commit `a0c1955`.

---

## Exit criterion check

| Criterion | Status | Evidence |
|---|---|---|
| `Cellpose.fromPretrained(url)` loads the FP16 ONNX | ✅ | `ready in 6.88 s` on first run; 2.20 s on second run (cache hit) |
| Identity forward pass on a (1, 3, 256, 256) tile | ✅ | Output shape `(1, 3, 256, 256)`, 196608 FP32 floats, finite, range ≈ [-3.24, 2.22] |
| WebGPU EP active, no WASM fallback | ✅ | `adapter: vendor=apple arch=metal-3` |
| IndexedDB cache works on reload | ✅ | `ready in` dropped 6.88 s → 2.20 s on second visit |
| Abort signal cancels in-flight work | ⏳ designed in, not exercised | Will be tested for real in M3 |

## Headline numbers

```
=== first run (cold cache, cold session) ===
ready in 6.88 s
  ├ ~5 s        — fetch 588 MB through Vite proxy
  └ ~1.9 s      — ort.InferenceSession.create + WebGPU shader compile

warmup 0: 907 ms   (one-time JIT/shader cost)
warmup 1: 287 ms
warmup 2: 279 ms

iter 0-4: 277, 280, 277, 278, 277 ms
inference median: 277 ms

=== second run (warm IDB cache, cold session) ===
ready in 2.20 s
  └ ~1.9 s      — session create only; bytes from IndexedDB

warmup 0: 477 ms   (some WebGPU pipeline state retained across reload)
iter 0-4: 280, 277, 277, 277, 286 ms
inference median: 277 ms
```

## What changed since Stage 0

| Metric | Stage 0 (Spike B) | Milestone 1 | Δ |
|---|---|---|---|
| ORT-web version | 1.20.1 (UMD) | 1.26.0 (ESM `/webgpu` entry) | newer |
| Inference median | 628 ms | **277 ms** | **2.27× faster** |
| Cold warmup | 2.29 s | 907 ms | 2.5× faster |
| Session create | 1.31 s | ~1.9 s | slightly slower |

The 2.27× inference speedup is the most material finding. We accidentally
landed on 1.26.0 because `^1.20.1` widened during `npm install`, but the
improvement is large enough that we kept it and re-pinned to `~1.26.0`. ORT
1.21–1.26 added a number of WebGPU kernel optimizations (op fusion,
shader-cache reuse) — the gain is real, not measurement noise.

## Friction encountered (and the fix that stuck)

Five real obstacles surfaced during M1. All are now resolved and documented
in code comments so they won't recur:

1. **`@types/node` + WebGPU + Float16Array gaps in TS.** TS 5.6's stock libs
   ship neither `Float16Array` (TC39 Stage-3, in browsers since Chrome 135)
   nor WebGPU types. Fixed by adding `@webgpu/types` to devDeps and a small
   ambient `Float16Array` shim in `src/types.d.ts`.

2. **`onnxruntime-web` default ESM entry is WASM-only.** `import * as ort from
   'onnxruntime-web'` resolves to the WASM-only build; the WebGPU EP lives at
   the subpath `onnxruntime-web/webgpu`. With the wrong import, ORT registered
   no WebGPU kernels and the protobuf parser blew up at session-create with a
   cryptic `e.getValue is not a function`. Documented at the imports in
   `src/session.ts:6` and `src/cellpose.ts:6`. If anyone moves these later,
   they will undo M1's last 90 minutes of debugging.

3. **ORT-WebGPU dynamically imports `.mjs` sidecars at runtime, cross-origin
   blocked.** The `ort.env.wasm.wasmPaths` URL must resolve same-origin for
   the dynamic `import()` to succeed. jsDelivr 1.20.1 didn't actually publish
   the JSEP/asyncify `.mjs` sidecars (they landed in later releases); 1.26.0
   has them. Solved via a Vite dev-server proxy at `/ort/* →
   cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/*` (see
   `examples/demo/vite.config.ts`). The version in that URL must stay in
   lock-step with `package.json`'s `onnxruntime-web` pin — a comment in the
   config file flags this.

4. **Vite refuses to serve `public/` files to module-import requests.** First
   attempt at fixing #3 was to copy the sidecars into `examples/demo/public/`.
   Vite's `?import` middleware blocks that path: public files are for raw
   fetch only, not for ES module resolution. The proxy approach (#3) sidesteps
   this entirely.

5. **The IDB cache was a red herring during debugging of #2.** The first
   manifestation of the `e.getValue` error happened *after* the fetch, which
   pointed at the cache write. Two layers of best-effort wrappers were added
   (`tryCacheRead`, `tryCacheWrite` in `src/model-cache.ts`) so a corrupted
   IDB entry from a previous run cannot silently break future loads — useful
   safety even though the cache wasn't the actual bug.

## What the M1 code shape gives us

```
~/git/Cellpose.js/
├── src/
│   ├── index.ts          # public exports
│   ├── env.ts            # WebGPU + Float16Array detection
│   ├── model-cache.ts    # streaming fetch + IDB cache (best-effort)
│   ├── session.ts        # thin ORT-WebGPU wrapper
│   ├── cellpose.ts       # Cellpose class, fromPretrained, segmentRawTile
│   └── types.d.ts        # ambient Float16Array + WebGPU type imports
├── examples/demo/        # Vite dev harness exercising the whole path
├── package.json          # onnxruntime-web ~1.26.0 pinned
├── tsconfig.json         # strict + exactOptionalPropertyTypes
└── vite.config.ts        # library mode, ESM-only build
```

Built bundle: **6.18 kB** (gzipped 2.52 kB). Excludes `onnxruntime-web` and
`onnxruntime-web/webgpu` — they're peer deps the consumer brings.

## Parked for later milestones

- **Real preprocessing** (M2): the demo's input is `Math.random() * 2 - 1`
  noise; `segmentRawTile()` doesn't normalize, doesn't tile, doesn't pick
  channels. The output values are stable but biologically meaningless.
- **Flow dynamics** (M4): the FP32 output tensor sits in `flows_cellprob` with
  no consumer. The 277 ms is *inference only* — full
  segmentation latency will be inference + dynamics.
- **Public model artifact**: the demo serves `cpsam_fp16.onnx` from a symlink
  to `~/cellpose-js-spike/`. The HF Hub upload (M6) is needed before the
  package is usable outside this machine.
- **WASM sidecars at build time**: the proxy is a dev-only crutch.
  Library consumers will configure `wasmPaths` themselves. Documented
  approach: a `configureOrt({ wasmPaths })` API that consumers call once
  with a same-origin URL of their choosing.
- **Restore streaming reader if regressed**: the user-rejected DIAG patch
  that would have replaced the streaming reader with `arrayBuffer()` never
  applied; streaming is intact and progress callbacks work. Mentioned here
  so if a future regression removes it, this memo flags the trap.

## Sources

- Initial commit: `~/git/Cellpose.js` `a0c1955`
- Live demo: `vite serve examples/demo/` at http://localhost:5173/
- Model artifact (dev): `examples/demo/public/cpsam_fp16.onnx` symlink →
  `~/cellpose-js-spike/cpsam_fp16.onnx` (588 MB FP16 ONNX)

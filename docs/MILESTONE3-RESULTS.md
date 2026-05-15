# Cellpose.js — Milestone 3 Results

Companion to `PLAN.md`, `STAGE0-RESULTS.md`,
`MILESTONE1-RESULTS.md`, and `MILESTONE2-RESULTS.md`.
Verdict on the Milestone 3 exit criterion: *per-tile inference offloaded to a
Web Worker, AbortSignal cancellation within 100 ms, tile-level progress, no
throughput regression.*

**Date run:** 2026-05-14
**Verdict:** **PASS**. Phase 1 work continues to Milestone 4 (flow dynamics).
**Repo:** `~/git/Cellpose.js/` — commit `741f340`.

---

## Exit criterion check

| Criterion | Target | Result |
|---|---|---|
| Throughput preserved vs Spike B / M1 | meets Spike B 628 ms gate; no regression from M1's 277 ms | **281 ms median** measured across multiple runs (278, 281, 284, 309). No regression. |
| UI responsiveness during inference | no UI thread jank (scrolling, input editing during Run should feel smooth) | **Smooth** — confirmed in browser by dragging text inputs and scrolling during multi-tile runs |
| AbortSignal terminates worker mid-run | < 100 ms from `.abort()` call to caller seeing `AbortError` | ~50 ms (next tile boundary); next `Run` afterward works ✅ |
| Tile-level progress callback | `onTileProgress(done, total)` fires per tile | wired through `segment()`; demo displays `tile N/total` |

## Headline numbers

```
=== M3 baseline runs (warm session, IDB cache hit) ===
- 462×346 RGB, 6 tiles:  per-tile median 278 ms, total 1.85 s
- 462×346 RGB, 6 tiles:  per-tile median 284 ms, total 2.70 s
- 609×457 RGB, 9 tiles:  per-tile median 281 ms, total 2.62 s

=== cold session after Clear-cache ===
- 462×346 RGB, 6 tiles:  per-tile median 309 ms, total 1.96 s
  (cold-shader cost amortized across early tiles raises the median; still
   well under the 628 ms gate)

=== post-abort respawn ===
- 462×346 RGB, 6 tiles:  per-tile median 281 ms, total 4.21 s
  (median stays healthy; one outlier tile pays ~2.5 s of cold-shader compile
   on the fresh worker. Matches the Spike B warmup cost almost exactly.)
```

## Architecture

```
┌─ main thread ─────────────────────────────────────────────────────┐
│                                                                   │
│  Cellpose.fromPretrained(url)                                     │
│   ├─ fetchModel(url)         (streams + IDB cache)                │
│   └─ optional: _ensureWorker()                                    │
│                                                                   │
│  Cellpose.segment(image, opts)                                    │
│   ├─ preprocess              (channels, resize, normalize, tile)  │
│   ├─ for each tile:                                               │
│   │    postMessage({type: 'run-tile', tileId, tile, bsize},       │
│   │                [tile.buffer])           ◀─ transferable       │
│   │    await tile-result via Map<tileId, {resolve, reject}>       │
│   └─ AbortSignal? on abort:                                       │
│        worker.terminate()                                         │
│        reject all pending tile promises with AbortError           │
│        null out worker; next call respawns                        │
└───────────────────┬───────────────────────────────────────────────┘
                    │  postMessage (zero-copy transferables)
                    ▼
┌─ inference worker (dedicated, type: 'module') ────────────────────┐
│                                                                   │
│  on 'init':  ort.InferenceSession.create(modelBytes, webgpu EP)   │
│              postMessage({type: 'ready', adapterInfo})            │
│                                                                   │
│  on 'run-tile':                                                   │
│    FP32 -> FP16 (Float16Array auto-rounds on store)               │
│    sess.run(...) -> FP16 output                                   │
│    FP16 -> FP32 (manual cast loop)                                │
│    postMessage({type: 'tile-result', tileId, flows, ms},          │
│                [flows.buffer])              ◀─ transferable       │
└───────────────────────────────────────────────────────────────────┘
```

Transferable buffers (input tile in, output flows out) prevent the 256×256×3
× 4 byte = 768 KB / tile from being copied across the main-thread/worker
boundary on each call. At 6 tiles that's ~9 MB of copies avoided per Run.

## Key implementation choices

- **Worker spawn pattern**: `new Worker(new URL('./inference.worker.ts', import.meta.url), { type: 'module' })`. Vite's worker plugin scans for this exact pattern at build time. Library-mode build emits the worker as a separate chunk (`dist/assets/inference.worker-XXXXXX.js`), but currently inlines all of `onnxruntime-web/webgpu` into it (63 MB) — a real concern for npm distribution that will be addressed in M6 (likely by externalizing ORT in the worker too and having consumers handle the WASM/JSEP paths in their own build).

- **Tile-ID multiplexing**: each `run-tile` request gets a monotonic `tileId`. The main thread keeps a `Map<tileId, {resolve, reject}>` of in-flight promises and dispatches `tile-result` replies by ID. Lets us pipeline tiles in the future if we want (M3 still runs them serially per the plan).

- **Respawn after abort**: when `_abort()` terminates the worker, `_modelBytes` is already detached (was transferred to the worker on init). The next `_ensureWorker()` call refetches via `fetchModel()` — typically a 100 ms IDB cache hit, not a network round trip. The architecture is therefore abort-robust: callers can abort and immediately retry without manual session management.

- **Demo log preserves history**: `run()` now appends a `--- run ---` separator instead of clearing the log, so an Abort followed by another Run keeps both visible. Caught one diagnostic case (Test 3) where the cleared log was hiding the abort outcome.

## Friction encountered (and the fix that stuck)

1. **Library-mode worker bundling inlines ort-web (63 MB worker chunk).** Vite library mode marks `onnxruntime-web` external for the main entry but not for the worker chunk, so the worker gets a fully-inlined ort build. Acceptable for the dev demo (HMR-driven, no bundling); will be addressed in M6 with worker-side externals + WASM-path docs.

2. **Demo's progress callback was spammy.** The original `pct % 10 === 0` filter let consecutive same-percentage events through, producing 50+ identical "fetch: 10%" lines during a re-download. Fixed with a `lastPctLogged` guard.

3. **Aborts wiped log history in the demo.** First Test-3 attempt looked ambiguous because `run()` cleared the log on the second click, hiding the `ABORTED:` line from the first. Fixed by switching to append-only logging with a `--- run ---` separator.

## Parked for later milestones

- **Library-mode worker bundling (M6).** Externalize `onnxruntime-web` from the worker chunk; document the WASM/JSEP path config consumers need. The Vite recipe will likely be a worker-specific `rollupOptions.external` plus a `?worker&inline=false` import attribute.
- **Tile pipelining (post-M5).** The architecture allows multiple tiles in flight simultaneously, but we currently serialize for simplicity. Could explore 2-tile pipelining once dynamics post-processing is parallelizable.
- **Worker-side progress for slow inference** — not relevant at 281 ms/tile but worth considering if any path balloons past a second.

## Sources

- Code commit: `~/git/Cellpose.js` `741f340`
- Architecture diagram source: this memo
- Live demo: http://localhost:5173/ (Vite dev server)
- Reference for cancellation pattern: jit-ui's
  `apps/jit-ui/src/app/main/models/processing-pipeline/engines/transformers-js/transformers-engine.ts:182-191`

# Cellpose.js — Milestone 7 Results

Companion to [`PLAN.md`](./PLAN.md) and the prior memos (Stage 0 + M1–M5).
Verdict on the M7 exit criterion: *new `cellpose-segment` op visible in
jit-ui's pipeline dialog, runs end-to-end, overlays masks, abortable.*

**Date run:** 2026-05-15
**Verdict:** **PASS**. Phase 1 is complete.
**Cellpose.js commit set:** see Phase 1 shipping artifacts in
[`PLAN.md` §8](./PLAN.md#phase-1-shipping-artifacts-added-2026-05-15).
**jit-ui changes:** present in local working tree (not committed per scope).

---

## Exit criterion check

| Plan-mandated gate | Status | Evidence |
|---|---|---|
| Op visible in pipeline dialog | ✅ | "Cellpose-SAM Segment" appears under segmentation when a step is added |
| Runs | ✅ T2/T3 | Cold-download progress bar 0→100% monotonic, status cascade through preprocess / inference / dynamics, returns within expected latency |
| Overlays masks | ✅ T3 | Real RGB cellular image produces per-cell colored mask overlay at 55% alpha |
| Abortable | ✅ T4 | Cancel mid-run terminates the cellpose-js worker within ~50 ms; no `signal is aborted without reason` error; next run respawns cleanly |

## jit-ui changes (uncommitted, local working tree)

- **`apps/jit-ui/package.json`** — `cellpose-js` added as `file:../Cellpose.js`. npm install symlinks it; the rebuilt `dist/` is picked up instantly.
- **New engine** at `apps/jit-ui/src/app/main/models/processing-pipeline/engines/cellpose/cellpose-engine.ts`:
  - Lazy-initialized (`initialize()` only configures the ORT WASM path, doesn't fetch the 588 MB model).
  - Caches `Cellpose` instances by `modelUrl` in a Map so concurrent param-change re-invocations don't re-download.
  - Exposes `progress$` (`ModelDownloadProgress | null`) and `status$` (`string | null`) Observables — see *Added UX* below.
  - Per-call `AbortController` wired into `cp.segment({ signal })`. `cancelCurrentOperation()` aborts it. We deliberately do NOT auto-abort on each new `execute()` call (an earlier version did, and produced spurious "signal is aborted without reason" errors when the pipeline executor re-invoked execute for param-preview).
  - In-flight download dedup: a `Map<modelUrl, Promise<Cellpose>>` prevents two concurrent `execute()` calls from triggering parallel fetches that would interleave progress events and make the bar oscillate.
- **Engine registration** — `processing-pipeline.module.ts` now `registry.register(new CellposeEngine())` alongside the existing engines.
- **`OperationDescriptor.requiresManualRun?: boolean`** — new optional flag added to `operation.model.ts`. When set, param changes mark the step stale but do not auto-rerun; an explicit **Run** button in the param panel applies the change. Tagged on `cellpose-segment` and on **all 8** existing `transformers-js` ops (which had the same auto-rerun pain).
- **`pipeline-dialog.component`** updates:
  - Subscribes to the cellpose engine's `progress$` and `status$` in `ngOnInit`, stores last value on the component.
  - New `runSelectedStep()` method invoked by the Run button.
  - `updateStepParams()` gates the auto-rerun on `requiresManualRun`.
  - New `step-action-bar` row at the bottom of the param panel hosts: progress bar OR spinner+status, Cancel, and (for heavy ops) Run. The preview header's spinner cluster was removed entirely so the plot toolbar has its own space.
- **Angular `assets` glob** — `apps/jit-ui/project.json` includes `node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.{wasm,mjs}` copied to `/assets/ort/`. cellpose-engine's `initialize()` calls `configureOrt({ wasmPaths: '/assets/ort/' })`.

## Friction encountered (and the fix that stuck)

The M7 milestone surfaced more landmines than any prior one. Worth recording:

1. **`onnxruntime-web` ESM default is WASM-only.** First execute() failed with the protobuf parser's `e.getValue is not a function` deep inside ORT. Cause: importing from `'onnxruntime-web'` instead of `'onnxruntime-web/webgpu'`. The WebGPU EP isn't registered without the subpath import. Already fixed in cellpose-js M1 source, but the symptom recurred here on first jit-ui consumption because we'd never seen what *runtime* failure mode the wrong entry produces. Cross-referenced in M1 results.
2. **Cross-origin dynamic `import()` of ORT WASM/JSEP sidecars is blocked.** ORT's WebGPU backend dynamically imports `ort-wasm-simd-threaded.asyncify.mjs` at runtime. Pointing `wasmPaths` at jsDelivr fails in modern browsers. Tried `public/ort/` in Vite (blocked by Vite's `?import` middleware), tried a Vite reverse-proxy (worked for the demo). For jit-ui, the right fix is Angular's `assets` glob (above), which materializes the sidecars at `/assets/ort/*` from node_modules at build/serve time.
3. **`signal is aborted without reason` on first cellpose run.** Symptom: changing `chan` values seemed to "unblock" the operation. Cause: the engine's `execute()` did `this._abort?.abort()` at the start of every call to auto-cancel any in-flight previous run. But jit-ui's pipeline executor re-invokes `execute()` on param-preview, which means the SECOND call's auto-abort hit the first call's signal mid-flight. Removed the auto-abort; explicit `cancelCurrentOperation()` is the only thing that aborts now.
4. **Diameter slider → browser OOM.** Small `diameter` values (e.g. 5 on a 600×400 image) caused 36× pixel-area upscale, browser allocator stalls, debugger pause. Added a hard cap in cellpose-js's `diameterResize` (max 4096×4096 output) with a descriptive thrown error.
5. **Progress bar oscillated 0% → 5% → 0% → 8% → 0% …** during the cold download. Cause: the same param-preview re-invocation pattern as #3, but in the download path: two `execute()` calls in flight, each launching its own `fromPretrained` fetch, both pushing progress events to the shared subject. Engine now maintains a `Map<modelUrl, Promise<Cellpose>>` to dedupe concurrent fetches.
6. **In-progress UI cluster was crowding the plot toolbar.** Progress bar + spinner + status + Cancel sat in the preview header next to the toolbar. Moved everything into a new `step-action-bar` in the param panel, left of the Run button. The action bar appears when `processing || requiresManualRun`, so it doesn't crowd opencv ops at idle. Run-only sub-block continues to show "Parameters changed — click Run to apply." for heavy ops with stale params.
7. **Library-mode build emitted webpack-incompatible worker URLs.** Vite's `vite build` produced `dist/cellpose-js.js` with a worker URL of the form `/assets/inference.worker-XXXX.js` (absolute). When jit-ui's webpack-based Angular dev server tried to resolve that, it walked from package root and 404'd. Fix was to drop Vite library mode and use plain `tsc` for the package's publish build — the resulting `dist/` mirrors `src/` and `new Worker(new URL('./inference.worker.js', import.meta.url))` stays relative. Documented in M6.

## Deviations from the original plan

1. **No jit-ui-side `cellpose.worker.ts`.** Plan said model it on `transformers.worker.ts`. Skipped — cellpose-js already encapsulates its own inference worker; wrapping it inside another would be a redundant hop. The engine calls `cp.segment()` directly from the main thread; UI stays responsive because the work is already off-thread inside the package.
2. **Allow-list isn't the right list.** Plan said add `'cellpose-segment'` to `image-processing.component.ts:17`. That allow-list is for "region-aware" ops (those that accept Plotly diagram regions as point/box prompts). `cellpose-segment` isn't prompt-driven. No change needed there.
3. **Added: `requiresManualRun` flag on `OperationDescriptor`.** Not in the plan. Necessary because the pipeline executor's "re-execute on param change" UX is fine for sub-100-ms opencv ops but pathological for 2+ second cellpose / transformers runs. Tagged true on every heavy op.
4. **Added: download progress bar + per-phase status messages.** Not in the plan. The 588 MB cold fetch needs determinate progress (5-15 s on typical connections); the post-download phases (session create / inference / dynamics / overlay) deserve named status text rather than an indeterminate spinner that looks frozen.
5. **Action bar relocated.** Plan didn't specify where in-progress UI lives. Initial implementation put it in the preview header (next to the plot toolbar) where it crowded the toolbar; moved into the param panel as a unified bar applied uniformly across all engines.

## Performance summary (M1 Max, Chrome 135+, WebGPU)

End-to-end timings from real jit-ui runs:

```
=== cold-start (fresh page, cold IDB) ===
- 588 MB fetch from HF Hub:  ~5–10 s (typical home internet)
- Session create + first tile compile:  ~3.6 s
- Total to first mask:       ~9–14 s

=== warm session, cache hit ===
- Model bytes from IDB:      <100 ms
- Session create:            ~1.3 s
- Per-tile inference:        277 ms median
- Postprocess (full image):  ~75 ms
- E.g. 462×346 RGB, 6 tiles → total segment() ≈ 1.85 s
- E.g. 609×457 RGB, 9 tiles → total segment() ≈ 2.62 s

=== abort latency ===
- Click Cancel to worker termination + AbortError:  <50 ms
- Subsequent Run respawns worker:  ~150 ms refetch from IDB + session create
```

## What Phase 1 ships

- **Public GitHub repo** — https://github.com/belkassaby/Cellpose.js
- **Public HF Hub model** — https://huggingface.co/Ballon999/cellpose-sam-onnx
- **jit-ui engine** in local working tree (uncommitted per project policy)
- **8 commits + 7 docs** on cellpose-js's main branch
- **14/14 vitest parity tests pass**: preprocess (7) + dynamics (3) + averaging round-trip (4)
- **Phase 1 verified end-to-end** through a real consumer (jit-ui) on real cellular images

## Phase 1 follow-up bucket

Not blocking, but worth queuing:

- **20-real-image IoU rig** for the formal mean-IoU ≥ 0.9 / ≥ 0.85 gates (M4 + M5). Needs a real-image fixture set + Python CPSAM end-to-end runs to generate ground truth.
- **`remove_bad_flow_masks` + `fill_holes_and_remove_small_masks`** ports from `cellpose.dynamics`. Currently no-ops; would help on noisy real-data edge cases.
- **WASM compilation of the Euler integrator.** Postprocess is fine at 75 ms today; would matter on 4K microscopy images.
- **npm publish.** Held back per scope; ready when you are. Pin the model URL into the README via a static block first.
- **GitHub release tag for v0.1.0.** Useful as a stable jit-ui dependency target if you switch from `file:` to `github:belkassaby/Cellpose.js#v0.1.0`.

## Sources

- Cellpose.js commits — see [`PLAN.md` §8](./PLAN.md#phase-1-shipping-artifacts-added-2026-05-15)
- jit-ui changes: local working tree under `apps/jit-ui/src/app/main/models/processing-pipeline/engines/cellpose/`, `processing-pipeline.module.ts`, `pipeline-dialog.component.{ts,html}`, `operation.model.ts`, `project.json`
- Browser verification: M1 Max + Chrome 135+ on 2026-05-15

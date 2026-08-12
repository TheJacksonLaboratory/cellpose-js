# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-08-12

Re-synced against upstream [`MouseLand/cellpose`](https://github.com/MouseLand/cellpose)
`main` @ `a54cb48` (2026-06-14). The previous parity baseline was the 2026-05-22
review. Output of `segment()` changes for callers on the default channel path;
snapshot or regression-pinned consumers should review.

### Changed

- **BREAKING (behavior): channels now pass through by default.** With neither
  `chan` nor `chan2` set, the first up-to-3 source channels are copied straight
  to the network and normalized independently — RGBA in, `[R, G, B]` out (alpha
  dropped). Previously the default was `chan = 0`, which collapsed the source to
  a grayscale **mean** in channel 0 and zeroed channels 1–2.

  This matches upstream v4, where `transforms.convert_image` performs no channel
  selection at all (it truncates with `x[..., :3]` and lets `normalize_img`
  rescale each channel on its own percentiles), and the `channels=` argument now
  logs _"Cellpose4 takes inputs with arbitrary channel orders"_. For RGB
  fluorescence composites this keeps every marker at full dynamic range instead
  of blending them.

  **`chan` / `chan2` are unchanged and still supported** — setting either one
  restores the Cellpose 1–3 selection mapping exactly as before. Callers who
  want the old default back can pass `{ chan: 0 }` explicitly.

- **`niter` is no longer scaled when dynamics run at network resolution.**
  `segment()` runs the dynamical system on the resized image, but was scaling
  the Euler iteration count as though it ran at source resolution
  (`200 / scale`). At `diameter = 15` (scale 2) that gave 100 iterations where
  200 were needed — under-integration, which fragments cells; above 30 it merely
  wasted time. Upstream gates the same scaling explicitly
  (`niter_scale = 1 if rescale is None or not resample else rescale`,
  `models.py:_run_cp`). Only affects calls that pass `diameter`.

### Added

- **`resample` option** (default `false`). When `diameter` triggers a resize,
  `resample: true` bilinear-upsamples the predicted flow field and cellprob back
  to source resolution before running dynamics there, with `200 / scale`
  iterations — upstream's default (`models.py:_run_net`). Mask boundaries then
  follow the flow field instead of a nearest-neighbor upscale of a smaller label
  map, at the cost of running dynamics over the full-resolution image. The
  default keeps the existing (cheaper) behavior.
- **`resizeChw`** exported from `preprocess/` — the bilinear CHW resize backing
  both `diameterResize` and the `resample` path.
- **Clear error for `tile !== 256`.** CPSAM's position embeddings are fixed at
  256/8 = 32×32 tokens and the ONNX export hardcodes H/W to 256, so other tile
  sizes previously failed deep inside ORT with an opaque shape error. Upstream
  raises the same restriction (_"bsize != 256 is not supported for cpsam"_).

### Documentation

- **Corrected the pipeline diagram in the README.** It showed
  `diameterResize → normalizePerChannel`; the code has run normalize _before_
  resize since 0.2.0 (matching `models.py:_run_cp`).
- **New "Upstream parity" section** recording the parity commit, the model-zoo
  situation, and known divergences. Notably: upstream's default model is now
  `cpsam_v2`, which is architecturally identical to `cpsam` (`get_backbone()`
  returns `"sam_vitl"` for it, instantiating the same `CPSAM` class), so the
  existing FP16 ONNX export recipe applies unchanged and `fromPretrained()`
  needs no API change to load it. `cpdino` / `cpdino-vitb` are DINOv3-based with
  `bsize` 384 and are not supported.

## [0.4.1] — 2026-07-22

### Fixed

- **`buildCpsamChannels` now validates the channel count.** A non-positive or
  non-integer `channels` argument (e.g. `channels = 0`) previously slipped past
  the length check and made the `chan = 0` grayscale mean divide by zero,
  silently producing `NaN`s. It now throws a clear error.

## [0.4.0] — 2026-07-22

### Fixed

- **Grayscale (`chan = 0`) now averages the color channels** instead of taking
  only source channel 0 (red). The old behavior blanked any image whose signal
  wasn't in the red channel: a green- or blue-dominant fluorescence image
  normalized to all-zeros and produced **empty masks**. `chan = 0` now computes
  the mean across color channels — with alpha excluded for RGBA input
  (`channels === 4`) so a fully-opaque image isn't dragged toward 255 — matching
  `cellpose.transforms` (`data.mean(axis=-1)` for `channels=[0, 0]`).
- **Demo "cellprob" panel rendered all-black.** It heatmapped
  `tiles[0].flows_cellprob`, which has been an empty `Float32Array` since 0.3.0
  moved the pipeline into the worker. Replaced with a "masks over input" overlay
  built from the returned label map, and removed the dead heatmap code.

### Changed

- **`chan` / `chan2` widened from `0 | 1 | 2 | 3` to `number`.** Values ≥ 1
  select source channel `chan − 1` (0-based), so true multichannel microscopy
  stacks (> 3 channels) can pick a specific marker without averaging — e.g. a
  5-channel image can use `chan = 4` to select channel 3. Backward compatible:
  existing `0`–`3` values behave identically (except `chan = 0`, per the fix
  above). Non-integer or out-of-range indices now throw.

### Removed

- **`SegmentMilestone1Output`** (deprecated in earlier releases). The type was a
  leftover from the step-by-step build and is no longer returned by any API —
  `segment()` returns `SegmentOutput`. Removed from the public exports.

## [0.3.0] — 2026-06-14

### Changed

- **Full pipeline runs in the Web Worker.** `segment()` now offloads the
  entire pipeline — preprocessing, per-tile inference, tile averaging, flow
  dynamics, and inverse-resize — to the inference worker via a single
  `segment` message, and transfers back only the final label map. Previously
  only per-tile inference ran off-thread, so preprocessing and dynamics ran on
  the main thread and blocked the UI. The public `segment(input, opts)` API and
  `onTileProgress` are unchanged.

### Removed

- `SegmentOutput.tiles[].flows_cellprob` is now an empty `Float32Array` — the
  per-tile flow tensors stay in the worker (returning them would defeat the
  offload). The per-tile timing diagnostics (`tx`/`ty`/`bsize`/`inferenceMs`)
  are unchanged.

## [0.2.0] — 2026-05-22

Python-parity fixes plus quality-of-life additions surfaced by a code review
against `cellpose/{transforms,dynamics,models,core}.py`. Output of `segment()`
may differ from `0.1.1` — values are closer to the upstream reference. Snapshot
or regression-pinned consumers should review.

### Fixed

- **Preprocess order.** Normalize now runs **before** resize, matching
  `models.py:_run_net`. Resizing first would change the per-channel percentile
  statistics that drive normalization.
- **Float32 bilinear resize.** Replaced the canvas-based `diameterResize`,
  which quantized each channel to uint8 (~1/255 per-pixel error) and required
  `OffscreenCanvas`/`HTMLCanvas`, with a pure-JS implementation using OpenCV's
  `INTER_LINEAR` pixel-center mapping + edge replication. Removes the canvas
  dependency entirely and is now testable in Node.
- **`niter` scaling.** Now computed as `floor(200 / scale)` when `diameter` is
  set, matching Python's `int(200 / image_scaling)`. Upscaled images get fewer
  iterations, downscaled get more. Explicit user-supplied `niter` still wins.
- **`follow_flows` rounding.** Switched the final coordinate cast from
  `Math.round` to `Math.trunc` to match PyTorch's `.int()`, which truncates
  toward zero (it's a C-style cast). The 1-pixel shift previously split two
  small overlap-zone labels off their Python counterparts in the
  `three_cells_192` fixture.
- **README quickstart example.** `cellprob_threshold: 0` at the top level was
  silently ignored — the actual API nests it under `dynamics:
{ cellprobThreshold }`. Also removed a stale "once published in M6
  follow-up" parenthetical referencing the model URL.
- **Worker init errors no longer dropped.** `worker.onerror`,
  `worker.messageerror`, and worker-side `error` messages received before
  the first tile now reject the `_workerReady` promise; previously they were
  only routed to pending-tile promises (empty during init) and disappeared
  silently.

### Added

- **`FromPretrainedOptions.signal` now cancels the preload phase.** Previously
  the abort signal was only honored during fetch and `segment()`; if the user
  aborted while ORT was creating its session, nothing happened. Now an abort
  during `_ensureWorker()` terminates the worker and rejects with
  `AbortError`.
- **`FromPretrainedOptions.onStatus`.** New optional callback that receives
  pre-ready phase strings from the inference worker (`'worker module loaded'`,
  `'init message received'`, `'configuring ORT (loading WASM sidecars)'`,
  `'creating ORT session (parsing 588 MB ONNX graph)'`, `'session created in
<ms> ms; describing adapter'`). Useful for showing a determinate status
  during the slow first-run worker init.
- **Test coverage**: 14 → 61 tests across 11 files. New direct-coverage suites
  for `buildCpsamChannels`, `percentile`, `taperMask`, `followFlows`,
  `getMasks`, the new bilinear `diameterResize`, plus an end-to-end
  postprocess pipeline test that mocks the inference step.

### Changed

- **Dynamics parity gate.** The `three_cells_192` fixture's mean-IoU gate
  tightened from 0.55 → 0.99 (was a known divergence, now matches Python at
  1.000 with all 5 per-cell IoUs at 1.000).
- **Demo image picker** surfaces errors visibly instead of silently failing.
  `createImageBitmap`'s catch path now nudges users toward PNG/JPEG (TIFF
  microscopy images aren't natively decoded by any browser).
- **Demo Vite dev mode** now correctly loads the inference worker. Vite's
  worker plugin transforms `new URL('./inference.worker.js', ...)` to a URL
  ending in `.js?worker_file`, but the source file on disk is `.ts` and the
  worker plugin doesn't auto-swap extensions. Demo's `vite.config.ts` now
  ships a small middleware that rewrites `.js?worker_file` →
  `.ts?worker_file` when the `.ts` source exists. Dev-only; production builds
  unaffected.

### Internal

- Added `'status'` message type to the worker → main thread protocol.
- Removed canvas dependency from `src/preprocess/resize.ts`.

## [0.1.1] — 2026-05-15

Patch release.

### Fixed

- README and docs: corrected Hugging Face Hub model URL to
  `ballon999/cellpose-sam-onnx` and normalized URL casing to lowercase across
  `docs/PLAN.md`, `docs/MILESTONE7-RESULTS.md`, and `examples/demo/index.html`.

## [0.1.0] — 2026-05-15

Initial public release. End-to-end browser segmentation pipeline:
preprocessing, IndexedDB-cached model load, WebGPU inference in a Web Worker
with `AbortSignal` cancellation, tile averaging, flow dynamics, full-image
label maps. See [`docs/PLAN.md`](./docs/PLAN.md) for the full milestone trail
and [`docs/STAGE0-RESULTS.md`](./docs/STAGE0-RESULTS.md) for parity / latency
measurements.

[0.5.0]: https://github.com/belkassaby/Cellpose.js/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/belkassaby/Cellpose.js/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/belkassaby/Cellpose.js/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/belkassaby/Cellpose.js/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/belkassaby/Cellpose.js/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/belkassaby/Cellpose.js/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/belkassaby/Cellpose.js/releases/tag/v0.1.0

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/belkassaby/Cellpose.js/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/belkassaby/Cellpose.js/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/belkassaby/Cellpose.js/releases/tag/v0.1.0

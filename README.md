# cellpose-js

[![CI / CD](https://github.com/belkassaby/Cellpose.js/actions/workflows/ci-cd.yaml/badge.svg)](https://github.com/belkassaby/Cellpose.js/actions/workflows/ci-cd.yaml)
[![npm version](https://img.shields.io/npm/v/cellpose-js.svg)](https://www.npmjs.com/package/cellpose-js)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178c6.svg)](https://www.typescriptlang.org/)
[![WebGPU](https://img.shields.io/badge/runtime-WebGPU-005A9C.svg)](https://caniuse.com/webgpu)

Browser-side cellular segmentation powered by [Cellpose-SAM](https://github.com/MouseLand/cellpose), running on WebGPU. Faithful TypeScript port of the Cellpose-SAM inference + dynamics pipeline, designed for in-browser microscopy workflows without a server round-trip.

> **Status:** The full pipeline — preprocessing, WebGPU inference, tile averaging, and flow dynamics — runs in a Web Worker, so `segment()` never blocks the UI thread (earlier releases ran only per-tile inference off-thread). Model loading uses an IndexedDB cache. SlimSAM-style compression and domain-specialized finetunes are out of scope — see the [implementation plan](./docs/PLAN.md) §6.

## Highlights

- **Single-call API**: `await Cellpose.fromPretrained(modelUrl)` → `await cp.segment(image, opts)` → a `Uint32Array` instance label map at source resolution.
- **WebGPU inference** via `onnxruntime-web/webgpu`. Measured **~277 ms / 256×256 tile on an M1 Max**. Cold start ~2.3 s (one-time shader compile).
- **Fully off the UI thread**: the entire pipeline (preprocess → inference → tile averaging → flow dynamics → resize) runs in a Web Worker via a single `segment` message; only the final label map is transferred back, so the UI never blocks. `AbortSignal` terminates the worker mid-run with sub-100 ms latency.
- **Faithful Python parity** for preprocess and dynamics — vitest parity tests pass against numpy-generated `.npy` fixtures.
- **IndexedDB cache** for the 588 MB FP16 model: first visit fetches from your CDN; subsequent visits load from local storage in <2 s.

## Browser requirements

- **Chrome ≥135 (Feb 2025)** or **Safari ≥17.4**. Native `Float16Array` is required to consume the FP16 ONNX graph IO.
- WebGPU available (`'gpu' in navigator`).
- `onnxruntime-web ~1.26.0` as a peer dependency.

Older browsers fail fast with a clear `UnsupportedEnvironmentError`.

## Install

```sh
npm install cellpose-js onnxruntime-web
```

You also need to host:

1. **The model**: `cpsam_fp16.onnx` (588 MB). Either upload to your own CDN, or use the public copy at `https://huggingface.co/ballon999/cellpose-sam-onnx/resolve/main/cpsam_fp16.onnx`.
2. **ORT-web's WASM/JSEP sidecars**: ORT dynamically imports `.mjs` and `.wasm` files at runtime. They must be served **same-origin** with your app (cross-origin dynamic `import()` is blocked). Either copy `node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.{wasm,mjs,jsep.wasm,jsep.mjs,asyncify.wasm,asyncify.mjs}` to your public assets, or proxy `/ort/*` to jsDelivr at build time (see `examples/demo/vite.config.ts` for the recipe).

## Quickstart

```ts
import { Cellpose, configureOrt } from 'cellpose-js';

// One-time: tell ORT where to find its WASM sidecars.
configureOrt({ wasmPaths: '/ort/' });

// Load the model. Cached in IndexedDB after the first visit.
const cp = await Cellpose.fromPretrained(
  'https://your-cdn/cpsam_fp16.onnx',
  { preload: true }, // eager session create
);

// Segment an image.
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

const result = await cp.segment(
  { data: imageData.data, width: imageData.width, height: imageData.height, channels: 4 },
  {
    diameter: 30, // estimated cell diameter in source pixels (omit for native resolution)
    chan: 0, // primary channel (0 = grayscale = mean of color channels)
    chan2: 0, // secondary channel (0 = none)
    dynamics: { cellprobThreshold: 0 }, // pixels above this enter the dynamical system
    onTileProgress: (done, total) => console.log(`tile ${done}/${total}`),
  },
);

console.log(`Found ${result.count} cells.`);
// result.masks      : Uint32Array — instance label map at source resolution, 0=background
// result.width      : number      — source image width
// result.height     : number      — source image height
// result.totalMs    : number      — wall-clock time for the segment() call
// result.tiles      : per-tile timing diagnostics (tx/ty/bsize/inferenceMs).
//                     Flow tensors stay in the worker, so flows_cellprob is empty.
```

## Parameter quick-reference

### `chan` / `chan2`

CPSAM was trained with channel-shuffling augmentation, so the cyto-vs-nuclei _assignment_ rarely matters — but you still need to point it at the channel(s) that actually carry signal.

- **`chan = 0`** — grayscale: the **mean** of the source's color channels (alpha is excluded for RGBA input, i.e. `channels === 4`). Mirrors `cellpose.transforms` (`data.mean(axis=-1)`). Use for brightfield / H&E / phase, or any RGB image whose signal is really one thing shown in color.
- **`chan = 1 | 2 | 3`** — pick red / green / blue directly, no averaging.
- **`chan = k` (k ≥ 1)** — selects source channel `k − 1` (0-based), for _any_ channel count. A 5-channel microscopy stack can use `chan = 4` to pick channel 3.
- **`chan2`** — same indexing for the secondary (nuclear) channel; `0` = none.

> **Multichannel images:** each channel is usually distinct biology (e.g. DAPI vs. GFP), so **never use `chan = 0`** on them — averaging different markers is meaningless. Select the channel you want with `chan`/`chan2 ≥ 1` instead (which never averages). CPSAM only ever consumes a 3-channel `[primary, secondary, 0]` input, so reducing a stack to "which channel is cytoplasm, which is nucleus" is required regardless.

| Image type                                             | `chan` | `chan2` |
| ------------------------------------------------------ | ------ | ------- |
| H&E histology, brightfield, phase contrast             | `0`    | `0`     |
| Fluorescence: green cyto, blue nuclei                  | `2`    | `3`     |
| Fluorescence: red cyto, green nuclei                   | `1`    | `2`     |
| Multichannel stack: cyto = ch3, nuclei = ch0 (0-based) | `4`    | `1`     |
| First run / unknown (single-signal image)              | `0`    | `0`     |

### `diameter`

Rescales the image so the median cell occupies ~30 px (CPSAM's training median). Omit to run at native resolution.

| Cells in source image   | Suggested            |
| ----------------------- | -------------------- |
| Roughly 20–60 px across | leave blank          |
| Tiny (5–15 px)          | ≈ 10                 |
| Large (80+ px)          | your visual estimate |

## Performance (M1 Max, Chrome 135+, WebGPU)

| Step                            | Time                | Notes                          |
| ------------------------------- | ------------------- | ------------------------------ |
| Model fetch (cold cache)        | ~5 s                | 588 MB from local proxy / CDN  |
| Model fetch (warm IDB)          | <100 ms             | IndexedDB hit                  |
| `ort.InferenceSession.create`   | ~1.3 s              | one-time per session           |
| First inference (cold shader)   | ~2.3 s              | one-time WebGPU shader compile |
| Steady-state per-tile inference | **277 ms**          | 256×256 FP16                   |
| Per-tile preprocessing          | ~14 ms amortized    | normalize + tile copy          |
| Full-image dynamics             | **74 ms** (400×400) | average + Euler + cluster      |
| Abort latency                   | <50 ms              | next tile boundary             |

## Architecture

The whole pipeline runs **inside the Web Worker**: the main thread posts the image with one `segment` message and receives the final `masks` (transferred back), so preprocessing and flow dynamics never block the UI.

```
input image → buildCpsamChannels → diameterResize → normalizePerChannel → makeTiles
                                                                              ⇣ (per tile)
                                                              ort.InferenceSession.run
                                                                              ⇣
                                                                       averageTiles
                                                                              ⇣
                                                                       computeMasks (Euler + cluster + renumber)
                                                                              ⇣
                                                                  (optional) nearest-neighbor unresize
                                                                              ⇣
                                                                        Uint32Array masks
```

See [`src/`](./src/) for module-level documentation.

## Testing

```sh
npm run test         # vitest: parity + unit tests against numpy fixtures
npm run typecheck    # tsc --noEmit
npm run build        # vite library build + tsc --emitDeclarationOnly
npm run demo         # vite serve examples/demo
```

The demo at `examples/demo/` is a complete client that exercises the full pipeline. Point it at a local model file via `examples/demo/public/cpsam_fp16.onnx` (symlink), or change the URL in the Model field.

## Troubleshooting

| Symptom                                                                              | Cause                               | Fix                                                                              |
| ------------------------------------------------------------------------------------ | ----------------------------------- | -------------------------------------------------------------------------------- |
| `e.getValue is not a function` at session-create                                     | Wrong ORT entry point               | Import from `onnxruntime-web/webgpu`, not `onnxruntime-web`.                     |
| `Failed to fetch dynamically imported module: …/ort-wasm-simd-threaded.asyncify.mjs` | Cross-origin dynamic import blocked | Serve ORT WASM files same-origin (or proxy). See `configureOrt({ wasmPaths })`.  |
| `Float16Array is not defined`                                                        | Browser too old                     | Chrome ≥135, Safari ≥17.4. No earlier polyfill is supported.                     |
| `Operation aborted` after AbortSignal fires                                          | Working as intended                 | Worker terminates; next `segment()` call respawns from IDB cache (~150 ms).      |
| Mask overlay has split cells at tile borders                                         | Tile-averaging regression           | Shouldn't happen — tile averaging stitches across borders. Please file an issue. |

## Credits

- Model and algorithm: [Cellpose-SAM (Stringer et al., 2025)](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1). Original implementation: [MouseLand/cellpose](https://github.com/MouseLand/cellpose) — BSD-3.
- Inference runtime: [`onnxruntime-web`](https://github.com/microsoft/onnxruntime).

## Security

See [SECURITY.md](./SECURITY.md) for the threat model, reporting process, and recommended consumer practices.

## License

MIT — see [LICENSE](./LICENSE).

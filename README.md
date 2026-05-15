# cellpose-js

Browser-side cellular segmentation powered by [Cellpose-SAM](https://github.com/MouseLand/cellpose), running on WebGPU. Faithful TypeScript port of the Cellpose-SAM inference + dynamics pipeline, designed for in-browser microscopy workflows without a server round-trip.

> **Status:** v0.1.0 — first end-to-end-working release. Phase 1 from the [implementation plan](./docs/PLAN.md) is complete: model loading + IndexedDB cache, preprocessing, WebGPU inference in a worker, tile averaging, flow dynamics, full-image label maps. Phase 2 (SlimSAM-style compression + domain-specialized slim models) is planned but not started.

## Highlights

- **Single-call API**: `await Cellpose.fromPretrained(modelUrl)` → `await cp.segment(image, opts)` → a `Uint32Array` instance label map at source resolution.
- **WebGPU inference** via `onnxruntime-web/webgpu`. Measured **~277 ms / 256×256 tile on an M1 Max**. Cold start ~2.3 s (one-time shader compile).
- **Web Worker offload**: inference doesn't block the UI thread; AbortSignal terminates the worker mid-run with sub-100 ms latency.
- **Faithful Python parity** for preprocess and dynamics — 14/14 vitest parity tests pass against numpy-generated `.npy` fixtures.
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

1. **The model**: `cpsam_fp16.onnx` (588 MB). Either upload to your own CDN, or use the public copy at `https://huggingface.co/belkassaby/cellpose-sam-onnx/resolve/main/cpsam_fp16.onnx` (once published in M6 follow-up).
2. **ORT-web's WASM/JSEP sidecars**: ORT dynamically imports `.mjs` and `.wasm` files at runtime. They must be served **same-origin** with your app (cross-origin dynamic `import()` is blocked). Either copy `node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.{wasm,mjs,jsep.wasm,jsep.mjs,asyncify.wasm,asyncify.mjs}` to your public assets, or proxy `/ort/*` to jsDelivr at build time (see `examples/demo/vite.config.ts` for the recipe).

## Quickstart

```ts
import { Cellpose, configureOrt } from 'cellpose-js';

// One-time: tell ORT where to find its WASM sidecars.
configureOrt({ wasmPaths: '/ort/' });

// Load the model. Cached in IndexedDB after the first visit.
const cp = await Cellpose.fromPretrained(
  'https://your-cdn/cpsam_fp16.onnx',
  { preload: true },                            // eager session create
);

// Segment an image.
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

const result = await cp.segment(
  { data: imageData.data, width: imageData.width, height: imageData.height, channels: 4 },
  {
    diameter: 30,                                // estimated cell diameter in source pixels (omit for native resolution)
    cellprob_threshold: 0,
    chan:  0,                                    // primary channel (0 = grayscale)
    chan2: 0,                                    // secondary channel (0 = none)
    onTileProgress: (done, total) => console.log(`tile ${done}/${total}`),
  },
);

console.log(`Found ${result.count} cells.`);
// result.masks      : Uint32Array — instance label map at source resolution, 0=background
// result.width      : number      — source image width
// result.height     : number      — source image height
// result.totalMs    : number      — wall-clock time for the segment() call
// result.tiles      : per-tile diagnostics (flow tensors, inference time)
```

## Parameter quick-reference

### `chan` / `chan2`

CPSAM was trained with channel-shuffling augmentation, so the choice rarely matters for segmentation quality. The legacy Cellpose 1–3 semantics are preserved:

| Image type | `chan` | `chan2` |
|---|---|---|
| H&E histology, brightfield, phase contrast | `0` | `0` |
| Fluorescence: green cyto, blue nuclei | `2` | `3` |
| Fluorescence: red cyto, green nuclei | `1` | `2` |
| First run / unknown | `0` | `0` |

### `diameter`

Rescales the image so the median cell occupies ~30 px (CPSAM's training median). Omit to run at native resolution.

| Cells in source image | Suggested |
|---|---|
| Roughly 20–60 px across | leave blank |
| Tiny (5–15 px) | ≈ 10 |
| Large (80+ px) | your visual estimate |

## Performance (M1 Max, Chrome 135+, WebGPU)

| Step | Time | Notes |
|---|---|---|
| Model fetch (cold cache) | ~5 s | 588 MB from local proxy / CDN |
| Model fetch (warm IDB) | <100 ms | IndexedDB hit |
| `ort.InferenceSession.create` | ~1.3 s | one-time per session |
| First inference (cold shader) | ~2.3 s | one-time WebGPU shader compile |
| Steady-state per-tile inference | **277 ms** | 256×256 FP16 |
| Per-tile preprocessing | ~14 ms amortized | normalize + tile copy |
| Full-image dynamics | **74 ms** (400×400) | average + Euler + cluster |
| Abort latency | <50 ms | next tile boundary |

## Architecture

```
input image → buildCpsamChannels → diameterResize → normalizePerChannel → makeTiles
                                                                              ⇣ (per tile, via worker)
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
npm run test         # vitest: 14 parity tests against numpy fixtures
npm run typecheck    # tsc --noEmit
npm run build        # vite library build + tsc --emitDeclarationOnly
npm run demo         # vite serve examples/demo
```

The demo at `examples/demo/` is a complete client that exercises the full pipeline. Point it at a local model file via `examples/demo/public/cpsam_fp16.onnx` (symlink), or change the URL in the Model field.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `e.getValue is not a function` at session-create | Wrong ORT entry point | Import from `onnxruntime-web/webgpu`, not `onnxruntime-web`. |
| `Failed to fetch dynamically imported module: …/ort-wasm-simd-threaded.asyncify.mjs` | Cross-origin dynamic import blocked | Serve ORT WASM files same-origin (or proxy). See `configureOrt({ wasmPaths })`. |
| `Float16Array is not defined` | Browser too old | Chrome ≥135, Safari ≥17.4. No earlier polyfill is supported. |
| `Operation aborted` after AbortSignal fires | Working as intended | Worker terminates; next `segment()` call respawns from IDB cache (~150 ms). |
| Mask overlay has split cells at tile borders | Tile stitching off | Bug — file an issue. (M5 averaging should eliminate this.) |

## Credits

- Model and algorithm: [Cellpose-SAM (Stringer et al., 2025)](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1). Original implementation: [MouseLand/cellpose](https://github.com/MouseLand/cellpose) — BSD-3.
- Inference runtime: [`onnxruntime-web`](https://github.com/microsoft/onnxruntime).

## License

MIT — see [LICENSE](./LICENSE).

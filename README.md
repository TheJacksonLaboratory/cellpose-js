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
    // chan / chan2 omitted → passthrough: R,G,B each become their own
    // independently-normalized network channel (what upstream Cellpose v4 does)
    resample: false, // true → run dynamics at source resolution (upstream default)
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

### Channels — the default is passthrough

**Omit `chan` and `chan2`.** The first up-to-3 source channels are copied straight to the network and normalized independently:

| Source           | Network input               |
| ---------------- | --------------------------- |
| Grayscale (1 ch) | `[gray, 0, 0]`              |
| 2-channel        | `[c0, c1, 0]`               |
| RGB (3 ch)       | `[R, G, B]`                 |
| RGBA (4 ch)      | `[R, G, B]` — alpha dropped |
| N > 4            | first 3 channels            |

This is what upstream Cellpose v4 does. `transforms.convert_image` performs no channel selection at all — it orders the array as YXC, truncates with `x[..., :3]`, and lets `normalize_img` rescale each channel on its own percentiles. CPSAM was trained with channel-shuffling augmentation, so it has no fixed expectation about which slot holds what; upstream's `channels=` argument is deprecated and logs _"Cellpose4 takes inputs with arbitrary channel orders"_.

For an RGB fluorescence composite this matters: passthrough keeps all three markers at full dynamic range, where collapsing to a grayscale mean would blend them.

### `chan` / `chan2` (legacy, opt-in)

Setting **either** option switches to the Cellpose 1–3 selection mapping, so existing parameter choices lift over unchanged:

- **`chan = 0`** — grayscale: the **mean** of the source's color channels (alpha is excluded for RGBA input, i.e. `channels === 4`). Mirrors Cellpose 1–3's `data.mean(axis=-1)` for `channels=[0,0]`. Reach for it when an RGB image's signal really is one thing shown in color and you want it in a single channel.
- **`chan = 1 | 2 | 3`** — pick red / green / blue directly, no averaging.
- **`chan = k` (k ≥ 1)** — selects source channel `k − 1` (0-based), for _any_ channel count. A 5-channel microscopy stack can use `chan = 4` to pick channel 3.
- **`chan2`** — same indexing for the secondary (nuclear) channel; `0` = none. Setting only `chan2` still leaves passthrough mode, with `chan` defaulting to `0`.

> **Multichannel images:** each channel is usually distinct biology (e.g. DAPI vs. GFP), so **never use `chan = 0`** on them — averaging different markers is meaningless. Either stay on the passthrough default (which keeps the first three markers separate) or select explicitly with `chan`/`chan2 ≥ 1`.

| Image type                                             | `chan`   | `chan2`  |
| ------------------------------------------------------ | -------- | -------- |
| Anything with ≤ 3 informative channels                 | _(omit)_ | _(omit)_ |
| RGB whose signal is one thing, want it collapsed       | `0`      | `0`      |
| Fluorescence: green cyto, blue nuclei (only those two) | `2`      | `3`      |
| Fluorescence: red cyto, green nuclei (only those two)  | `1`      | `2`      |
| Multichannel stack: cyto = ch3, nuclei = ch0 (0-based) | `4`      | `1`      |

### `diameter`

Rescales the image so the median cell occupies ~30 px (CPSAM's training median). Omit to run at native resolution.

| Cells in source image   | Suggested            |
| ----------------------- | -------------------- |
| Roughly 20–60 px across | leave blank          |
| Tiny (5–15 px)          | ≈ 10                 |
| Large (80+ px)          | your visual estimate |

### `resample`

Only relevant when `diameter` triggers a resize. Controls **where the flow-dynamics step runs**:

- **`false` (default)** — dynamics run at the resized (network) resolution with the base 200 Euler iterations, and the label map is upscaled nearest-neighbor. Cheapest.
- **`true`** — the predicted flow field and cellprob are bilinear-upsampled back to source resolution first, and dynamics run there with `200 / scale` iterations. This is upstream's default (`models.py:_run_net`, `resample=True`): mask boundaries follow the flow field rather than a blocky upscale, at the cost of running the dynamical system over the full-resolution image.

Upstream ties the iteration count to this choice — `niter_scale = 1 if rescale is None or not resample else rescale` — and so do we. Passing `dynamics.niter` explicitly overrides both.

### `tile`

Must be **256**. CPSAM's position embeddings are baked in at 256/8 = 32×32 tokens and the ONNX export hardcodes H/W to 256, so any other value is rejected (upstream raises the same restriction: _"bsize != 256 is not supported for cpsam"_).

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
input image → buildCpsamChannels → normalizePerChannel → diameterResize → makeTiles
                                                                              ⇣ (per tile)
                                                              ort.InferenceSession.run
                                                                              ⇣
                                                                       averageTiles
                                                                              ⇣
                                                       resample ? resizeChw(flows → source res) : —
                                                                              ⇣
                                                                       computeMasks (Euler + cluster + renumber)
                                                                              ⇣
                                                       resample ? — : nearest-neighbor unresize
                                                                              ⇣
                                                                        Uint32Array masks
```

Normalization runs **before** the diameter resize, matching `models.py:_run_cp` — resizing first would shift the per-channel percentiles that drive normalization.

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

## Upstream parity

This is a port, so it tracks a specific point in [MouseLand/cellpose](https://github.com/MouseLand/cellpose). Current parity baseline: **`a54cb48`** (main, 2026-06-14), covering `transforms.py`, `dynamics.py`, `models.py`, `core.py`, and `vit.py` for the **2D** path.

### Model zoo

Upstream's default model is now **`cpsam_v2`**, with `cpdino` / `cpdino-vitb` alongside it. Cellpose.js ships against `cpsam`:

| Upstream model | Architecture            | Status here                                          |
| -------------- | ----------------------- | ---------------------------------------------------- |
| `cpsam`        | `CPSAM` (SAM ViT-L)     | **Supported** — `cpsam_fp16.onnx`                    |
| `cpsam_v2`     | `CPSAM` (SAM ViT-L)     | Export-compatible, not yet published — see below     |
| `cpdino`       | DINOv3 ViT-L, bsize 384 | Not supported — different architecture and tile size |
| `cpdino-vitb`  | DINOv3 ViT-B, bsize 384 | Not supported                                        |

`cpsam_v2` is **architecturally identical** to `cpsam`: upstream's `models.get_backbone()` inspects the checkpoint for `encoder.cls_token`, finds none, and returns `"sam_vitl"`, so it instantiates the same `CPSAM` class with the same `ps=8` / `bsize=256` / `nout=3` geometry. The existing FP16 ONNX export recipe (see [`docs/STAGE0-RESULTS.md`](./docs/STAGE0-RESULTS.md) — export directly in FP16 from `CPSAM(dtype=torch.float16)`; post-hoc FP16 conversion is broken) therefore applies unchanged; only the weights differ. Once an ONNX build is published, no library change is needed — `fromPretrained()` already accepts any URL:

```ts
const cp = await Cellpose.fromPretrained('https://your-cdn/cpsam_v2_fp16.onnx');
```

### Known divergences

- **`resample` defaults to `false`**, where upstream defaults to `true`. Opt in for upstream-identical mask geometry.
- **Labels are always `Uint32Array`.** Upstream downcasts to `uint16` when an image has fewer than 65,536 masks; changing the returned type here would break consumers, so it stays 32-bit.
- **`normalize99` percentile downsampling is not implemented.** Upstream estimates percentiles from a strided subsample once an image exceeds 224³ ≈ 11.2 M pixels (a 2D image beyond ~3350×3350). Below that threshold the two agree exactly; above it, upstream's percentiles are an approximation of the ones computed here.
- **3D (`do_3D`, `stitch3D`, anisotropy), training, and the model-zoo loader are out of scope.**

## Credits

- Model and algorithm: [Cellpose-SAM (Stringer et al., 2025)](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1). Original implementation: [MouseLand/cellpose](https://github.com/MouseLand/cellpose) — BSD-3.
- Inference runtime: [`onnxruntime-web`](https://github.com/microsoft/onnxruntime).

## Security

See [SECURITY.md](./SECURITY.md) for the threat model, reporting process, and recommended consumer practices.

## License

MIT — see [LICENSE](./LICENSE).

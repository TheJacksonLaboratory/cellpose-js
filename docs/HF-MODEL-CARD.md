---
license: bsd-3-clause
library_name: onnx
pipeline_tag: image-segmentation
base_model: mouseland/cellpose-sam
tags:
  - cellpose
  - cellpose-sam
  - cpsam
  - segmentation
  - cellular-segmentation
  - microscopy
  - bioimage
  - onnx
  - webgpu
  - browser
  - fp16
language:
  - en
---

# cellpose-sam-onnx (CPSAM, FP16, browser-ready)

Single-file FP16 ONNX export of **Cellpose-SAM** (CPSAM), the ViT-L–based cellular
segmentation model from [Stringer et al., 2025](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1).
Intended for in-browser inference via WebGPU through
[`cellpose-js`](https://github.com/belkassaby/Cellpose.js), but usable from any
ONNX Runtime backend that supports opset 18 and FP16 graph IO.

- **Architecture:** SAM ViT-L image encoder (modified — patch size 8, no windowed
  attention) + 3-channel dense regression head (`flow_y`, `flow_x`, `cellprob`).
- **Source weights:** [`mouseland/cellpose-sam`](https://huggingface.co/mouseland/cellpose-sam)
  (1.23 GB PyTorch checkpoint, 304.6 M params).
- **File:** `cpsam_fp16.onnx` — **588 MB**, self-contained (weights merged into
  the graph, no external `.data` sidecar).
- **Graph IO dtype:** `float16`.
- **Input:** `image: tensor(float16)`, shape `(1, 3, 256, 256)`, RGB,
  per-channel percentile-normalized to ~[0, 1].
- **Output:** `flows_cellprob: tensor(float16)`, shape `(1, 3, 256, 256)` —
  channels are `(flow_y, flow_x, cellprob)`.
- **Opset:** 18.
- **License:** BSD-3-Clause (inherited from upstream Cellpose).

## TL;DR

This is the **same model** as `mouseland/cellpose-sam`, just re-packaged as a
588 MB FP16 ONNX file that browsers can load and run on WebGPU. No retraining,
no pruning, no architecture changes. Numerical parity vs the PyTorch source is
**1.24e-05 worst max-abs-error** across 10 random tiles (gate 1e-3).

If you want a smaller model, this is not it — that's Phase 2 (SlimCPSAM),
which has not been built yet.

## Intended use

- **Browser-side cellular segmentation** in microscopy applications: H&E,
  brightfield, phase contrast, fluorescence (multi-channel).
- Drop-in inference engine for any ORT-compatible runtime (Python, Node, Rust,
  C++) at FP16. The browser is the primary target but not the only one.
- 2D images only. CPSAM's 3D `gradient_tracking_3D` routine is not part of the
  export and is not represented in this graph.

## How to use

### In the browser, via `cellpose-js` (recommended)

```ts
import { Cellpose, configureOrt } from 'cellpose-js';

configureOrt({ wasmPaths: '/ort/' }); // serve ORT WASM sidecars same-origin

const cp = await Cellpose.fromPretrained(
  'https://huggingface.co/ballon999/cellpose-sam-onnx/resolve/main/cpsam_fp16.onnx',
  { preload: true },
);

const result = await cp.segment(
  { data: imageData.data, width, height, channels: 4 },
  { diameter: 30, cellprob_threshold: 0, chan: 0, chan2: 0 },
);
// result.masks: Uint32Array — instance label map at source resolution
```

The first call fetches 588 MB from the Hub; subsequent calls hit IndexedDB and
cold-start in under ~2 s.

See the [cellpose-js README](https://github.com/belkassaby/Cellpose.js#readme)
for the full API, parameter reference, and Python parity notes.

### With ONNX Runtime directly (Python)

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("cpsam_fp16.onnx", providers=["CPUExecutionProvider"])
tile = np.random.rand(1, 3, 256, 256).astype(np.float16)
out = sess.run(None, {"image": tile})[0]   # (1, 3, 256, 256) float16
flow_y, flow_x, cellprob = out[0, 0], out[0, 1], out[0, 2]
```

For flow-dynamics postprocessing (Euler integration → convergence clustering →
connected components → size/flow filtering), use either:

- the JS port in [`cellpose-js/src/dynamics`](https://github.com/belkassaby/Cellpose.js/tree/main/src), or
- the original Python implementation in
  [`cellpose.dynamics`](https://cellpose.readthedocs.io/) — input/output
  contracts match.

## Browser requirements

Because the graph IO is FP16, ORT-web needs the **native `Float16Array`**
typed-array, not a `Uint16Array` bit-pattern. That requires:

- **Chrome ≥ 135** (Feb 2025), or
- **Safari ≥ 17.4**.
- **WebGPU** available (`'gpu' in navigator`). No WASM fallback in v1 of the
  consumer (`cellpose-js`) — see "Why not FP32?" below.

Older browsers fail fast with a clear error in `cellpose-js`. For direct ORT
use, ORT will throw on session create or input binding.

## Performance (browser, M1 Max, Chrome 135+)

Measured end-to-end through `cellpose-js`:

| Step                                        | Time             |
| ------------------------------------------- | ---------------- |
| Cold model fetch (588 MB, CDN)              | ~5 s             |
| Warm fetch (IndexedDB)                      | < 100 ms         |
| `ort.InferenceSession.create`               | ~1.3 s           |
| Cold shader compile (first forward)         | ~2.3 s           |
| Steady-state per-tile inference (256×256)   | **~277 ms**      |
| Per-tile preprocess (normalize + tile copy) | ~14 ms amortized |
| Full-image flow dynamics (400×400)          | **~74 ms**       |

ORT-web 1.26 is ~2.3× faster than 1.20 on the WebGPU kernels — the steady-state
277 ms number is on 1.26. Stage 0 originally measured 628 ms/tile on 1.20.

## How the model was generated

The export path is documented in
[`docs/STAGE0-RESULTS.md`](https://github.com/belkassaby/Cellpose.js/blob/main/docs/STAGE0-RESULTS.md)
and [`docs/PLAN.md §1.5, §2`](https://github.com/belkassaby/Cellpose.js/blob/main/docs/PLAN.md).
The short version:

1. **Source weights**: `mouseland/cellpose-sam` (PyTorch, 1.23 GB, 304.6 M params).
2. **Wrap** `cellpose.vit_sam.Transformer` (this is _not_ a HuggingFace
   Transformers class — `optimum-cli` does not apply here).
3. **Instantiate in FP16 directly**: `Transformer(dtype=torch.float16)` then
   load the FP32 checkpoint and cast. Post-export FP16 conversion via
   `onnxconverter-common` or `onnxruntime.transformers.float16` produced
   broken graphs on the dynamo-exported topology (dangling FP16→FP32 type
   mismatches and duplicate node names respectively) — re-exporting from a
   natively-FP16 `nn.Module` is the only path that worked.
4. **Export with `torch.onnx.export(..., dynamo=True, strict=True)`** at
   opset 17 (auto-upgraded to 18 by the dynamo exporter). Requires `onnxscript`
   as an extra dependency. `strict=False` failed; `strict=True` succeeded.
5. **`dynamic_axes`**: batch only. H/W are hardcoded to 256 by the dynamo
   exporter — acceptable because CPSAM is always tiled at 256×256.
6. **Merge externalized weights** back into the graph file via
   `onnx.save_model(..., save_as_external_data=False)`. The 588 MB result fits
   comfortably under the 2 GB protobuf limit, so the browser fetches one file
   instead of `.onnx` + `.onnx.data`.
7. **Parity check** vs PyTorch on 10 deterministic random tiles
   (`(1, 3, 256, 256)` FP32, seed 0): **worst max abs error 1.24e-05**, mean
   8.96e-06. Gate was 1e-3.

The PyTorch and exporter versions used: `torch 2.12.0`, `cellpose 4.1.1`,
`onnx 1.21.0`, `onnxruntime 1.26.0`, `onnxscript` (latest at export time).

## How this differs from the original `mouseland/cellpose-sam`

It is the **same network, same weights, same outputs** — only the serialization
format differs. Specifically:

| Aspect          | `mouseland/cellpose-sam` (PyTorch) | This repo (ONNX FP16)                 |
| --------------- | ---------------------------------- | ------------------------------------- |
| Format          | PyTorch `.pt` checkpoint           | ONNX, single file                     |
| Size            | 1.23 GB                            | **588 MB**                            |
| Precision       | FP32                               | **FP16**                              |
| Runtime targets | PyTorch (Python only)              | ORT WebGPU/CUDA/CPU/CoreML/DirectML   |
| Input dtype     | `float32`                          | **`float16`** (native `Float16Array`) |
| Input shape     | Variable; CPSAM tiles internally   | **Fixed `(1, 3, 256, 256)`**          |
| Postprocessing  | Bundled in `cellpose.dynamics`     | **Not included** — caller's job       |
| 3D segmentation | Yes (`gradient_tracking_3D`)       | **No** — 2D only                      |
| Promptable      | No (CPSAM is dense regression)     | No (unchanged)                        |

**Numerical:** worst observed max abs error vs the FP32 PyTorch reference on the
same input is **1.24e-05** — the FP16 export is numerically indistinguishable
from the original at the granularity that matters for downstream flow
dynamics.

**What's not here:**

- **The flow-dynamics postprocessing.** This repo ships the encoder + head
  only. Postprocessing (Euler integration, convergence clustering, connected
  components, size / flow-consistency filtering) lives in `cellpose-js`
  (TypeScript port, ~500 LOC) and `cellpose` itself (Python). Output of this
  ONNX graph is raw `(flow_y, flow_x, cellprob)` — you still need to turn that
  into instance masks.
- **3D mode.** CPSAM's Python implementation handles z-stacks via a separate
  routine; that's out of scope here.
- **Prompt encoder / mask decoder.** CPSAM does not have them — unlike SAM /
  SlimSAM, CPSAM is dense regression, not promptable mask generation.

## Why FP16 (and why not FP32 or INT8)?

- **WebGPU runs FP16 well.** The whole point of this export is browser
  inference, and the steady-state tile latency is fine (277 ms on M1 Max).
- **FP32 would double the download.** ~1.1 GB instead of 588 MB, and the same
  WebGPU adapter ends up converting much of the graph internally anyway.
- **INT8 wasn't worth the validation budget.** Flow regression is more
  numerically sensitive than mask classification, so an INT8 path would need
  end-to-end IoU validation against the original Cellpose. We may revisit
  this in a future release if size complaints arrive.

## Limitations and caveats

- **Fixed input size 256×256.** Larger images must be tiled by the caller.
  `cellpose-js` does this transparently with 32-px overlap.
- **Browser version floor (Chrome 135 / Safari 17.4).** Native `Float16Array`
  is non-negotiable for FP16 graph IO under ORT-web 1.20+.
- **Not promptable.** This is dense per-pixel regression, not SAM-style
  prompted segmentation.
- **No domain-specialized variants.** This is the generalist CPSAM. Historical
  Cellpose variants (cyto / cyto2 / cyto3 / nuclei) and Omnipose bacteria
  models are not provided here — they are planned as **Phase 2** SlimCPSAM
  finetunes and have not been built yet.
- **No INT8 / FP32 fallback.** WebGPU only.

## Citation

If you use this model, please cite the original Cellpose-SAM paper:

```
Stringer, C., Pachitariu, M. et al.
Cellpose-SAM: superhuman generalization for cellular segmentation.
bioRxiv 2025.04.28.651001 (2025).
https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1
```

## Provenance and reproducibility

- **Source checkpoint:** `mouseland/cellpose-sam` on Hugging Face Hub.
- **Export scripts and ONNX artifact ETag** (`52fd6881…`) are recorded in
  [`docs/STAGE0-RESULTS.md`](https://github.com/belkassaby/Cellpose.js/blob/main/docs/STAGE0-RESULTS.md).
- **Parity test fixtures** (numpy-generated FP32 reference tiles + expected
  flow outputs) live in `tests/fixtures/` in the `cellpose-js` repo.

## License

- This ONNX artifact: **BSD-3-Clause**, inherited from MouseLand/cellpose.
- The `cellpose-js` consumer library: **MIT** (compatible).
- Use, redistribution, and modification are permitted; attribution to the
  original Cellpose authors is required by the BSD-3 license.

## Maintainers

- ONNX export, `cellpose-js`, and this card: [@belkassaby](https://github.com/belkassaby)
  (HF: [`ballon999`](https://huggingface.co/ballon999)) — same person, different
  username on each platform.
- Original Cellpose-SAM model and algorithm: the [MouseLand](https://github.com/MouseLand) team.

## Related

- **Code:** [`belkassaby/Cellpose.js`](https://github.com/belkassaby/Cellpose.js) — TypeScript inference + dynamics port.
- **npm:** [`cellpose-js`](https://www.npmjs.com/package/cellpose-js).
- **Upstream:** [`MouseLand/cellpose`](https://github.com/MouseLand/cellpose) and [`mouseland/cellpose-sam`](https://huggingface.co/mouseland/cellpose-sam).
- **Paper:** [Cellpose-SAM (bioRxiv 2025.04.28)](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1).

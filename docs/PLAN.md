# Cellpose.js — Feasibility Review & Implementation Plan

Browser-side Cellpose-SAM via a new `Cellpose.js` package, consumed by jit-ui as a new client-side operation in the processing pipeline.

**Scope decisions (locked in):**
- **Phase 1 only ships first** — port stock CPSAM to the browser. No SlimSAM-style compression in Phase 1.
- **WebGPU required** — ship FP16, no WASM fallback in v1.
- **Separate repo** `Cellpose.js` — not an Nx lib inside jit-ui.

**Phase 2 planned but NOT started** — SlimSAM-style compression of CPSAM (§7) plus domain-specialized slim finetunes (cyto/cyto2/cyto3, nuclei, Omnipose bacteria, TissueNet, LiveCell). Research/ML project, multi-month, gated on Phase 1 shipping. Do not begin without explicit go.

**Stage 0 status: PASS** (2026-05-14, run on M1 Max). Full run report:
[`STAGE0-RESULTS.md`](./STAGE0-RESULTS.md).
- Spike A: worst FP32 parity error **1.24e-05** vs gate 1e-3.
- Spike B: median **628 ms/tile** WebGPU vs gate 2 s; cold start ~2.3 s.
- Deployment artifact: **588 MB** single-file FP16 ONNX with FP16 graph IO.

---

## 1. Feasibility Review

### 1.1 What CPSAM actually is

From `cellpose.vit_sam` and the Cellpose-SAM preprint (bioRxiv 2025.04.28):

- **Backbone:** SAM **ViT-L** image encoder, modified:
  - patch size 16 → **8** (with weight downsampling `w[:,:,::16//ps,::16//ps]`)
  - `window_size = 0` on every block — all attention is global, no windowed attention
  - positional embeddings subsampled to match the new patch size
- **Head:** very small dense regression head — `Conv2d(256, nout·ps²)` then a transposed conv that unfolds tokens back to pixels. `nout = 3` channels: **flow_y, flow_x, cellprob**.
- **No prompt encoder. No mask decoder.** CPSAM is *not* promptable. It is a dense per-pixel regressor that reuses SAM weights as initialization.
- **Weights:** single ~**1.23 GB** PyTorch checkpoint at `mouseland/cellpose-sam` on Hugging Face.
- **Training:** tiles around `bsize = 256`, diameters 7.5–120 px, 2D only.
- **Post-processing (Python today):** flow-field Euler integration → pixel convergence clustering → connected components → size / flow-consistency filtering. NumPy + SciPy + Numba + a touch of OpenCV. ~500 LOC of real algorithm, no model.

### 1.2 What SlimSAM actually is — and why it isn't a CPSAM drop-in

- SlimSAM (Chen et al., NeurIPS 2024) is **channel-pruned + distilled SAM-ViT-B** plus the *standard* SAM prompt encoder and SAM mask decoder. `Xenova/slimsam-50-uniform` is the ONNX export already wired into Transformers.js (jit-ui already uses the sibling `Xenova/sam-vit-base` in `transformers.worker.ts:238`).
- It is **promptable mask generation**, not dense regression. Outputs are `pred_masks` + `iou_scores`, given `input_points` / `input_boxes`.
- It is **not a drop-in replacement** for CPSAM's head. Plugging SlimSAM in where CPSAM goes would produce per-prompt mask candidates, not flow-derived instance masks. Making a SlimSAM-style CPSAM would require either (a) pruning CPSAM's ViT-L while keeping its flow head, or (b) training a new flow head on top of SlimSAM's encoder. Both are multi-month ML work, out of scope for Phase 1.

### 1.3 Specific feasibility checks

- **ONNX exportability of CPSAM encoder.** Architecture is standard ViT ops + global attention. The only non-export-friendly bits (`torch.rand`/`linspace` for stochastic depth, dropout) fire only in `.train()` mode — `.eval()` exports cleanly. **Confirmed in Stage 0:** torch 2.12 dynamo exporter with `strict=True` succeeds at opset 18.
- **Browser model size.** ViT-L is **304.6 M params** (measured). FP16 ONNX = **588 MB** (measured, single-file self-contained). INT8 ≈ ~310 MB. jit-ui already ships a 375 MB SAM-ViT-B, so 588 MB is within precedent.
- **Tile size 256 × global attention.** 256 / 8 = 32 tokens per side → 1024 tokens. Global attention at 1024 tokens is fine — equivalent compute to SAM's standard 64×64 grid.
- **Per-tile latency.** **Measured 628 ms median on M1 Max WebGPU** for `(1, 3, 256, 256)` FP16 input. Cold start ~2.3 s on first forward pass (shader compile). Session create ~1.3 s.
- **Flow dynamics in JS.** The Euler integration loop is the only non-trivial bit. ~200 lines of straightforward JS, with the option to promote the hot loop to WASM later if profiling demands it. No blockers.
- **Existing pipeline fit.** The current `sam-auto-segment` worker (`transformers.worker.ts:237`) is the right template. CPSAM is *simpler* to integrate — one forward pass per tile, no per-point grid, then postprocessing.
- **Tiling.** CPSAM expects ~256-px tiles. Large images need a tile-and-stitch loop with overlap and mask-merge. ~150 LOC, required for production use on whole-slide imagery.
- **Browser version floor.** The deployed FP16 ONNX has **FP16 graph IO** (input and output tensors are `float16`). ORT-web 1.20 requires the native `Float16Array` type for these tensors — available in **Chrome ≥135 (Feb 2025) and Safari ≥17.4**. Older browsers are not supported in v1.

### 1.4 Honest risks

- **Browser memory at ViT-L scale.** Whole-slide images at native resolution will OOM. Mitigation: tile + chunk + don't retain full-resolution embeddings.
- **Cold-start UX.** First-run penalty is ~2.3 s of WebGPU shader compile **plus** the 588 MB cold-cache model fetch from CDN. Mandatory mitigations in v1 (see Milestone 1): IndexedDB cache, eager session creation while the user configures params.
- **License.** Cellpose is BSD-3; SlimSAM is Apache-2.0. Both fine. The new repo can be MIT or BSD without friction.

### 1.5 Why FP16 over INT8 in v1

WebGPU is the required runtime, and WebGPU runs FP16 natively well. Flow regression is more numerically sensitive than mask classification, so INT8 introduces a quality-validation budget we don't need to spend in v1. Ship FP16. Revisit INT8 only if download size complaints arrive.

**FP16 production note (Stage 0 finding):** post-export FP16 conversion is *broken* on the dynamo-exported graph — `onnxconverter-common` leaves dangling FP16→FP32 type mismatches, and `onnxruntime.transformers.float16` generates duplicate node names. The working path is to **export directly in FP16** by instantiating `cellpose.vit_sam.Transformer(dtype=torch.float16)` and tracing from there. This produces FP16 graph IO, which is why the browser-version floor in §1.3 applies.

---

## 2. Hard-Dependency Spikes (Stage 0) — COMPLETE

Full results: [`STAGE0-RESULTS.md`](./STAGE0-RESULTS.md).
Scratch rig at `~/cellpose-js-spike/` (Python 3.11 venv, export scripts, ONNX
artifacts, browser harness). Both gates passed on 2026-05-14.

### Spike A — ONNX export parity ✅ PASS
- **NOT** via `optimum-cli` (cellpose's `Transformer` is not a HF Transformers
  class). Use `torch.onnx.export(net, dummy, …)` directly. Requires `onnxscript`
  as an extra dep for the torch 2.12 dynamo exporter.
- Use `dynamic_axes` for batch only; H/W get hardcoded by the dynamo exporter
  but Phase 1 always tiles at 256 so this is acceptable.
- Verify FP32 parity vs PyTorch on 10 deterministic random tiles: max abs error
  ≤ 1e-3.
- **Measured worst error: 1.24e-05** (passes by ~80×).

### Spike B — WebGPU tile latency ✅ PASS
- FP16 ONNX (export with `Transformer(dtype=torch.float16)` — see §1.5 finding).
- After export, **merge externalized weights back into the .onnx file** with
  `onnx.save_model(..., save_as_external_data=False)` so the browser fetches a
  single artifact (588 MB fits comfortably under the 2 GB protobuf limit).
- Load via `onnxruntime-web` UMD build (`dist/ort.webgpu.min.js`) — the jsDelivr
  `+esm` wrapper does **not** work for ort-web.
- Send FP16 input as native `Float16Array` (not a Uint16Array bit-pattern).
- Benchmark one 256×256 tile forward pass on a mid-range laptop GPU.
- **Measured: 628 ms median on M1 Max** (passes by ~3.2×). Cold start ~2.3 s.

---

## 3. Repo: `Cellpose.js`

**Stack**

- TypeScript, ESM only
- Vite library mode
- `onnxruntime-web` 1.20+ directly (UMD `ort.webgpu.min.js`, not the ESM `+esm`
  wrapper — that 404s on jsDelivr for ort-web). We don't need
  `@huggingface/transformers` — no tokenizers or pipelines, just an ORT session.
- Vitest for unit tests
- No Angular dependency

**Browser support**

- Chrome ≥135 (Feb 2025), Safari ≥17.4. Required for native `Float16Array`,
  which ORT-web demands for FP16 graph IO.
- WebGPU required (already a Phase 1 decision; reaffirmed by Spike B).

**Weights distribution**

- Host the **588 MB single-file FP16 ONNX** on Hugging Face Hub under your
  account (e.g. `belkassaby/cellpose-sam-onnx`).
- Download-and-cache via IndexedDB on first use — same UX users already accept
  for the 375 MB `sam-vit-base`. Mandatory in v1 (Stage 0 measured 588 MB
  cold-cache fetch + 2.3 s shader compile on first run).
- Do **not** bundle weights into the npm package.

**Public API**

```ts
const cp = await Cellpose.fromPretrained('belkassaby/cellpose-sam-onnx', {
  preload: true,  // create the ORT session eagerly; trades latency at construct time for a fast first segment()
});
const result = await cp.segment(rgbaImage, {
  diameter: 30,
  tile: 256,
  overlap: 32,
  cellprob_threshold: 0.0,
  flow_threshold: 0.4,
  onProgress: (done, total) => {},
  signal: abortController.signal,
});
// result.masks  : Uint32Array  — instance label map at original resolution
// result.flows  : Float32Array — converted from FP16 model output for dynamics
// result.cellprob: Float32Array — converted from FP16 model output
```

The FP16→FP32 conversion at the model boundary keeps the flow-dynamics
implementation in plain FP32 numerics (simpler, no risk of underflow during
Euler integration).

---

## 4. Milestones

| # | Milestone | Effort | Exit criterion |
|---|-----------|--------|----------------|
| 0 | Stage-0 spikes (export parity + WebGPU latency) | 2 days | **✅ DONE** — see [`STAGE0-RESULTS.md`](./STAGE0-RESULTS.md) |
| 1 | Repo skeleton, ORT-WebGPU session loader, model fetch + IndexedDB cache | 2 days | Loads model, runs identity forward pass on a tile |
| 2 | Pre-processing port (`cellpose.transforms`): percentile normalization, diameter-resize, tiling, channel selection | 3 days | Bit-exact match vs Python on a fixture set |
| 3 | Per-tile inference with WebGPU EP, progress callback, abort signal | 2 days | Throughput meets Spike B benchmark; abort terminates within 100 ms |
| 4 | Flow dynamics post-processing port (`cellpose.dynamics`): Euler integration, convergence clustering, connected components, size + flow-consistency filtering | 5 days | Per-tile output matches Python CPSAM mean IoU ≥ 0.9 on a 20-image reference set |
| 5 | Tile stitching with IoU-based label merging in overlap regions | 2 days | Full-image output matches Python CPSAM mean IoU ≥ 0.85 |
| 6 | API polish, README with quality + perf numbers, npm publish | 2 days | `npm i cellpose-js` works in a hello-world Vite app |
| 7 | **jit-ui integration** | 2 days | New `cellpose-segment` op visible in pipeline dialog, runs, overlays masks, abortable |

**Total:** ~3.5 weeks of focused work, assuming both spikes pass.

---

## 5. Per-Milestone Detail

### Milestone 0 — Stage-0 spikes
See §2. Two scripts, one for export parity, one for WebGPU latency. Output is a go/no-go memo.

### Milestone 1 — Repo skeleton
- New repo `Cellpose.js`, BSD-3 or MIT.
- `package.json` with ESM-only build via Vite library mode.
- `Cellpose.fromPretrained(modelId, { preload })` loads ORT session with WebGPU
  EP, fails fast with a clear error if any of: WebGPU is unavailable,
  `Float16Array` is undefined (browser too old), or session creation throws.
- IndexedDB cache keyed by model ID + version hash. First-run downloads 588 MB;
  subsequent runs read from IndexedDB.
- `preload: true` option creates the ORT session eagerly at `fromPretrained()`
  time so the 1.3 s session-create + 2.3 s cold shader compile happen while the
  user is configuring params, not while waiting for `segment()` to start.

### Milestone 2 — Pre-processing
Port from `cellpose.transforms`:
1. **Normalization:** per-channel 1st/99th percentile rescaling to [0, 1], optional invert.
2. **Resize-to-diameter:** if user supplies `diameter`, resize so target ≈ 30 px (CPSAM's training median).
3. **Tiling:** split image into 256×256 tiles with 32-px overlap; pad edges; output `{ tile, tx, ty }` records.
4. **Channel handling:** CPSAM expects 3-ch input. Map grayscale → 3-ch; for multi-channel fluorescence, mirror Cellpose's `chan` / `chan2` semantics with user-selectable nuclear/cyto channels.

### Milestone 3 — Inference
- WebGPU EP only.
- Input tensor: `(1, 3, 256, 256)` as `Float16Array`. Build from the
  preprocessed `Float32Array` via direct assignment (`Float16Array` auto-rounds
  on store).
- One forward pass per tile → output tensor `(1, 3, 256, 256)` of dtype
  `float16` for `(flow_y, flow_x, cellprob)`.
- **Convert output to `Float32Array` at the model boundary** before handing off
  to dynamics. Native `Float16Array.prototype.set(other)` casts on write; or
  iterate once and assign into a fresh `Float32Array`.
- Cancel-on-abort by terminating the worker, mirroring `transformers-engine.ts:182-191`.

### Milestone 4 — Flow dynamics (hardest stage)
Port `cellpose.dynamics`:
1. Threshold `cellprob > threshold` (default 0).
2. Euler-integrate flows ~200 steps; record each pixel's convergence point.
3. Histogram-bin convergence points into a 2D grid; peaks → seed labels.
4. Connected components on the label map (small JS lib or hand-roll).
5. Drop tiny objects; drop masks whose mean flow disagrees with re-predicted flow (`flow_threshold`).

Implement in pure JS first for correctness, profile, promote hot loops to WASM only if needed.

### Milestone 5 — Tile stitching
- Run dynamics per tile.
- Merge across overlapping borders by matching labels with IoU > 0.5 in the overlap region.
- Renumber labels globally.
- Return a `Uint32Array` instance label map at the original (pre-resize) resolution.

### Milestone 6 — Polish & publish
- README with quality (IoU vs Python CPSAM) and perf (ms/tile, ms/megapixel) numbers.
- Versioned npm release.
- Public model ID stable.

### Milestone 7 — jit-ui integration

Touchpoints in this repo:

- `apps/jit-ui/package.json` — add `cellpose-js` dependency.
- **Sibling engine** (recommended over folding into the transformers.js engine): new `apps/jit-ui/src/app/main/models/processing-pipeline/engines/cellpose/cellpose-engine.ts` implementing the `ProcessingEngine` interface. `cellpose-js` shares nothing with transformers.js beyond the worker-dispatch pattern, so keeping them separate is cleaner long-term.
- New worker `engines/cellpose/cellpose.worker.ts` modeled on `transformers.worker.ts:237-307`. Off-thread model load + inference. Honor the abort contract from `engine.model.ts:34`.
- Register in `engine-registry.service.ts`.
- Operation descriptor: `id: 'cellpose-segment'`, category `segmentation`, `runsOn: 'client'`. Params: `model`, `diameter`, `cellprob_threshold`, `flow_threshold`, `channels`.
- `apps/jit-ui/src/app/main/components/process/image-processing/image-processing.component.ts:17` — add `'cellpose-segment'` to the allow-list.
- Overlay rendering: instance label map → colored mask via the existing `segmentColor()` palette in `transformers.worker.ts`.

---

## 6. Explicitly Out of Scope (v1)

- Server-side `Cellpose.js`. CPSAM on the server already exists via the JIT registry (`CPSAM` model id is in `request.ts:472`).
- 3D segmentation. Python CPSAM uses a separate `gradient_tracking_3D` routine; not ported.
- Training UI, model uploader, custom-model support.
- INT8 path. Revisit only if FP16 download size becomes a complaint.
- WASM fallback. WebGPU required.
- **Pre-Chrome-135 / pre-Safari-17.4 browsers.** Native `Float16Array` is mandatory for FP16 graph IO. Reaching older browsers would require a working post-export FP16 conversion path (currently broken, see §1.5) or an FP32 model (~1.1 GB, double the bandwidth).
- SlimSAM-style compression of CPSAM. Separate research project; not part of Phase 1.

---

## 7. Phase 2 — SlimCPSAM and Domain-Specialized Slim Models (PLANNED, NOT STARTED)

A research workstream to produce small (~50–150 MB) browser-runnable cellular
segmentation models. Gated on Phase 1 shipping. Multi-month timeline. Requires
GPU compute, labeled data access, and ML iteration. **Do not begin without an
explicit go decision and a compute/budget allocation.**

### 7.1 Reframing the model list

The list of targets needs cleanup before planning. The original ask combined
architectures, model variants, and datasets:

| Item                  | What it actually is                                              | SlimSAM strategy                                            |
|-----------------------|------------------------------------------------------------------|-------------------------------------------------------------|
| **CPSAM**             | ViT-L + flow head (Cellpose 4 default)                           | **Compress directly** — channel-prune + distill ViT-L       |
| **cyto, cyto2, cyto3**| Pre-v4 **U-Net** generalist cytoplasm models (Cellpose 1/2/3)    | Re-train slim CPSAM head on cyto-style data — *not* prune U-Net |
| **nuclei**            | Pre-v4 U-Net nuclei model                                        | Same as cyto: finetune slim CPSAM on nuclei corpus          |
| **Bacteria (omni)**   | Omnipose `bact_phase_omni` / `bact_fluor_omni` — U-Net + Omnipose distance-transform postprocessing | Finetune slim CPSAM on bacteria images; keep CPSAM flow postprocessing (no need to port Omnipose dynamics) |
| **Bacteria (omni+)**  | Omnipose's improved bacteria models (`bact_phase_cp`, `bact_phase_omnipose+`) | Same as above                                               |
| **TissueNet**         | **Dataset**, not a model. Multi-channel tissue imaging.          | Finetune slim CPSAM on TissueNet                            |
| **LiveCell**          | **Dataset**, not a model. Phase-contrast cell line images.       | Finetune slim CPSAM on LiveCell                             |

Key insight: only CPSAM itself is a SAM-derived architecture. Everything else
in the list is either a different architecture (U-Net) or a dataset. So the
strategy is **not** "make a SlimSAM version of each existing model" — it is
"build one SlimCPSAM (the architecture-compression step), then produce 7
domain-specialized finetunes from that single slim backbone." This gives us
the same domain coverage as historical Cellpose without porting U-Net code or
Omnipose's distance-transform postprocessing.

### 7.2 Compute, data, and timeline preconditions

| Resource | Need | Notes |
|---|---|---|
| GPU compute | ~500–1500 GPU-hours | A100/H100 preferred; a 4090 desktop is the floor (slower). The pruning-distill loop dominates. |
| Labeled data | Public training corpora for each domain | TissueNet (gated — research registration), LiveCell (public), Omnipose bacteria (public via Cutler lab), Cellpose 4 training set (public via the Cellpose-SAM release), cyto/nuclei legacy (public via Cellpose 1–3 release) |
| Storage | ~500 GB | Datasets + checkpoints + intermediate distillation states |
| Eval infrastructure | Per-domain held-out test splits + matching metrics (per-instance IoU, AP@0.5, segmentation accuracy) | Reuse Cellpose's evaluation harness where possible |
| Personnel | One ML engineer focused for ~3 months, or shared across ~5 months | The pruning recipe is non-trivial; expect iteration |
| Cost | $5–20K in cloud GPU if not done on owned hardware | Lower if 4090 desktop is used end-to-end |

If any of these is missing, **the right call is to skip Phase 2** and rely on
Phase 1's full-precision CPSAM in the browser.

### 7.3 Sub-phase A — Build SlimCPSAM (the base slim model)

The one-time architecture-compression step. Output: a single ~50–150 MB ONNX
model with the same input/output contract as CPSAM but a pruned ViT backbone.

**Approach.** Apply SlimSAM's recipe (Chen et al., NeurIPS 2024) to CPSAM's
ViT-L encoder, keeping CPSAM's flow regression head. SlimSAM was developed
against SAM-ViT-B, so the recipe needs adaptation for ViT-L (longer training,
adjusted pruning ratios). The flow head is small (~2 MB) and stays untouched
structurally but may need finetuning after backbone pruning.

**Steps**
1. **Replicate SlimSAM training infra.** Reproduce the public SlimSAM training
   pipeline on a small SAM-ViT-B sanity check before touching CPSAM. Goal:
   confirm we can hit the published SlimSAM-50 IoU numbers within 2 points.
   Time-box: 2 weeks.
2. **Adapt to ViT-L + CPSAM head.** Modify the SlimSAM training loop:
   - Teacher: full CPSAM (ViT-L, 304 M params).
   - Student: pruned ViT-L (target ~50% channels retained on ViT-S/B-class
     scale, OR retrain a ViT-B-sized student from scratch with SlimSAM-style
     distillation).
   - Loss: SlimSAM's intermediate feature alignment + a regression loss on the
     CPSAM flow outputs (`flow_y`, `flow_x`, `cellprob`).
   - Training data: Cellpose 4's public training corpus (the same data CPSAM
     was trained on).
   Time-box: 4 weeks of training iteration.
3. **Quality gate.** SlimCPSAM mean per-instance IoU on the Cellpose 4
   validation set must be **within 3 points** of full CPSAM. If we can't hit
   that, abandon Phase 2 — the slim model is not worth a domain-specific
   training campaign that bakes in the quality gap.
4. **Export to ONNX FP16**, single-file, same IO contract as Phase 1's CPSAM
   ONNX so `cellpose-js` can swap models transparently.

**Deliverable**: `belkassaby/slimcpsam-onnx` on HF Hub, ~50–150 MB.

### 7.4 Sub-phase B — Domain-specialized slim finetunes

Once SlimCPSAM exists and clears the quality gate, finetune it on each domain
corpus.

| Slim model | Training corpus | Source | License notes |
|---|---|---|---|
| `slimcpsam-cyto`     | Cellpose 1 cyto training data        | MouseLand `cellpose` release | BSD-3 |
| `slimcpsam-cyto2`    | Cellpose 2 cyto2 training data       | MouseLand `cellpose` release | BSD-3 |
| `slimcpsam-cyto3`    | Cellpose 3 cyto3 training data       | MouseLand `cellpose` release | BSD-3 |
| `slimcpsam-nuclei`   | Cellpose nuclei training data        | MouseLand `cellpose` release | BSD-3 |
| `slimcpsam-bact-omni`| Omnipose `bact_phase_omni` + `bact_fluor_omni` training data | kcutler `omnipose` release | MIT, with Cutler-lab attribution |
| `slimcpsam-bact-omni+`| Omnipose's improved bacteria corpus | kcutler `omnipose` release | MIT |
| `slimcpsam-tissuenet`| TissueNet                            | vanvalenlab (Caltech)        | Research-use license; registration required |
| `slimcpsam-livecell` | LiveCell                             | Sartorius / Chalmers         | CC-BY-NC 4.0 (non-commercial!) |

**Strategy per finetune**
1. **Frozen backbone + head finetune (1–3 epochs).** Cheap, fast, often
   sufficient if the domain isn't far from CPSAM's training distribution.
2. **Full finetune (5–10 epochs)** only if the frozen-backbone pass doesn't
   meet a per-domain quality gate (e.g., AP@0.5 within 2 points of the
   original full-precision Cellpose/Omnipose model on that domain's test set).
3. **Per-domain evaluation against two baselines:**
   - The historical Cellpose/Omnipose model for that domain (the user-facing
     comparison: "is the slim version still good enough?").
   - Full CPSAM (the upper-bound comparison: "how much accuracy are we trading
     for size?").
4. **Re-export to ONNX FP16** with the same IO contract; publish to HF Hub.

**Effort per model**: ~3–5 days of training + eval, parallelizable across GPUs.
Total: ~6–8 weeks if run serially, ~2–3 weeks with 3+ GPUs in parallel.

**License gotcha — LiveCell.** LiveCell's CC-BY-NC 4.0 license restricts
commercial use. If jit-ui or downstream Jackson Lab products have any
commercial use, `slimcpsam-livecell` must be flagged as non-commercial-only
in the package metadata, or the dataset must be replaced. Decide before
training, not after.

### 7.5 Shipping plan in `cellpose-js` and jit-ui

After Phase 2, the `cellpose-js` API extends to support model selection:

```ts
const cp = await Cellpose.fromPretrained('slimcpsam-cyto3', { preload: true });
// or
const cp = await Cellpose.fromPretrained('cpsam', { preload: true });   // Phase 1 full-precision
```

In jit-ui's pipeline-dialog operation descriptor, add a `model` dropdown to the
`cellpose-segment` op with the 8 slim variants plus the full CPSAM. Default:
`slimcpsam-cyto3` (smallest acceptable generalist), with full CPSAM available
for users who want maximum accuracy and can wait for the 588 MB download.

Caching footprint: if a user runs all 9 models, ~9 × ~100 MB = ~900 MB total
in IndexedDB. Add a "clear cached models" affordance in jit-ui settings.

### 7.6 Decision points (revisit before starting Phase 2)

1. **Does Phase 1 generalize well enough?** If users find full CPSAM works for
   every domain they care about, the per-domain finetunes are wasted effort.
   Collect user feedback after Phase 1 ships before committing to §7.4.
2. **Compute availability.** Without ≥1 A100-class GPU for ~6 weeks, Phase 2
   stalls. Confirm before kickoff.
3. **TissueNet / LiveCell access and licensing.** Datasets gated or
   non-commercial — confirm acceptable use up front, not mid-training.
4. **SlimCPSAM quality gate (§7.3 step 3).** If we can't hit "within 3 IoU
   points of full CPSAM," abandon. Don't ship a strictly worse smaller model.

---

## 8. Sources

- [Cellpose-SAM: superhuman generalization for cellular segmentation (bioRxiv 2025.04.28)](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1)
- [cellpose.vit_sam module source](https://cellpose.readthedocs.io/en/latest/_modules/cellpose/vit_sam.html)
- [MouseLand/cellpose GitHub](https://github.com/MouseLand/cellpose)
- [mouseland/cellpose-sam weights on Hugging Face](https://huggingface.co/mouseland/cellpose-sam)
- [Xenova/slimsam-50-uniform (transformers.js ONNX export)](https://huggingface.co/Xenova/slimsam-50-uniform)
- [SlimSAM: 0.1% Data Makes Segment Anything Slim (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/45a7ca247462d9e465ee88c8a302ca70-Paper-Conference.pdf)
- [Cellpose post-processing algorithm overview](https://deepwiki.com/stegmaierj/Cellpose3D/6.2-cellpose-post-processing-algorithm)
- [Transformers.js (huggingface/transformers.js)](https://github.com/huggingface/transformers.js)
- [Omnipose (kcutler/omnipose) — bacterial segmentation fork of Cellpose](https://github.com/kevinjohncutler/omnipose)
- [TissueNet dataset (vanvalenlab)](https://datasets.deepcell.org/data)
- [LiveCell dataset (Sartorius)](https://sartorius-research.github.io/LIVECell/) — CC-BY-NC 4.0 (non-commercial)

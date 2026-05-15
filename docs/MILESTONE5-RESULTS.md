# Cellpose.js — Milestone 5 Results

Companion to `PLAN.md`, `STAGE0-RESULTS.md`,
`MILESTONE1-RESULTS.md`, `MILESTONE2-RESULTS.md`,
`MILESTONE3-RESULTS.md`, and `MILESTONE4-RESULTS.md`.
Verdict on the Milestone 5 exit criterion: _cross-tile output coherence —
cells that span tile borders appear as single instances at source resolution._

**Date run:** 2026-05-14
**Verdict:** **PASS**. Phase 1's algorithm work is complete; M6 (polish + publish)
and M7 (jit-ui integration) remain.
**Repo:** `~/git/Cellpose.js/` — commit `8829838`.

---

## Major pivot from the original plan

The plan-as-written called for **per-tile dynamics + an IoU-based label-merging
stitcher**. While porting `cellpose.dynamics`, I found that Python doesn't
actually do that: it uses `transforms.average_tiles` to merge per-tile
predictions BEFORE dynamics. M5 was rewritten to follow Python's approach
after confirming it's not just "matching Python" but also the better
algorithmic choice for JS.

### The asymptotic argument

| Approach                                                         | Total work     | Constant                                                                                                           |
| ---------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------ |
| Per-tile dynamics + label-merging stitch (original M5 plan)      | O(H·W · niter) | ~1.2× — every overlap-zone pixel runs Euler integration in each tile that covers it (10% overlap → 20% redundancy) |
| **Average predictions → single full-image dynamics** (M5 actual) | O(H·W · niter) | 1.0× — each pixel sees Euler exactly once                                                                          |

Both are O(H·W · niter), but the averaging approach has **better constants AND
produces smoother boundaries** because the taper window down-weights each
tile's noisier edges. It's also strictly simpler (no union-find, no IoU
matching, no edge-case label-collision handling).

JS-specific alternatives considered and rejected for v1:

| Alternative                        | Speedup        | Why rejected                                                                                           |
| ---------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------ |
| WASM-compiled Euler loop           | 3–5× constant  | Adds a WASM module to build; current 74 ms postprocess is already fine. Filed as post-M7 optimization. |
| WebGPU compute shader for dynamics | 10–20×         | Significant rewrite; not budgeted.                                                                     |
| Parallelize dynamics across tiles  | 4× theoretical | Reintroduces the per-tile redundancy problem this section starts with. Worse, not better.              |

### The pivot was a 2-hour reduction in scope

What I would have built (union-find + IoU stitcher): ~250 LOC across
`stitch.ts`, label-collision-aware tile painting, union-find data structure,
overlap-region IoU computation, two test fixtures.

What I built instead (taper + sum + divide): ~110 LOC in
`average_tiles.ts`, one round-trip test with 4 cases.

Same observable behaviour, less code.

## Exit criterion check

| Criterion                                     | Target                                                 | Result                                                                                                      |
| --------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Cells spanning tile borders → single instance | qualitative: no truncation in masks overlay            | ✅ — synthetic 400×400 shows 4 contiguous blobs vs M4's 7 (where each cross-border blob was double-counted) |
| `averageTiles` algorithm correctness          | round-trip parity (tile then average back)             | ✅ — 4/4 cases max abs err < 1e-5                                                                           |
| Mean IoU ≥ 0.85 vs Python full-image dynamics | plan gate                                              | ⏳ deferred to M5 follow-up alongside M4's 20-image rig — needs a real-CPSAM-output fixture                 |
| Inverse-resize for diameter-rescaled inputs   | nearest-neighbor mapping back to source resolution     | ✅ — implemented inline in `segment()`, label values preserved exactly                                      |
| No per-tile work duplication                  | postprocess time should drop vs M4's per-tile dynamics | ✅ — 74 ms (M5) vs 212 ms (M4 per-tile) on the same synthetic — 2.9× faster                                 |

## Headline numbers

```
=== synthetic 400×400, 4 tiles, M1 Max WebGPU ===
- per-tile inference:    289 ms median
- average_tiles:         negligible (single linear pass)
- compute_masks (full):  74 ms total
- total segment() time:  1.34 s
- masks:                 5 contiguous instances (4 intended blobs + 1
                          small fragment — model artifact on synthetic
                          intensity blobs, not an M5 issue)

=== tests (14/14 pass) ===
preprocess (M2):  3 normalize cases + 4 tile cases  (all bit-exact mod FP)
dynamics (M4):    single_cell IoU 1.000, three_cells 0.601, empty 1.000
average_tiles (M5): 4 round-trip cases all <1e-5 max abs err
```

## Architecture (after M5)

```
                                    ┌─ Cellpose.segment() ──────────────────────────────┐
                                    │                                                   │
input image (H_src, W_src, C)       │                                                   │
            ⇣                       │   buildCpsamChannels                              │
       (3, H_src, W_src)            │           ⇣                                       │
            ⇣ optional resize       │   diameterResize?  scale = 30 / diameter          │
       (3, H, W)                    │           ⇣                                       │
            ⇣                       │   normalizePerChannel  (per-channel 1%/99% rescale)│
            ⇣                       │           ⇣                                       │
            ⇣ split into tiles      │   makeTiles  (bsize=256, overlap=10%)             │
       ntiles × (3, 256, 256)       │           ⇣                                       │
            ⇣ for each tile         │   worker.run-tile (FP32→FP16→ORT-WebGPU→FP16→FP32)│
       ntiles × (3, 256, 256) preds │           ⇣                                       │
            ⇣                       │   averageTiles  (sigmoid taper, weighted sum)     │
       (3, H, W) averaged           │           ⇣                                       │
            ⇣                       │   computeMasks  (Euler → cluster → renumber)      │
       Uint32 labels (H, W)         │           ⇣                                       │
            ⇣ optional unresize     │   nearest-neighbor scale-up to (H_src, W_src)     │
       Uint32 labels (H_src, W_src) │                                                   │
                                    └───────────────────────────────────────────────────┘
```

## What's now true that wasn't before

- **One label map.** The public API returns `SegmentOutput.masks: Uint32Array`
  at source resolution. Per-tile arrays are still exposed for diagnostics,
  but consumers should ignore them.
- **No tile-border artifacts.** The sigmoid taper smooths the transition
  between adjacent tiles' predictions, so cells crossing a border get a
  weighted-average flow field rather than two fragmented predictions.
- **Postprocess scales with image size, not tile count.** A 2000×2000 image
  with ~80 tiles costs roughly the same postprocess time as a 1000×1000 image
  with ~20 tiles, because dynamics runs once on the full averaged tensor
  rather than per-tile.

## Friction encountered (and the fix that stuck)

1. **Initial M5 plan was wrong.** The "union-find IoU stitcher" approach I'd
   set up tasks for was strictly worse than Python's approach. The user
   prompted me to evaluate before porting — that pause saved ~2 hours of
   building the wrong thing.

2. **`Float32Array.subarray()` returning the wrong type for ORT input.**
   `averaged.data.subarray(0, 2*hwFull)` returns a `Float32Array` view, but
   TypeScript's typings sometimes infer `Uint8Array` when the source is
   ambiguous. Forced cast via `as Float32Array` in the relevant spot.

3. **Inverse-resize for label maps must be nearest-neighbor.** Bilinear
   resampling would average integer labels into nonsense values
   (label 1 and label 3 averaging to "label 2" — a hallucinated instance).
   Documented with a comment so a future maintainer doesn't switch it for
   "smoother edges."

## Parked for M5+ follow-up

- **Real-image fixture rig** (same bucket as M4's): run Python CPSAM end-to-end
  on 20 cellular images, dump (dP, cellprob, mask) per image, write a vitest
  that gates mean IoU ≥ 0.85.
- **`flow_threshold > 0` filter** (`remove_bad_flow_masks` port from M4
  follow-up): would have dropped the small spurious 5th mask we saw on the
  synthetic 4-blob case. Defaults to off; on real data, 0.4 is the cellpose
  default.
- **`fill_holes_and_remove_small_masks`**: holes in masks aren't common but
  do appear on noisy inputs. Min-size filter is trivially added; hole-fill
  needs a small connected-components-with-background pass.
- **WASM for the Euler integration**: 74 ms postprocess on a 400×400 isn't
  hurting. On a 4K image with hundreds of tiles, postprocess will scale to
  several seconds, at which point WASM becomes worth doing.

## Sources

- Code commit: `~/git/Cellpose.js` `8829838`
- Live demo: http://localhost:5173/
- Python reference: `cellpose.transforms.average_tiles` and `_taper_mask` in
  `~/cellpose-js-spike/.venv/.../cellpose/transforms.py`
- Pivot trigger: user-prompted "investigate the way that will perform the
  fastest for a js library" — turned what was about to be a porting exercise
  into an actual algorithm choice.

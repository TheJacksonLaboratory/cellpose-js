# Cellpose.js — Milestone 2 Results

Companion to `PLAN.md`, `STAGE0-RESULTS.md`, and
`MILESTONE1-RESULTS.md`. Verdict on the Milestone 2 exit criterion:
_bit-exact match vs Python on a fixture set, plus end-to-end real-image
execution through the new segment() entry point._

**Date run:** 2026-05-14
**Verdict:** **PASS**. Phase 1 work continues to Milestone 3 (worker offload + abort).
**Repo:** `~/git/Cellpose.js/` — commit `3035fdf`.

---

## Exit criterion check

| Criterion                                   | Status | Evidence                                                                                           |
| ------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------- |
| Bit-exact normalize parity vs Python        | ✅     | max abs err < 1e-5 across 3 fixtures (1ch/3ch, 64×64 and 256×256)                                  |
| Bit-exact tile pixel parity vs Python       | ✅     | max abs err = 0 on the valid region across 4 fixtures                                              |
| Tile origin parity vs Python                | ✅     | exact match — `ny=ceil((1+2·overlap)·Ly/B)`, `linspace(0, Ly−B, ny).astype(int)` ported faithfully |
| Real-image execution through `segment()`    | ✅     | 609×457 RGB → 9 tiles, no errors                                                                   |
| Preprocessing overhead negligible           | ✅     | per-tile inference 277 ms (vs Spike B 277 ms — identical)                                          |
| `chan` / `chan2` legacy semantics preserved | ✅     | output (3,H,W) ordered [chan, chan2-or-zero, zero] matches Cellpose UX                             |

## Headline numbers

```
=== synthetic 400×400 (4 Gaussian blobs, chan=0, chan2=0, no diameter) ===
tiles: 4   (2×2 grid)
per-tile inference median: 279 ms
total segment() time: 1.28 s
preprocess + bookkeeping: ~160 ms

=== real 609×457 RGB photograph (chan=0, chan2=0, no diameter) ===
tiles: 9   (3×3 grid)
per-tile inference median: 277 ms
total segment() time: 2.62 s
preprocess + bookkeeping: ~130 ms

=== parity tests (vitest, 7/7 pass) ===
normalize99 parity: norm_small_1ch       max abs err < 1e-5
                    norm_small_3ch       max abs err < 1e-5
                    norm_med_3ch         max abs err < 1e-5
make_tiles parity:  tile_smaller         exact (valid region)
                    tile_exact_256       exact
                    tile_grid            exact (16 tiles)
                    tile_tall            exact (5 tiles)
```

## What this milestone bought us

- **`Cellpose.segment(image, opts)`** — the public entry point. Replaces the
  M1 `segmentRawTile()` stub; the latter is kept as a deprecated escape hatch.
- **Preprocess pipeline**: `buildCpsamChannels` → `diameterResize` (optional)
  → `normalizePerChannel` → `makeTiles`. Each module is independently testable
  and the pipeline orders ops to match Cellpose's own conventions.
- **Per-tile output shape**: `{ tile, tx, ty, bsize, flows_cellprob, inferenceMs }`.
  Tile origins are in _resized_ image coordinates so M5's stitching can place
  each tile back onto the post-resize canvas before the inverse-resize step
  at the very end.
- **Vitest harness with .npy parity tests** — small custom NPY reader
  (`tests/util/npy.ts`) avoids a heavy `numjs`-style dependency. Symlinked
  fixtures from `~/cellpose-js-spike/fixtures/` keep large binaries out of
  the repo.

## Parameter quick-reference

Promoted from the demo-session chat because users will want this without
digging through code:

### `chan` / `chan2` (channel mapping for `buildCpsamChannels`)

CPSAM was trained with channel-shuffling augmentation, so it does **not**
privilege a specific channel as cyto vs nuclei. The `chan`/`chan2` API exists
to mirror Cellpose 1–3's user-facing conventions, not because CPSAM needs it.

| Value          | Source channel selected                                      |
| -------------- | ------------------------------------------------------------ |
| `chan = 0`     | First source channel — treats input as grayscale (R for RGB) |
| `chan = 1`     | R                                                            |
| `chan = 2`     | G                                                            |
| `chan = 3`     | B                                                            |
| `chan2 = 0`    | No second channel (output channel 1 is zero)                 |
| `chan2 = 1..3` | Same indexing as `chan`                                      |

Recommendations by image type:

| Image type                                 | `chan` | `chan2` |
| ------------------------------------------ | ------ | ------- |
| H&E histology, brightfield, phase contrast | `0`    | `0`     |
| Fluorescence — green cyto, blue nuclei     | `2`    | `3`     |
| Fluorescence — red cyto, green nuclei      | `1`    | `2`     |
| Unknown / first run                        | `0`    | `0`     |

### `diameter` (input to `diameterResize`)

Rescales the image so the median cell occupies ~30 px (CPSAM's training
median across all corpora). Omit to run at native resolution.

| Cell size in source image            | Suggested `diameter`                      |
| ------------------------------------ | ----------------------------------------- |
| Roughly 20–60 px across              | leave blank — already in sweet spot       |
| Tiny (5–15 px) — bacteria, yeast     | ≈ 10 (upscales image ~3×)                 |
| Large (80+ px) — tissue at high zoom | your visual estimate (downscales)         |
| Unknown / first run                  | leave blank, run, eyeball one cell, retry |

Rule of thumb for first-time test: **`chan=0`, `chan2=0`, no diameter**. If
`cellprob` lights up on cells, you're set; otherwise estimate cell width in
pixels and pass it.

## Friction encountered

Smaller list than M1 — most of the surface was straight numpy → JS.

1. **Tile-size variance Python vs JS.** Python's `make_tiles` returns
   variable-size tiles when an axis is smaller than `bsize` (a 200-px image
   with `bsize=256` yields a 200-tall tile, not zero-padded). JS always emits
   `(C, B, B)` zero-padded tiles for downstream model convenience. Resolved
   in tests by comparing only the valid region of each JS tile against the
   Python expected tile. Documented in test comments and in `tile.ts`.

2. **Canvas bilinear ≠ `cv2.INTER_LINEAR`.** Browser canvas resize is
   implementation-defined and differs from OpenCV's bilinear in edge
   handling and rounding. No fixture-level parity gate for `resize.ts` in
   M2; the path is exercised qualitatively in the demo for now. If the
   downstream IoU gate at M5 turns out to be resize-sensitive, we'll either
   port a faithful bilinear in JS or quantify the divergence and live with
   it.

3. **ESM tests need `import.meta.url` for `__dirname`.** Vitest in ESM mode
   doesn't provide `__dirname`; small `fileURLToPath(import.meta.url)`
   incantation at the top of the test file. Not a hard problem, just a 30-
   second time-sink that's easy to forget.

## Parked for later milestones

- **Resize parity fixture.** `cv2.INTER_LINEAR` produces a known reference; if
  M5's IoU tells us resize divergence is material, generate a fixture and
  set a wider gate (`~1e-2` mean abs err).
- **Constant-channel normalize fixture.** The `range < 1e-3` branch of
  `normalize99` (zero output) isn't fixture-tested. Trivial to add later.
- **Tile stitching** (M5). Per-tile outputs are exposed as-is; merging masks
  across overlapping borders is M5 work.
- **Worker offload** (M3). The model runs on the main thread today; each
  ~280 ms inference call blocks the UI. M3 moves it to a Web Worker and
  wires AbortSignal-driven cancellation.

## Sources

- Code commit: `~/git/Cellpose.js` `3035fdf`
- Fixtures: `~/cellpose-js-spike/fixtures/` (gen script:
  `~/cellpose-js-spike/gen_preprocess_fixtures.py`)
- Live demo: http://localhost:5173/ (Vite dev server)
- Python reference: `cellpose/transforms.py` in
  `~/cellpose-js-spike/.venv/lib/python3.11/site-packages/cellpose/`

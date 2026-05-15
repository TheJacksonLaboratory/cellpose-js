/**
 * Cluster Euler-integrated convergence points into instance masks. Faithful
 * port of cellpose.dynamics.get_masks_torch (2D).
 *
 * Algorithm:
 *   1. Apply rpad=20 padding and clamp converged points into the padded
 *      histogram grid of shape (H+40, W+40).
 *   2. Build the 2D histogram of converged points.
 *   3. Find seeds: pixels where h(p) >= 5x5 max-pool of h(p) AND h(p) > 10.
 *   4. Sort seeds by histogram count (ascending — matches Python's argsort).
 *   5. For each seed, take an 11x11 window of the histogram centred on the
 *      seed. Initialise a 11x11 binary "seed_mask" with a 1 at (5, 5).
 *      Run 5 iterations of:  seed_mask = max_pool(seed_mask, 3) & (h_window > 2)
 *      The result identifies all histogram-space pixels assigned to this seed.
 *   6. Paint each seed's footprint in a histogram-shape label map M1
 *      (later seeds may overwrite earlier ones — matches Python).
 *   7. For each integrated point p[i], its label is M1[p[i]].
 *   8. Map back to image-space: M0[(seedY[i], seedX[i])] = M1[p[i]].
 *   9. Discard masks larger than max_size_fraction * total_pixels.
 *  10. Renumber labels to 1..K with no gaps.
 */

export interface GetMasksResult {
  /** Uint32 instance label map, length = H * W, row-major. 0 = background. */
  masks: Uint32Array;
  /** Final label count K. */
  count: number;
}

const RPAD = 20;
const PEAK_HIST_THRESHOLD = 10; // h(p) > 10 to be a seed
const GROW_HIST_THRESHOLD = 2; // grow into pixels where h > 2
const GROW_ITERS = 5;
const HALF_WIN = 5; // 11x11 window = ±5

export function getMasks(
  pFinal: Int32Array,
  seedY: Int32Array,
  seedX: Int32Array,
  H: number,
  W: number,
  maxSizeFraction = 0.4,
): GetMasksResult {
  const n = seedY.length;
  const total = H * W;
  const masks = new Uint32Array(total);
  if (n === 0) return { masks, count: 0 };

  // 1+2. Build padded histogram (Hp = H + 2*rpad, Wp = W + 2*rpad).
  const Hp = H + 2 * RPAD;
  const Wp = W + 2 * RPAD;
  const hist = new Int32Array(Hp * Wp);
  // Padded integer convergence coords per seed
  const py = new Int32Array(n);
  const px = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    // Clamp + pad (matches Python: pt += rpad; clamp min=0; clamp max=shape+rpad-1)
    let y = (pFinal[2 * i] as number) + RPAD;
    let x = (pFinal[2 * i + 1] as number) + RPAD;
    if (y < 0) y = 0;
    else if (y > H + RPAD - 1) y = H + RPAD - 1;
    if (x < 0) x = 0;
    else if (x > W + RPAD - 1) x = W + RPAD - 1;
    py[i] = y;
    px[i] = x;
    hist[y * Wp + x]++;
  }

  // 3. Find seeds via 5x5 max-pool comparison.
  // Local maxima where h(p) >= max5x5(h)(p) - eps and h(p) > 10. Matches Python.
  // Use a single-pass approach: for each candidate pixel, check the 5x5 neighborhood.
  type Seed = { y: number; x: number; count: number };
  const seedList: Seed[] = [];
  for (let y = 2; y < Hp - 2; y++) {
    for (let x = 2; x < Wp - 2; x++) {
      const h = hist[y * Wp + x] as number;
      if (h <= PEAK_HIST_THRESHOLD) continue;
      let isMax = true;
      for (let dy = -2; dy <= 2 && isMax; dy++) {
        const yOff = (y + dy) * Wp;
        for (let dx = -2; dx <= 2; dx++) {
          if ((hist[yOff + (x + dx)] as number) > h) {
            isMax = false;
            break;
          }
        }
      }
      if (isMax) seedList.push({ y, x, count: h });
    }
  }
  if (seedList.length === 0) return { masks, count: 0 };

  // 4. Sort by histogram count ascending — matches Python's argsort.
  seedList.sort((a, b) => a.count - b.count);

  // 5. Region-grow each seed in histogram space.
  // 6. Paint into a histogram-shape label map. Use a flat Uint32Array of size Hp*Wp.
  const M1 = new Uint32Array(Hp * Wp);
  const WIN = 2 * HALF_WIN + 1; // 11
  const seedMask = new Uint8Array(WIN * WIN);
  const next = new Uint8Array(WIN * WIN);

  for (let k = 0; k < seedList.length; k++) {
    const s = seedList[k] as Seed;
    const label = k + 1;
    seedMask.fill(0);
    seedMask[HALF_WIN * WIN + HALF_WIN] = 1;

    for (let it = 0; it < GROW_ITERS; it++) {
      next.fill(0);
      // 3x3 dilate
      for (let yy = 0; yy < WIN; yy++) {
        for (let xx = 0; xx < WIN; xx++) {
          let mx = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const ny = yy + dy;
            if (ny < 0 || ny >= WIN) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const nx = xx + dx;
              if (nx < 0 || nx >= WIN) continue;
              if ((seedMask[ny * WIN + nx] as number) > mx) mx = seedMask[ny * WIN + nx] as number;
            }
          }
          next[yy * WIN + xx] = mx;
        }
      }
      // AND with h_window > GROW_HIST_THRESHOLD
      for (let yy = 0; yy < WIN; yy++) {
        const histY = s.y - HALF_WIN + yy;
        if (histY < 0 || histY >= Hp) {
          for (let xx = 0; xx < WIN; xx++) next[yy * WIN + xx] = 0;
          continue;
        }
        for (let xx = 0; xx < WIN; xx++) {
          const histX = s.x - HALF_WIN + xx;
          if (histX < 0 || histX >= Wp) {
            next[yy * WIN + xx] = 0;
            continue;
          }
          if ((hist[histY * Wp + histX] as number) <= GROW_HIST_THRESHOLD) next[yy * WIN + xx] = 0;
        }
      }
      seedMask.set(next);
    }

    // Paint seedMask positions into M1 (in histogram coordinates).
    for (let yy = 0; yy < WIN; yy++) {
      const histY = s.y - HALF_WIN + yy;
      if (histY < 0 || histY >= Hp) continue;
      for (let xx = 0; xx < WIN; xx++) {
        const histX = s.x - HALF_WIN + xx;
        if (histX < 0 || histX >= Wp) continue;
        if (seedMask[yy * WIN + xx]) {
          M1[histY * Wp + histX] = label;
        }
      }
    }
  }

  // 7+8. Each seed pixel inherits the label at its convergence point.
  for (let i = 0; i < n; i++) {
    const label = M1[(py[i] as number) * Wp + (px[i] as number)] as number;
    if (label > 0) {
      masks[(seedY[i] as number) * W + (seedX[i] as number)] = label;
    }
  }

  // 9. Remove oversized masks.
  const limit = Math.floor(total * maxSizeFraction);
  const counts = new Map<number, number>();
  for (let i = 0; i < total; i++) {
    const v = masks[i] as number;
    if (v > 0) counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const toDrop = new Set<number>();
  for (const [label, c] of counts) {
    if (c > limit) toDrop.add(label);
  }
  if (toDrop.size > 0) {
    for (let i = 0; i < total; i++) {
      if (toDrop.has(masks[i] as number)) masks[i] = 0;
    }
  }

  // 10. Renumber remaining labels to 1..K (no gaps).
  const remap = new Map<number, number>();
  let nextLabel = 1;
  for (let i = 0; i < total; i++) {
    const v = masks[i] as number;
    if (v === 0) continue;
    let m = remap.get(v);
    if (m === undefined) {
      m = nextLabel++;
      remap.set(v, m);
    }
    masks[i] = m;
  }
  return { masks, count: nextLabel - 1 };
}

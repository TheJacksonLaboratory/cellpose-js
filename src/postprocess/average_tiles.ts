/**
 * Average per-tile model outputs back into a single full-image flow+cellprob
 * tensor. Faithful port of cellpose.transforms.average_tiles + _taper_mask.
 *
 * Why this is the right algorithm for JS too (not just because Python does it):
 *
 *   Per-tile dynamics + label stitching is asymptotically O(H·W · niter · 1.2)
 *   because every overlap-zone pixel runs through Euler integration in each
 *   tile that covers it (10% overlap ≈ 20% redundancy). Averaging the
 *   predictions first and running dynamics once on the full image is
 *   O(H·W · niter · 1.0) — same asymptotic class, better constants, and
 *   produces smoother boundaries because the taper window down-weights the
 *   noisier edges of each tile's prediction.
 *
 *   See CELLPOSE-JS-MILESTONE5-RESULTS.md (when written) for the full analysis.
 *
 * Algorithm:
 *   1. Build a (B, B) taper mask: 2D outer product of a 1D sigmoid window
 *      that's ≈ 1 in the center and ≈ 0 in the last ~20 pixels of each edge.
 *   2. For each tile, accumulate (prediction × mask) into a full-image yf,
 *      and accumulate mask into a per-pixel weight Navg.
 *   3. yf /= Navg.
 */

const TAPER_SIGMA = 7.5;
const TAPER_EDGE_INSET = 20;

const _taperCache = new Map<number, Float32Array>();

/** Build a (B, B) taper mask matching Python's _taper_mask. Cached per B. */
export function taperMask(B: number): Float32Array {
  const cached = _taperCache.get(B);
  if (cached) return cached;
  const bsize = Math.max(224, B);
  const m1d = new Float32Array(bsize);
  const mean = (bsize - 1) / 2;
  const inflection = bsize / 2 - TAPER_EDGE_INSET;
  for (let x = 0; x < bsize; x++) {
    m1d[x] = 1 / (1 + Math.exp((Math.abs(x - mean) - inflection) / TAPER_SIGMA));
  }
  const lo = (bsize - B) >> 1;
  const out = new Float32Array(B * B);
  for (let y = 0; y < B; y++) {
    const my = m1d[lo + y] as number;
    const rowOff = y * B;
    for (let x = 0; x < B; x++) {
      out[rowOff + x] = my * (m1d[lo + x] as number);
    }
  }
  _taperCache.set(B, out);
  return out;
}

export interface TileInputForAveraging {
  /** (3, B, B) row-major: dy, dx, cellprob */
  flowsCellprob: Float32Array;
  tx: number;
  ty: number;
  bsize: number;
}

export interface AveragedTensor {
  /** (3, H, W) row-major. */
  data: Float32Array;
  H: number;
  W: number;
}

export function averageTiles(tiles: TileInputForAveraging[], H: number, W: number): AveragedTensor {
  if (tiles.length === 0) {
    return { data: new Float32Array(3 * H * W), H, W };
  }
  const fullHW = H * W;
  const yf = new Float32Array(3 * fullHW);
  const nav = new Float32Array(fullHW);

  for (const t of tiles) {
    const B = t.bsize;
    const mask = taperMask(B);
    const tileHW = B * B;
    const y0 = t.ty,
      x0 = t.tx;
    const yMax = Math.min(H, y0 + B);
    const xMax = Math.min(W, x0 + B);
    for (let dy = 0; dy < yMax - y0; dy++) {
      const tileRowOff = dy * B;
      const fullRow = (y0 + dy) * W;
      for (let dx = 0; dx < xMax - x0; dx++) {
        const w = mask[tileRowOff + dx] as number;
        const fullIdx = fullRow + (x0 + dx);
        yf[fullIdx] += (t.flowsCellprob[tileRowOff + dx] as number) * w;
        yf[fullHW + fullIdx] += (t.flowsCellprob[tileHW + tileRowOff + dx] as number) * w;
        yf[2 * fullHW + fullIdx] += (t.flowsCellprob[2 * tileHW + tileRowOff + dx] as number) * w;
        nav[fullIdx] += w;
      }
    }
  }

  for (let i = 0; i < fullHW; i++) {
    const n = nav[i] as number;
    if (n > 0) {
      const inv = 1 / n;
      yf[i] *= inv;
      yf[fullHW + i] *= inv;
      yf[2 * fullHW + i] *= inv;
    }
  }
  return { data: yf, H, W };
}

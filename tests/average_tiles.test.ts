/**
 * Round-trip parity for averageTiles.
 *
 * If we tile an image, then average_tiles back, the reconstruction should match
 * the original within FP rounding (the taper window weights divide out cleanly
 * since the same weight on numerator and denominator cancels).
 *
 * No Python fixture needed — this is testing the algebra, not Python parity.
 * Python parity for the full M5 pipeline (M5 gate IoU ≥ 0.85) needs a real
 * CPSAM-output fixture and is filed as M5 follow-up alongside the M4 IoU rig.
 */
import { describe, it, expect } from 'vitest';
import { averageTiles, type TileInputForAveraging } from '../src/postprocess/average_tiles.js';
import { makeTiles } from '../src/preprocess/tile.js';

function synthesizeImage(C: number, H: number, W: number, seed = 0): Float32Array {
  let s = seed;
  const out = new Float32Array(C * H * W);
  for (let i = 0; i < out.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 4294967296) * 2 - 1;
  }
  return out;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs((a[i] as number) - (b[i] as number));
    if (d > m) m = d;
  }
  return m;
}

describe('averageTiles round-trip', () => {
  const cases: Array<[string, number, number]> = [
    ['256x256_singletile', 256, 256],
    ['400x400_2x2_grid', 400, 400],
    ['609x457_3x3_grid', 609, 457],
    ['1024x512_wide', 1024, 512],
  ];
  for (const [name, H, W] of cases) {
    it(name, () => {
      const img = synthesizeImage(3, H, W, 42);
      const tiles = makeTiles(img, W, H, 3, { bsize: 256, overlap: 0.1 });
      const tileInputs: TileInputForAveraging[] = tiles.map((t) => ({
        flowsCellprob: t.tile,
        tx: t.tx,
        ty: t.ty,
        bsize: t.bsize,
      }));
      const avg = averageTiles(tileInputs, H, W);
      // For images < bsize on an axis, makeTiles pads with zeros. averageTiles
      // ignores out-of-bounds pixels so the pad zeros don't pollute. Compare
      // only the in-bounds (H, W) region.
      const err = maxAbsDiff(img, avg.data);
      // Reconstruction is exact modulo FP rounding because the taper weights
      // cancel: yf/N = (sum w·x) / (sum w) = x when x is uniform across the
      // window. The pad-zero regions in tiles smaller than bsize are
      // never accumulated (out-of-bounds skip), so they don't pollute.
      expect(err).toBeLessThan(1e-5);
    });
  }
});

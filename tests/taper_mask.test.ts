/**
 * Unit tests for taperMask (postprocess/average_tiles.ts).
 *
 * The taper mask is a 2D outer-product of a sigmoid window. It should be:
 *   - Symmetric around the center on each axis.
 *   - Maximal at the center (where the sigmoid saturates at ~1).
 *   - Close to zero in the 20-pixel border (sigmoid inflection point).
 *   - Cached per B so repeated calls return the same array.
 *
 * Matches cellpose.transforms._taper_mask formulation:
 *   bsize = max(224, B);
 *   xm = abs(arange(bsize) - mean)
 *   m1d = 1 / (1 + exp((xm - (bsize/2 - 20)) / 7.5))
 *   mask = outer(m1d, m1d)
 *   then cropped to (B, B).
 */
import { describe, it, expect } from 'vitest';
import { taperMask } from '../src/postprocess/average_tiles.js';

describe('taperMask', () => {
  it('returns the same array on repeat calls (caching)', () => {
    const a = taperMask(256);
    const b = taperMask(256);
    expect(a).toBe(b);
  });

  it('has correct shape (B*B floats)', () => {
    expect(taperMask(256).length).toBe(256 * 256);
    expect(taperMask(128).length).toBe(128 * 128);
  });

  it('peaks near the center (~1.0)', () => {
    const B = 256;
    const m = taperMask(B);
    const center = m[(B / 2) * B + B / 2] as number;
    // Sigmoid saturates near 1 in the center.
    expect(center).toBeGreaterThan(0.99);
    expect(center).toBeLessThanOrEqual(1.0);
  });

  it('is small in the 20-px border (well under 0.5)', () => {
    const B = 256;
    const m = taperMask(B);
    // The sigmoid inflection is at bsize/2 - 20 ≈ 108 (for B=256, since
    // bsize = max(224, 256) = 256). So at the very corner (0, 0) we're
    // 128 px from center → 20 px past inflection → sigmoid is ~exp(-20/7.5)
    // ≈ 0.07, but the OUTER product squares this, so ~0.005.
    const corner = m[0] as number;
    expect(corner).toBeLessThan(0.05);
    expect(corner).toBeGreaterThan(0);
  });

  it('is symmetric on both axes', () => {
    const B = 64;
    const m = taperMask(B);
    for (let y = 0; y < B; y++) {
      for (let x = 0; x < B; x++) {
        const a = m[y * B + x] as number;
        const b = m[y * B + (B - 1 - x)] as number;
        expect(a).toBeCloseTo(b, 6);
      }
    }
    for (let y = 0; y < B; y++) {
      for (let x = 0; x < B; x++) {
        const a = m[y * B + x] as number;
        const b = m[(B - 1 - y) * B + x] as number;
        expect(a).toBeCloseTo(b, 6);
      }
    }
  });

  it('matches Python formula at a handful of points (B=256)', () => {
    const B = 256;
    const m = taperMask(B);
    const bsize = Math.max(224, B); // 256
    const mean = (bsize - 1) / 2;
    const inflection = bsize / 2 - 20;
    const sigma = 7.5;
    // Cropping offset: lo = (bsize - B) >> 1 → 0 when bsize == B.
    const lo = (bsize - B) >> 1;
    const sigmoid = (i: number): number => {
      const x = lo + i;
      return 1 / (1 + Math.exp((Math.abs(x - mean) - inflection) / sigma));
    };
    // Sample five (y, x) positions.
    const samples: Array<[number, number]> = [
      [0, 0],
      [10, 10],
      [50, 200],
      [128, 128],
      [200, 50],
    ];
    for (const [y, x] of samples) {
      const expected = sigmoid(y) * sigmoid(x);
      const got = m[y * B + x] as number;
      expect(got).toBeCloseTo(expected, 6);
    }
  });
});

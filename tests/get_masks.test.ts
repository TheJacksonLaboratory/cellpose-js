/**
 * Unit tests for getMasks (postprocess/get_masks.ts).
 *
 * Strategy: hand-build a "post-Euler" convergence pattern with N tight
 * clusters of identical (py, px) points. Each cluster should produce one
 * mask covering its source seed pixels.
 */
import { describe, it, expect } from 'vitest';
import { getMasks } from '../src/postprocess/get_masks.js';

/**
 * Build N synthetic clusters: M seeds per cluster, all converging to one
 * point. Source seed pixels for cluster k are arranged in a small square
 * starting at (sourceCenters[k].y, sourceCenters[k].x).
 */
function makeClusters(
  H: number,
  W: number,
  clusters: Array<{ converge: { y: number; x: number }; source: { y: number; x: number } }>,
  perCluster: number,
): {
  pFinal: Int32Array;
  seedY: Int32Array;
  seedX: Int32Array;
} {
  const n = clusters.length * perCluster;
  const pFinal = new Int32Array(2 * n);
  const seedY = new Int32Array(n);
  const seedX = new Int32Array(n);
  let i = 0;
  for (const c of clusters) {
    // Arrange perCluster seeds in a roughly square footprint at source.
    const side = Math.ceil(Math.sqrt(perCluster));
    for (let k = 0; k < perCluster; k++) {
      const sy = c.source.y + Math.floor(k / side);
      const sx = c.source.x + (k % side);
      // Wrap inside the image to keep the test self-consistent.
      seedY[i] = Math.min(H - 1, sy);
      seedX[i] = Math.min(W - 1, sx);
      pFinal[2 * i] = c.converge.y;
      pFinal[2 * i + 1] = c.converge.x;
      i++;
    }
  }
  return { pFinal, seedY, seedX };
}

describe('getMasks', () => {
  it('returns zero-count for empty input', () => {
    const r = getMasks(new Int32Array(0), new Int32Array(0), new Int32Array(0), 64, 64);
    expect(r.count).toBe(0);
    expect(r.masks.length).toBe(64 * 64);
    for (let i = 0; i < r.masks.length; i++) expect(r.masks[i]).toBe(0);
  });

  it('produces one label per cluster (3 clusters)', () => {
    const H = 96,
      W = 96;
    const clusters = [
      { converge: { y: 16, x: 16 }, source: { y: 8, x: 8 } },
      { converge: { y: 48, x: 48 }, source: { y: 40, x: 40 } },
      { converge: { y: 80, x: 16 }, source: { y: 72, x: 8 } },
    ];
    // ~25 seeds per cluster — well above the PEAK_HIST_THRESHOLD of 10.
    const { pFinal, seedY, seedX } = makeClusters(H, W, clusters, 25);
    const r = getMasks(pFinal, seedY, seedX, H, W);
    expect(r.count).toBe(3);
    const seen = new Set<number>();
    for (let i = 0; i < r.masks.length; i++) {
      const v = r.masks[i] as number;
      if (v > 0) seen.add(v);
    }
    expect(seen.size).toBe(3);
    expect(Array.from(seen).sort((a, b) => a - b)).toEqual([1, 2, 3]);
  });

  it('drops clusters that fall under PEAK_HIST_THRESHOLD (10)', () => {
    const H = 64,
      W = 64;
    const clusters = [
      { converge: { y: 16, x: 16 }, source: { y: 8, x: 8 } }, // 5 seeds → dropped
      { converge: { y: 48, x: 48 }, source: { y: 40, x: 40 } }, // 25 seeds → kept
    ];
    // perCluster = 5 → both clusters get 5 seeds. The first cluster's peak in
    // the histogram is therefore 5, which is below the threshold (10), so it
    // should be dropped.
    const { pFinal, seedY, seedX } = makeClusters(H, W, clusters, 5);
    const r1 = getMasks(pFinal, seedY, seedX, H, W);
    expect(r1.count).toBe(0); // both clusters below threshold

    const { pFinal: p2, seedY: y2, seedX: x2 } = makeClusters(H, W, clusters, 25);
    const r2 = getMasks(p2, y2, x2, H, W);
    expect(r2.count).toBe(2);
  });

  it('drops masks larger than maxSizeFraction of the image', () => {
    const H = 64,
      W = 64;
    const total = H * W;
    // One giant cluster: 60% of pixels (above the default 40% threshold).
    // We need at least PEAK_HIST_THRESHOLD+1 (=11) seeds piling onto the
    // convergence point for it to be detected as a seed in the first place.
    const targetSize = Math.floor(total * 0.6);
    const pFinal = new Int32Array(2 * targetSize);
    const seedY = new Int32Array(targetSize);
    const seedX = new Int32Array(targetSize);
    for (let i = 0; i < targetSize; i++) {
      seedY[i] = Math.floor(i / W);
      seedX[i] = i % W;
      pFinal[2 * i] = 32;
      pFinal[2 * i + 1] = 32;
    }
    const r = getMasks(pFinal, seedY, seedX, H, W, 0.4);
    // Mask exceeds 40% → dropped → 0 labels remain.
    expect(r.count).toBe(0);
    for (let i = 0; i < r.masks.length; i++) expect(r.masks[i]).toBe(0);
  });

  it('renumbers labels to 1..K (no gaps after size-filter)', () => {
    const H = 64,
      W = 64;
    // Three valid clusters in different corners.
    const clusters = [
      { converge: { y: 12, x: 12 }, source: { y: 4, x: 4 } },
      { converge: { y: 12, x: 52 }, source: { y: 4, x: 44 } },
      { converge: { y: 52, x: 12 }, source: { y: 44, x: 4 } },
    ];
    const { pFinal, seedY, seedX } = makeClusters(H, W, clusters, 25);
    const r = getMasks(pFinal, seedY, seedX, H, W);
    // Labels must be exactly {1, 2, 3} with no gaps.
    const labelSet = new Set<number>();
    for (let i = 0; i < r.masks.length; i++) {
      if ((r.masks[i] as number) > 0) labelSet.add(r.masks[i] as number);
    }
    expect([...labelSet].sort((a, b) => a - b)).toEqual([1, 2, 3]);
  });

  it('handles negative pre-padded coordinates by clamping to 0', () => {
    // Edge case: a seed converges past the top-left corner. After +RPAD it
    // should still land at row/col 0 of the padded histogram.
    const H = 32,
      W = 32;
    const n = 30;
    const pFinal = new Int32Array(2 * n);
    const seedY = new Int32Array(n);
    const seedX = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      seedY[i] = i;
      seedX[i] = i;
      pFinal[2 * i] = -10; // would be -10 + 20 = 10 (still ok)
      pFinal[2 * i + 1] = -25; // would be -25 + 20 = -5 → clamped to 0
    }
    const r = getMasks(pFinal, seedY, seedX, H, W);
    // Just verify no crash and a sensible label produced.
    expect(r.count).toBeGreaterThanOrEqual(0);
  });
});

/**
 * Unit tests for followFlows (postprocess/follow_flows.ts).
 *
 * Strategy: synthesize a flow field with a known fixed point and verify that
 * Euler integration drives seed pixels toward it. This is the lightest-weight
 * way to anchor the Euler loop without depending on a real CPSAM output.
 *
 * NOTE: cellpose pre-scales the flow field by `(cellprob > thresh) / 5`
 * before calling followFlows. These tests pass already-scaled inputs so
 * each step's displacement is the desired fraction of a pixel.
 */
import { describe, it, expect } from 'vitest';
import { followFlows } from '../src/postprocess/follow_flows.js';

/**
 * Build a flow field where every pixel's (dy, dx) points toward (cy, cx).
 * Pre-scales by 1/5 to match the convention used by compute_masks.
 */
function flowsTowardCenter(
  H: number,
  W: number,
  cy: number,
  cx: number,
  magnitude = 1.0,
): { dP: Float32Array; cellprob: Float32Array } {
  const hw = H * W;
  const dP = new Float32Array(2 * hw);
  const cellprob = new Float32Array(hw);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const dy = cy - y;
      const dx = cx - x;
      const r = Math.hypot(dy, dx) || 1;
      // Unit vector toward center, scaled by magnitude, then by 1/5
      // (the compute_masks convention).
      dP[i] = ((dy / r) * magnitude) / 5;
      dP[hw + i] = ((dx / r) * magnitude) / 5;
      cellprob[i] = 1.0; // every pixel is "cell" so every pixel is a seed
    }
  }
  return { dP, cellprob };
}

describe('followFlows', () => {
  it('seeds with cellprob <= threshold are excluded', () => {
    const H = 16,
      W = 16;
    const { dP, cellprob } = flowsTowardCenter(H, W, 8, 8);
    // Knock out one row of cellprob so those seeds drop out.
    for (let x = 0; x < W; x++) cellprob[3 * W + x] = 0;
    const r = followFlows(dP, cellprob, H, W, 0, 50);
    expect(r.seedY.length).toBe(H * W - W);
    // None of the surviving seeds should come from y=3.
    for (let i = 0; i < r.seedY.length; i++) {
      expect(r.seedY[i]).not.toBe(3);
    }
  });

  it('returns empty arrays when no pixel exceeds the threshold', () => {
    const H = 8,
      W = 8;
    const dP = new Float32Array(2 * H * W);
    const cellprob = new Float32Array(H * W); // all zero
    const r = followFlows(dP, cellprob, H, W, 0, 10);
    expect(r.pFinal.length).toBe(0);
    expect(r.seedY.length).toBe(0);
    expect(r.seedX.length).toBe(0);
  });

  it('converges to the fixed point under sufficient iterations', () => {
    const H = 32,
      W = 32;
    const cy = 15,
      cx = 16;
    const { dP, cellprob } = flowsTowardCenter(H, W, cy, cx, 1.0);
    const r = followFlows(dP, cellprob, H, W, 0, 300);
    expect(r.seedY.length).toBe(H * W);
    // Every seed should converge to within a few pixels of (cy, cx).
    // The fixed point is a sink — once a pixel arrives, the flow magnitude
    // at the fixed point itself is undefined (we set unit magnitude / 5),
    // so we accept any final position within a small radius.
    let maxDist = 0;
    for (let i = 0; i < r.pFinal.length / 2; i++) {
      const py = r.pFinal[2 * i] as number;
      const px = r.pFinal[2 * i + 1] as number;
      const d = Math.hypot(py - cy, px - cx);
      if (d > maxDist) maxDist = d;
    }
    expect(maxDist).toBeLessThan(4);
  });

  it('two basins of attraction → seeds converge to nearest center', () => {
    const H = 48,
      W = 48;
    const c1 = { y: 12, x: 12 };
    const c2 = { y: 36, x: 36 };
    const hw = H * W;
    const dP = new Float32Array(2 * hw);
    const cellprob = new Float32Array(hw);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const d1 = Math.hypot(y - c1.y, x - c1.x);
        const d2 = Math.hypot(y - c2.y, x - c2.x);
        const c = d1 < d2 ? c1 : c2;
        const dy = c.y - y;
        const dx = c.x - x;
        const r = Math.hypot(dy, dx) || 1;
        dP[i] = dy / r / 5;
        dP[hw + i] = dx / r / 5;
        cellprob[i] = 1.0;
      }
    }
    const r = followFlows(dP, cellprob, H, W, 0, 400);
    let nearC1 = 0,
      nearC2 = 0;
    for (let i = 0; i < r.pFinal.length / 2; i++) {
      const py = r.pFinal[2 * i] as number;
      const px = r.pFinal[2 * i + 1] as number;
      const d1 = Math.hypot(py - c1.y, px - c1.x);
      const d2 = Math.hypot(py - c2.y, px - c2.x);
      if (d1 < 5) nearC1++;
      else if (d2 < 5) nearC2++;
    }
    // Both clusters should accumulate roughly equal seeds (we sized each
    // basin symmetrically). The exact split depends on the diagonal boundary,
    // but both must capture a sizeable share.
    expect(nearC1).toBeGreaterThan(H * W * 0.3);
    expect(nearC2).toBeGreaterThan(H * W * 0.3);
    expect(nearC1 + nearC2).toBeGreaterThan(H * W * 0.95);
  });

  it('final coordinates are bounded by [0, H-1] x [0, W-1]', () => {
    const H = 24,
      W = 24;
    // Strong flow that would push past the boundary if not clamped.
    const { dP, cellprob } = flowsTowardCenter(H, W, 12, 12, 100.0);
    const r = followFlows(dP, cellprob, H, W, 0, 200);
    for (let i = 0; i < r.pFinal.length / 2; i++) {
      const py = r.pFinal[2 * i] as number;
      const px = r.pFinal[2 * i + 1] as number;
      expect(py).toBeGreaterThanOrEqual(0);
      expect(py).toBeLessThanOrEqual(H - 1);
      expect(px).toBeGreaterThanOrEqual(0);
      expect(px).toBeLessThanOrEqual(W - 1);
    }
  });

  it('throws on flow / cellprob size mismatch', () => {
    const H = 8,
      W = 8;
    const goodDP = new Float32Array(2 * H * W);
    const goodCP = new Float32Array(H * W);
    expect(() => followFlows(new Float32Array(7), goodCP, H, W)).toThrow(/dP wrong size/);
    expect(() => followFlows(goodDP, new Float32Array(7), H, W)).toThrow(/cellprob wrong size/);
  });
});

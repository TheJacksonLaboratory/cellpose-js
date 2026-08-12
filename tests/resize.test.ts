/**
 * Direct correctness tests for diameterResize's bilinear interpolation.
 *
 * The pure-JS bilinear (resize.ts) replaced the earlier canvas-based path
 * that quantized to 8-bit. These tests anchor the math against:
 *   - identity (no resize),
 *   - 2× downscale of a known pattern (each dst pixel is the mean of a 2×2
 *     src block under the pixel-center mapping),
 *   - 2× upscale of a known pattern (corners replicate edge, middles
 *     interpolate),
 *   - per-channel independence (channels don't bleed into each other),
 *   - scale return value matches Python's image_scaling = 30 / diameter.
 */
import { describe, it, expect } from 'vitest';
import { diameterResize, resizeChw } from '../src/preprocess/resize.js';

describe('resizeChw (used by the resample path to upsample flows)', () => {
  it('identity: same dst size returns the same values', () => {
    const W = 4,
      H = 4;
    const chw = new Float32Array(3 * H * W);
    for (let i = 0; i < chw.length; i++) chw[i] = i;
    const out = resizeChw(chw, 3, W, H, W, H);
    for (let i = 0; i < chw.length; i++) expect(out[i]).toBeCloseTo(chw[i] as number, 5);
  });

  it('matches diameterResize channel-for-channel on the same target size', () => {
    const W = 16,
      H = 16;
    const chw = new Float32Array(3 * H * W);
    for (let i = 0; i < chw.length; i++) chw[i] = Math.sin(i * 0.37);
    // diameter=60 → scale 0.5 → 8x8.
    const viaDiameter = diameterResize(chw, W, H, { channels: 3, diameter: 60 });
    const direct = resizeChw(chw, 3, W, H, viaDiameter.width, viaDiameter.height);
    expect(direct.length).toBe(viaDiameter.data.length);
    for (let i = 0; i < direct.length; i++) {
      expect(direct[i]).toBeCloseTo(viaDiameter.data[i] as number, 6);
    }
  });

  it('upsamples a 3-plane flow+cellprob stack without cross-channel bleed', () => {
    // 3 planes (dPy, dPx, cellprob), each constant and distinct. Bilinear
    // interpolation of a constant plane must stay that constant everywhere.
    const W = 4,
      H = 4;
    const chw = new Float32Array(3 * H * W);
    chw.fill(-2, 0, H * W);
    chw.fill(5, H * W, 2 * H * W);
    chw.fill(0.75, 2 * H * W, 3 * H * W);
    const out = resizeChw(chw, 3, W, H, 8, 8);
    expect(out.length).toBe(3 * 64);
    for (let i = 0; i < 64; i++) expect(out[i]).toBeCloseTo(-2, 5);
    for (let i = 64; i < 128; i++) expect(out[i]).toBeCloseTo(5, 5);
    for (let i = 128; i < 192; i++) expect(out[i]).toBeCloseTo(0.75, 5);
  });

  it('throws when the input length does not match channels * srcW * srcH', () => {
    expect(() => resizeChw(new Float32Array(10), 3, 4, 4, 8, 8)).toThrow(/expected 48 floats/);
  });
});

describe('diameterResize bilinear (pure JS)', () => {
  it('identity: srcSize == dstSize → values bit-exact match input', () => {
    const W = 8,
      H = 8;
    const chw = new Float32Array(3 * H * W);
    for (let i = 0; i < chw.length; i++) chw[i] = i;
    // diameter = targetDiameter = 30 → fast-path scale=1 copy.
    const r = diameterResize(chw, W, H, { channels: 3, diameter: 30 });
    expect(r.scale).toBe(1);
    expect(r.width).toBe(W);
    expect(r.height).toBe(H);
    for (let i = 0; i < chw.length; i++) expect(r.data[i]).toBe(chw[i]);
  });

  it('2× downscale: each dst pixel = mean of corresponding 2×2 src block', () => {
    // 4×4 src with values 0..15 row-major (single channel).
    const srcW = 4,
      srcH = 4;
    const chw = new Float32Array(srcW * srcH);
    for (let i = 0; i < chw.length; i++) chw[i] = i;
    // diameter=60, target=30 → scale = 0.5 → dst 2x2.
    const r = diameterResize(chw, srcW, srcH, { channels: 1, diameter: 60 });
    expect(r.scale).toBe(0.5);
    expect(r.width).toBe(2);
    expect(r.height).toBe(2);
    // dst[0,0] samples sy = 0.5, sx = 0.5 → corners (0,0)(0,1)(1,0)(1,1) at 25%
    // each → mean(0, 1, 4, 5) = 2.5.
    expect(r.data[0]).toBeCloseTo(2.5, 5);
    // dst[0,1]: sy=0.5, sx=2.5 → mean(2, 3, 6, 7) = 4.5.
    expect(r.data[1]).toBeCloseTo(4.5, 5);
    // dst[1,0]: sy=2.5, sx=0.5 → mean(8, 9, 12, 13) = 10.5.
    expect(r.data[2]).toBeCloseTo(10.5, 5);
    // dst[1,1]: sy=2.5, sx=2.5 → mean(10, 11, 14, 15) = 12.5.
    expect(r.data[3]).toBeCloseTo(12.5, 5);
  });

  it('2× upscale: corners replicate, interior interpolates linearly', () => {
    // 2×2 src: row 0 = [0, 10], row 1 = [20, 30]. Single channel.
    const chw = new Float32Array([0, 10, 20, 30]);
    // diameter=15, target=30 → scale=2 → dst 4x4.
    const r = diameterResize(chw, 2, 2, { channels: 1, diameter: 15 });
    expect(r.scale).toBe(2);
    expect(r.width).toBe(4);
    expect(r.height).toBe(4);
    // Top-left dst corner samples sy=sx=-0.25; both y0/y1 and x0/x1 clamp to
    // 0 (edge replication) → result == src[0,0] == 0.
    expect(r.data[0]).toBeCloseTo(0, 5);
    // Top-right dst corner: sy=-0.25, sx=1.25 → y clamps to row 0, x to
    // col 1 (also clamps since x1=2 is out of bounds) → result == src[0,1] == 10.
    expect(r.data[3]).toBeCloseTo(10, 5);
    // Bottom-left corner: → src[1,0] == 20.
    expect(r.data[12]).toBeCloseTo(20, 5);
    // Bottom-right corner: → src[1,1] == 30.
    expect(r.data[15]).toBeCloseTo(30, 5);
    // Interior dst[1, 1] samples sy=0.25, sx=0.25 → fy=fx=0.25 → weighted
    // sum: 0*0.75² + 10*0.75*0.25 + 20*0.25*0.75 + 30*0.25² = 1.875 + 3.75
    // + 1.875 = 7.5. (Also the value of the linear extrapolation 10x + 20y
    // at (0.25, 0.25), which is the sanity check: bilinear is exact for
    // linear functions.)
    expect(r.data[5]).toBeCloseTo(7.5, 5);
  });

  it('channels are independent (no bleed between channel buffers)', () => {
    // 2x2 image, 3 channels: each channel filled with a distinct constant.
    const chw = new Float32Array(3 * 4);
    chw.fill(100, 0, 4);
    chw.fill(200, 4, 8);
    chw.fill(50, 8, 12);
    // 2x upscale: each channel of the dst should still be (almost) constant.
    const r = diameterResize(chw, 2, 2, { channels: 3, diameter: 15 });
    expect(r.width).toBe(4);
    expect(r.height).toBe(4);
    // dst layout is also CHW: channel c occupies [c*hwOut, (c+1)*hwOut).
    for (let i = 0; i < 16; i++) expect(r.data[i]).toBeCloseTo(100, 5);
    for (let i = 16; i < 32; i++) expect(r.data[i]).toBeCloseTo(200, 5);
    for (let i = 32; i < 48; i++) expect(r.data[i]).toBeCloseTo(50, 5);
  });

  it('scale return value equals 30 / diameter (cellpose convention)', () => {
    const chw = new Float32Array(3 * 32 * 32);
    expect(diameterResize(chw, 32, 32, { channels: 3, diameter: 60 }).scale).toBeCloseTo(0.5, 6);
    expect(diameterResize(chw, 32, 32, { channels: 3, diameter: 15 }).scale).toBeCloseTo(2.0, 6);
    expect(diameterResize(chw, 32, 32, { channels: 3, diameter: 30 }).scale).toBe(1);
  });

  it('FP precision: constant input round-trips to FP-rounding tolerance', () => {
    // Motivation: fix #2 replaced the canvas-based resize that quantized to
    // uint8 (~1/255 = 0.0039 error per pixel, AFTER rescaling back to the
    // source range). A constant-valued image through that path picked up
    // ~0.004 error from the 8-bit quantization. With pure-JS bilinear on
    // Float32, a constant input stays constant to FP-rounding noise.
    const W = 24,
      H = 24;
    const src = new Float32Array(W * H);
    src.fill(0.42);
    const down = diameterResize(src, W, H, { channels: 1, diameter: 60 });
    const up = diameterResize(down.data, down.width, down.height, {
      channels: 1,
      diameter: 15,
    });
    let maxAbs = 0;
    for (let i = 0; i < up.data.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs((up.data[i] as number) - 0.42));
    }
    // The 1/255 floor is 0.0039. Anything well below that proves no
    // 8-bit-quantization step is in the path. FP roundtrip on a constant is
    // typically <1e-7.
    expect(maxAbs).toBeLessThan(1e-5);
  });
});

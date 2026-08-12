/**
 * Unit tests for buildCpsamChannels (preprocess/channels.ts).
 *
 * Coverage:
 *   - Passthrough (default, neither chan nor chan2 given): first up-to-3 source
 *     channels copied through, matching upstream v4's `x[..., :3]`.
 *   - chan=0 grayscale: MEAN across color channels (alpha excluded for RGBA).
 *   - chan=1/2/3 picks R/G/B from a pixel-interleaved source (no averaging).
 *   - chan=k (k>3) picks an arbitrary source channel from N-channel data.
 *   - chan2 != 0 places the secondary channel in output slot 1.
 *   - Supplying either chan or chan2 leaves passthrough mode.
 *   - Layout: output is CHW (channel-major), so adjacent floats within a
 *     channel are pixel neighbors.
 *   - Error paths: wrong total length; out-of-range / non-integer channel index.
 *   - Regression: a green-dominant RGBA image survives the default (the old
 *     red-only pick used to blank it, yielding empty masks).
 */
import { describe, it, expect } from 'vitest';
import { buildCpsamChannels } from '../src/preprocess/channels.js';

describe('buildCpsamChannels — passthrough default', () => {
  it('RGBA source: R,G,B pass through to slots 0/1/2, alpha dropped', () => {
    // 2x1 RGBA: p0 = (10, 11, 12, 255), p1 = (20, 21, 22, 0).
    const src = new Uint8Array([10, 11, 12, 255, 20, 21, 22, 0]);
    const out = buildCpsamChannels(src, 2, 1, 4);
    expect(Array.from(out.slice(0, 2))).toEqual([10, 20]); // R
    expect(Array.from(out.slice(2, 4))).toEqual([11, 21]); // G
    expect(Array.from(out.slice(4, 6))).toEqual([12, 22]); // B — alpha never reaches a slot
  });

  it('RGB source: all three channels pass through unchanged', () => {
    const src = new Uint8Array([1, 2, 3, 4, 5, 6]); // 2 pixels RGB
    const out = buildCpsamChannels(src, 2, 1, 3);
    expect(Array.from(out.slice(0, 2))).toEqual([1, 4]);
    expect(Array.from(out.slice(2, 4))).toEqual([2, 5]);
    expect(Array.from(out.slice(4, 6))).toEqual([3, 6]);
  });

  it('grayscale source: lands in slot 0, slots 1/2 stay zero', () => {
    const src = new Uint8Array([10, 20, 30, 40]);
    const out = buildCpsamChannels(src, 2, 2, 1);
    expect(out.length).toBe(3 * 2 * 2);
    expect(Array.from(out.slice(0, 4))).toEqual([10, 20, 30, 40]);
    expect(Array.from(out.slice(4, 8))).toEqual([0, 0, 0, 0]);
    expect(Array.from(out.slice(8, 12))).toEqual([0, 0, 0, 0]);
  });

  it('2-channel source: c0 → slot 0, c1 → slot 1, slot 2 stays zero', () => {
    const src = new Uint8Array([1, 2, 3, 4]); // 2 pixels, 2 channels
    const out = buildCpsamChannels(src, 2, 1, 2);
    expect(Array.from(out.slice(0, 2))).toEqual([1, 3]);
    expect(Array.from(out.slice(2, 4))).toEqual([2, 4]);
    expect(Array.from(out.slice(4, 6))).toEqual([0, 0]);
  });

  it('N>4 source: truncates to the first 3 channels (matches upstream x[..., :3])', () => {
    // 2 pixels, 5 channels: p0 = [0,1,2,3,4], p1 = [10,11,12,13,14].
    const src = new Uint8Array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14]);
    const out = buildCpsamChannels(src, 2, 1, 5);
    expect(Array.from(out.slice(0, 2))).toEqual([0, 10]);
    expect(Array.from(out.slice(2, 4))).toEqual([1, 11]);
    expect(Array.from(out.slice(4, 6))).toEqual([2, 12]);
  });

  it('regression: green-dominant RGBA is not blanked (signal survives in slot 1)', () => {
    // Fluorescence-like pixel: signal only in green, red/blue ~0, opaque.
    // The original bug took red → 0 everywhere → empty masks. Passthrough keeps
    // green as its own normalized channel, which is what upstream v4 feeds too.
    const src = new Uint8Array([0, 120, 0, 255, 0, 240, 0, 255]);
    const out = buildCpsamChannels(src, 2, 1, 4);
    expect(Array.from(out.slice(0, 2))).toEqual([0, 0]); // R
    expect(Array.from(out.slice(2, 4))).toEqual([120, 240]); // G — the point: not blanked
    expect(Array.from(out.slice(4, 6))).toEqual([0, 0]); // B
  });

  it('Float32 source preserves precision (no quantization)', () => {
    const src = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const out = buildCpsamChannels(src, 2, 2, 1);
    expect(out[0]).toBeCloseTo(0.1, 6);
    expect(out[3]).toBeCloseTo(0.4, 6);
  });

  it('setting only chan2 leaves passthrough mode (chan then defaults to 0)', () => {
    // p0 = (3,6,9,255), p1 = (30,60,90,255). chan defaults to 0 → grayscale mean.
    const src = new Uint8Array([3, 6, 9, 255, 30, 60, 90, 255]);
    const out = buildCpsamChannels(src, 2, 1, 4, { chan2: 2 });
    expect(Array.from(out.slice(0, 2))).toEqual([6, 60]); // mean(R,G,B)
    expect(Array.from(out.slice(2, 4))).toEqual([6, 60]); // chan2=2 → G
  });
});

describe('buildCpsamChannels — explicit chan / chan2', () => {
  it('grayscale source (channels=1, chan=0): mean of one channel is that channel', () => {
    // 2x2 grayscale: pixel values 10, 20, 30, 40.
    const src = new Uint8Array([10, 20, 30, 40]);
    const out = buildCpsamChannels(src, 2, 2, 1, { chan: 0 });
    // (3, 2, 2) CHW: channel 0 = grayscale, channels 1/2 = zeros.
    expect(out.length).toBe(3 * 2 * 2);
    expect(Array.from(out.slice(0, 4))).toEqual([10, 20, 30, 40]); // channel 0
    expect(Array.from(out.slice(4, 8))).toEqual([0, 0, 0, 0]); // channel 1
    expect(Array.from(out.slice(8, 12))).toEqual([0, 0, 0, 0]); // channel 2
  });

  it('RGB source (channels=3, chan=0): grayscale = mean of R,G,B', () => {
    // 2 pixels RGB: p0=(3,6,9) mean 6, p1=(30,60,90) mean 60.
    const src = new Uint8Array([3, 6, 9, 30, 60, 90]);
    const out = buildCpsamChannels(src, 2, 1, 3, { chan: 0 });
    expect(Array.from(out.slice(0, 2))).toEqual([6, 60]);
    expect(Array.from(out.slice(2, 4))).toEqual([0, 0]);
    expect(Array.from(out.slice(4, 6))).toEqual([0, 0]);
  });

  it('RGBA source (channels=4, chan=0): alpha is EXCLUDED from the mean', () => {
    // p0=(R=3,G=6,B=9,A=255) → mean(3,6,9)=6 ; p1=(30,60,90,0) → mean=60.
    // Alpha (255 / 0) must not perturb the grayscale value.
    const src = new Uint8Array([3, 6, 9, 255, 30, 60, 90, 0]);
    const out = buildCpsamChannels(src, 2, 1, 4, { chan: 0 });
    expect(Array.from(out.slice(0, 2))).toEqual([6, 60]);
  });

  it('explicit chan=0 on green-dominant RGBA still averages (alpha excluded)', () => {
    const src = new Uint8Array([0, 120, 0, 255, 0, 240, 0, 255]);
    const out = buildCpsamChannels(src, 2, 1, 4, { chan: 0 });
    expect(out[0]).toBeCloseTo(120 / 3, 5); // 40
    expect(out[1]).toBeCloseTo(240 / 3, 5); // 80
  });

  it('RGBA source (channels=4): chan=1 picks R, chan2=3 picks B (no averaging)', () => {
    // 2x1 RGBA: pixel0 = (R=10, G=11, B=12, A=255), pixel1 = (R=20, G=21, B=22, A=255).
    const src = new Uint8Array([10, 11, 12, 255, 20, 21, 22, 255]);
    const out = buildCpsamChannels(src, 2, 1, 4, { chan: 1, chan2: 3 });
    // Channel 0 = R = [10, 20]; Channel 1 = B = [12, 22]; Channel 2 = zeros.
    expect(Array.from(out.slice(0, 2))).toEqual([10, 20]);
    expect(Array.from(out.slice(2, 4))).toEqual([12, 22]);
    expect(Array.from(out.slice(4, 6))).toEqual([0, 0]);
  });

  it('RGB source (channels=3): chan=2 picks G, chan2=0 leaves slot 1 empty', () => {
    const src = new Uint8Array([1, 2, 3, 4, 5, 6]); // 2 pixels RGB
    const out = buildCpsamChannels(src, 2, 1, 3, { chan: 2, chan2: 0 });
    // Channel 0 = G = [2, 5]
    expect(Array.from(out.slice(0, 2))).toEqual([2, 5]);
    expect(Array.from(out.slice(2, 4))).toEqual([0, 0]);
    expect(Array.from(out.slice(4, 6))).toEqual([0, 0]);
  });

  it('multichannel (channels=5): chan=5 picks channel 4, chan2=2 picks channel 1', () => {
    // 2 pixels, 5 channels each. Distinct markers must be selectable, not blended.
    // p0 = [0,1,2,3,4], p1 = [10,11,12,13,14].
    const src = new Uint8Array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14]);
    const out = buildCpsamChannels(src, 2, 1, 5, { chan: 5, chan2: 2 });
    // Channel 0 = source channel 4 = [4, 14]; Channel 1 = source channel 1 = [1, 11].
    expect(Array.from(out.slice(0, 2))).toEqual([4, 14]);
    expect(Array.from(out.slice(2, 4))).toEqual([1, 11]);
    expect(Array.from(out.slice(4, 6))).toEqual([0, 0]);
  });

  it('Float32 source preserves precision (no quantization)', () => {
    const src = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const out = buildCpsamChannels(src, 2, 2, 1, { chan: 0 });
    expect(out[0]).toBeCloseTo(0.1, 6);
    expect(out[3]).toBeCloseTo(0.4, 6);
  });

  it('chan=0 on a 1-channel source is the identity (mean of one channel)', () => {
    const src = new Uint8Array([10, 20, 30, 40]);
    const out = buildCpsamChannels(src, 2, 2, 1, { chan: 0 });
    expect(Array.from(out.slice(0, 4))).toEqual([10, 20, 30, 40]);
    expect(Array.from(out.slice(4, 8))).toEqual([0, 0, 0, 0]);
  });

  it('throws on length mismatch', () => {
    const src = new Uint8Array([1, 2, 3]); // 3 floats for a 2x2 grayscale (needs 4)
    expect(() => buildCpsamChannels(src, 2, 2, 1)).toThrow(/expected 4/);
  });

  it('throws on channel index exceeding source channel count', () => {
    const src = new Uint8Array([1, 2, 3, 4]); // 2x2 grayscale
    // chan=2 requires a source with >=2 channels.
    expect(() => buildCpsamChannels(src, 2, 2, 1, { chan: 2 })).toThrow(
      /channel index 2 not available/,
    );
  });

  it('throws on a non-integer channel index', () => {
    const src = new Uint8Array([10, 11, 12, 255]); // 1px RGBA
    expect(() => buildCpsamChannels(src, 1, 1, 4, { chan: 1.5 })).toThrow(/non-negative integer/);
  });

  it('throws on a non-positive channel count (avoids a divide-by-zero → NaN)', () => {
    // channels=0 would otherwise slip past the length check (0 === 0) and make
    // the grayscale mean divide by zero, silently yielding NaNs.
    expect(() => buildCpsamChannels(new Uint8Array(0), 2, 2, 0)).toThrow(
      /channels must be a positive integer/,
    );
  });
});

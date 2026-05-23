/**
 * Unit tests for buildCpsamChannels (preprocess/channels.ts).
 *
 * Coverage:
 *   - chan=0 grayscale: single source channel replicates into output 0.
 *   - chan=1/2/3 picks R/G/B from a pixel-interleaved source.
 *   - chan2 != 0 places the secondary channel in output slot 1.
 *   - Output slot 2 is always zero.
 *   - Layout: output is CHW (channel-major), so adjacent floats within a
 *     channel are pixel neighbors.
 *   - Error paths: wrong total length; out-of-range channel index.
 */
import { describe, it, expect } from 'vitest';
import { buildCpsamChannels } from '../src/preprocess/channels.js';

describe('buildCpsamChannels', () => {
  it('grayscale source (channels=1, chan=0) replicates to output 0', () => {
    // 2x2 grayscale: pixel values 10, 20, 30, 40.
    const src = new Uint8Array([10, 20, 30, 40]);
    const out = buildCpsamChannels(src, 2, 2, 1, { chan: 0 });
    // (3, 2, 2) CHW: channel 0 = grayscale, channels 1/2 = zeros.
    expect(out.length).toBe(3 * 2 * 2);
    expect(Array.from(out.slice(0, 4))).toEqual([10, 20, 30, 40]); // channel 0
    expect(Array.from(out.slice(4, 8))).toEqual([0, 0, 0, 0]); // channel 1
    expect(Array.from(out.slice(8, 12))).toEqual([0, 0, 0, 0]); // channel 2
  });

  it('RGBA source (channels=4): chan=1 picks R, chan2=3 picks B', () => {
    // 2x1 RGBA: pixel0 = (R=10, G=11, B=12, A=255), pixel1 = (R=20, G=21, B=22, A=255).
    const src = new Uint8Array([10, 11, 12, 255, 20, 21, 22, 255]);
    const out = buildCpsamChannels(src, 2, 1, 4, { chan: 1, chan2: 3 });
    // Channel 0 = R = [10, 20]
    // Channel 1 = B = [12, 22]
    // Channel 2 = zeros
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

  it('Float32 source preserves precision (no quantization)', () => {
    const src = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const out = buildCpsamChannels(src, 2, 2, 1, { chan: 0 });
    expect(out[0]).toBeCloseTo(0.1, 6);
    expect(out[3]).toBeCloseTo(0.4, 6);
  });

  it('defaults to chan=0, chan2=0 (grayscale-equivalent)', () => {
    const src = new Uint8Array([10, 20, 30, 40]);
    const out = buildCpsamChannels(src, 2, 2, 1);
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
});

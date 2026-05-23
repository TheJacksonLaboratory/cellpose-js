/**
 * Unit tests for diameterResize parameter checks (preprocess/resize.ts).
 *
 * The real bilinear resize uses OffscreenCanvas / HTMLCanvas, which aren't
 * available in the Node-based test runner. These tests cover the
 * pre-canvas paths only: invalid diameter, the no-op pass-through when
 * scale ≈ 1, and the 4096×4096 memory guard added in M7.
 */
import { describe, it, expect } from 'vitest';
import { diameterResize } from '../src/preprocess/resize.js';

describe('diameterResize parameter validation', () => {
  it('throws when diameter <= 0', () => {
    const img = new Float32Array(3 * 64 * 64);
    expect(() => diameterResize(img, 64, 64, { channels: 3, diameter: 0 })).toThrow(
      /diameter must be positive/,
    );
    expect(() => diameterResize(img, 64, 64, { channels: 3, diameter: -5 })).toThrow(
      /diameter must be positive/,
    );
  });

  it('returns a copy + scale=1 when source diameter already matches target', () => {
    // diameter = targetDiameter (default 30) → scale = 1.0 → no canvas op.
    const img = new Float32Array(3 * 32 * 32);
    for (let i = 0; i < img.length; i++) img[i] = i;
    const r = diameterResize(img, 32, 32, { channels: 3, diameter: 30 });
    expect(r.scale).toBe(1);
    expect(r.width).toBe(32);
    expect(r.height).toBe(32);
    // Distinct buffer (no aliasing).
    expect(r.data).not.toBe(img);
    expect(r.data.length).toBe(img.length);
    for (let i = 0; i < img.length; i++) expect(r.data[i]).toBe(img[i]);
  });

  it('enforces the 4096x4096 output cap before allocating canvases', () => {
    // Source 1000×1000, asked for a 12.5x upscale → 12500×12500 destination.
    // That exceeds 4096×4096 — should throw with a clear message.
    const img = new Float32Array(3 * 1000 * 1000);
    expect(() => diameterResize(img, 1000, 1000, { channels: 3, diameter: 2.4 })).toThrow(
      /exceeds the safe limit/,
    );
  });

  it('memory-guard error includes the requested scale and source size', () => {
    const img = new Float32Array(3 * 2000 * 2000);
    let caught: Error | null = null;
    try {
      diameterResize(img, 2000, 2000, { channels: 3, diameter: 3 });
    } catch (e) {
      caught = e as Error;
    }
    expect(caught).not.toBeNull();
    expect(caught!.message).toMatch(/2000x2000/);
    expect(caught!.message).toMatch(/diameter is 3/);
  });
});

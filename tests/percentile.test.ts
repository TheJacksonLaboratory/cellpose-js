/**
 * Unit tests for the percentile() helper (preprocess/normalize.ts).
 *
 * percentile() is the foundation of normalize99 — anchoring it with direct
 * tests catches regressions that would otherwise only show up indirectly in
 * the normalize99 fixture diffs.
 *
 * Reference values match numpy.percentile(..., method='linear') which is
 * what cellpose's normalize99 calls.
 */
import { describe, it, expect } from 'vitest';
import { percentile } from '../src/preprocess/normalize.js';

describe('percentile (numpy linear interpolation)', () => {
  it('returns the only element for a singleton array', () => {
    expect(percentile([42], 50)).toBe(42);
    expect(percentile([42], 0)).toBe(42);
    expect(percentile([42], 99)).toBe(42);
  });

  it('returns NaN for empty input', () => {
    expect(Number.isNaN(percentile([], 50))).toBe(true);
  });

  it('p=0 is min, p=100 is max', () => {
    const data = [5, 3, 8, 1, 9, 2];
    expect(percentile(data, 0)).toBe(1);
    expect(percentile(data, 100)).toBe(9);
  });

  it('median (p=50) matches numpy on a uniform 1..10', () => {
    // np.percentile(range(1, 11), 50) = 5.5
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(data, 50)).toBeCloseTo(5.5, 6);
  });

  it('p=1 and p=99 on 0..99 match numpy linear interpolation', () => {
    // np.arange(100); np.percentile(x, 1) = 0.99; np.percentile(x, 99) = 98.01
    const data = Array.from({ length: 100 }, (_, i) => i);
    expect(percentile(data, 1)).toBeCloseTo(0.99, 6);
    expect(percentile(data, 99)).toBeCloseTo(98.01, 6);
  });

  it('sorts before computing — unordered input gives same answer', () => {
    const a = [1, 2, 3, 4, 5];
    const b = [5, 1, 3, 4, 2];
    expect(percentile(a, 25)).toBe(percentile(b, 25));
    expect(percentile(a, 75)).toBe(percentile(b, 75));
  });

  it('does not mutate the input (works on a copy)', () => {
    const data = [5, 3, 8, 1, 9, 2];
    const snap = [...data];
    percentile(data, 50);
    expect(data).toEqual(snap);
  });

  it('accepts Float32Array input', () => {
    const f = Float32Array.from([0, 0.25, 0.5, 0.75, 1.0]);
    expect(percentile(f, 50)).toBeCloseTo(0.5, 6);
  });
});

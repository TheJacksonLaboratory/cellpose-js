/**
 * Percentile normalization, mirroring cellpose.transforms.normalize99.
 *
 * For each channel independently:
 *   1. Compute the `lower`-th and `upper`-th percentile (default 1, 99).
 *   2. If `p99 - p01 > 1e-3`: `(x - p01) / (p99 - p01)`.
 *   3. Else: zero the channel.
 *
 * Values are NOT clipped to [0, 1] — CPSAM was trained on percentile-rescaled
 * (but not value-clipped) data, and negatives / >1 values are expected at the
 * tails.
 */

export interface NormalizeOptions {
  /** Lower percentile. Default 1 (matches cellpose). */
  lower?: number;
  /** Upper percentile. Default 99 (matches cellpose). */
  upper?: number;
  /** Optional pixel-value inversion after normalization (1 - x). */
  invert?: boolean;
}

/**
 * Compute the requested percentile of a Float32Array.
 *
 * Uses the same "linear" interpolation method numpy.percentile defaults to:
 *   idx = p/100 * (n-1)
 *   k = floor(idx), d = idx - k
 *   sorted[k] * (1 - d) + sorted[k+1] * d
 *
 * Sorts in-place on a copy. O(n log n).
 */
export function percentile(data: ArrayLike<number>, p: number): number {
  if (data.length === 0) return NaN;
  const sorted = Float32Array.from(data as ArrayLike<number>);
  sorted.sort();
  const idx = (p / 100) * (sorted.length - 1);
  const k = Math.floor(idx);
  const d = idx - k;
  if (k + 1 >= sorted.length) return sorted[k] as number;
  return (sorted[k] as number) * (1 - d) + (sorted[k + 1] as number) * d;
}

/**
 * Per-channel percentile normalization for an image in CHW layout.
 *
 * @param chw      Float32 array of length channels * height * width, channel-major.
 * @param channels Number of channels.
 * @param hw       Height * width per channel.
 * @returns        New Float32Array (same length), normalized.
 */
export function normalizePerChannel(
  chw: Float32Array,
  channels: number,
  hw: number,
  opts: NormalizeOptions = {}
): Float32Array {
  const { lower = 1, upper = 99, invert = false } = opts;
  if (chw.length !== channels * hw) {
    throw new Error(`normalizePerChannel: expected ${channels * hw} floats, got ${chw.length}`);
  }
  const out = new Float32Array(chw.length);
  for (let c = 0; c < channels; c++) {
    const offset = c * hw;
    const view = chw.subarray(offset, offset + hw);
    const p01 = percentile(view, lower);
    const p99 = percentile(view, upper);
    const range = p99 - p01;
    if (range > 1e-3) {
      const inv = 1 / range;
      for (let i = 0; i < hw; i++) {
        const v = ((view[i] as number) - p01) * inv;
        out[offset + i] = invert ? 1 - v : v;
      }
    }
    // else: leave zeros (matches cellpose's `X[:] = 0` branch).
  }
  return out;
}

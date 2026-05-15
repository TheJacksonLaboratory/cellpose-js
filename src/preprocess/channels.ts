/**
 * Map arbitrary input image layouts to CPSAM's 3-channel NCHW expectation.
 *
 * Cellpose-SAM (per the bioRxiv preprint) was trained with channel-shuffling
 * augmentation, so it does not privilege any specific channel as cyto vs
 * nuclei. Still, we keep the legacy `chan` (cytoplasm) / `chan2` (nuclei)
 * semantics so callers familiar with Cellpose 1-3 can lift their existing
 * parameter choices unchanged.
 *
 * Channel mapping (mirrors Cellpose's convention):
 *   chan = 0    → grayscale: replicate the single channel into all 3 outputs
 *   chan = 1    → red
 *   chan = 2    → green
 *   chan = 3    → blue
 *   chan2 same indexing for the secondary (nuclear) channel; 0 = no second channel.
 *
 * Output is always a (3, H, W) Float32 array, channel-major, in the order
 * [chan, chan2-or-zero, zero].
 */

export interface ChannelMapOptions {
  /** Primary (cytoplasm) channel index. 0 = grayscale source. Default 0. */
  chan?: 0 | 1 | 2 | 3;
  /** Secondary (nuclear) channel index. 0 = none. Default 0. */
  chan2?: 0 | 1 | 2 | 3;
}

/**
 * Build the 3-channel CPSAM input from an arbitrary source layout.
 *
 * @param src       Source pixel data. RGBA from canvas (channels=4), RGB
 *                  (channels=3), or grayscale (channels=1).
 * @param width     Image width.
 * @param height    Image height.
 * @param channels  Channel count of `src` (1, 3, or 4).
 * @returns         (3, H, W) Float32 array, channel-major.
 */
export function buildCpsamChannels(
  src: Uint8ClampedArray | Uint8Array | Float32Array,
  width: number,
  height: number,
  channels: number,
  opts: ChannelMapOptions = {},
): Float32Array {
  const { chan = 0, chan2 = 0 } = opts;
  const hw = width * height;
  if (src.length !== hw * channels) {
    throw new Error(
      `buildCpsamChannels: expected ${hw * channels} values for ${channels}ch image, got ${src.length}`,
    );
  }
  const out = new Float32Array(3 * hw);

  // Source is pixel-interleaved (e.g. RGBA): src[i*ch + c].
  // Extract channel `idx` (1=R, 2=G, 3=B). idx=0 means "grayscale source" —
  // we treat that as the first source channel (handles single-channel input
  // directly, and recovers a sensible grayscale-ish from RGB by taking R).
  const pickChannel = (idx: 0 | 1 | 2 | 3): Float32Array => {
    const buf = new Float32Array(hw);
    const srcIdx = idx === 0 ? 0 : idx - 1;
    if (srcIdx >= channels) {
      throw new Error(`channel index ${idx} not available in ${channels}-channel source`);
    }
    for (let i = 0; i < hw; i++) {
      buf[i] = src[i * channels + srcIdx] as number;
    }
    return buf;
  };

  // Primary channel goes to output channel 0.
  out.set(pickChannel(chan), 0);

  // Secondary (nuclear) channel goes to output channel 1 if requested.
  if (chan2 !== 0) {
    out.set(pickChannel(chan2), hw);
  }
  // Output channel 2 is left as zeros (matches Cellpose convention for the
  // unused third channel slot).

  return out;
}

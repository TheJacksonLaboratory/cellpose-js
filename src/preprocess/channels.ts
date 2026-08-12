/**
 * Map arbitrary input image layouts to CPSAM's 3-channel NCHW expectation.
 *
 * Cellpose-SAM (per the bioRxiv preprint) was trained with channel-shuffling
 * augmentation, so it does not privilege any specific channel as cyto vs
 * nuclei. Upstream Cellpose v4 leans on that: `cellpose.transforms.convert_image`
 * performs NO channel selection — it just orders the array as YXC, truncates to
 * the first 3 channels, and hands them to the network, which normalizes each
 * independently. The legacy `channels=` argument is deprecated there and logs
 * "Cellpose4 takes inputs with arbitrary channel orders".
 *
 * This module follows that default, while keeping the legacy `chan` / `chan2`
 * selection available for callers who need it.
 *
 * ## Passthrough (default — neither `chan` nor `chan2` given)
 *
 * The first `min(3, channels)` source channels are copied straight through to
 * output slots 0..2; any remaining output slot stays zero. This matches
 * upstream's `x[..., :3]` truncation.
 *
 *   1-channel source → [gray,  0,     0    ]
 *   2-channel source → [c0,    c1,    0    ]
 *   3-channel (RGB)  → [R,     G,     B    ]
 *   4-channel (RGBA) → [R,     G,     B    ]   (alpha dropped by truncation)
 *   N > 4            → [c0,    c1,    c2   ]   (upstream warns and truncates)
 *
 * Because `normalizePerChannel` runs per channel, an RGB fluorescence composite
 * keeps all three markers with independent dynamic ranges — the same input the
 * Python reference would build.
 *
 * ## Explicit selection (`chan` and/or `chan2` given)
 *
 * Supplying either option switches to the Cellpose 1-3 style mapping, so
 * callers with existing parameter choices can lift them unchanged:
 *
 *   chan = 0    → grayscale: MEAN across the source's color channels
 *   chan = 1    → red    (source channel 0)
 *   chan = 2    → green  (source channel 1)
 *   chan = 3    → blue   (source channel 2)
 *   chan = k    → source channel k-1 (0-based), for any k up to the source's
 *                 channel count — lets true multichannel microscopy images pick
 *                 a specific marker (e.g. a 5-channel stack: chan=4 → channel 3).
 *   chan2 same indexing for the secondary (nuclear) channel; 0 = no second channel.
 *
 * Grayscale (chan=0) AVERAGES channels — appropriate only when the caller
 * explicitly wants grayscale (e.g. an RGB display image whose signal is really
 * one thing). It mirrors Cellpose 1-3's `data.mean(axis=-1)` for channels=[0,0].
 * Distinct-data multichannel images must NOT use chan=0 — averaging different
 * markers is meaningless; select the relevant channel with chan/chan2 (>= 1)
 * instead, which never averages, or use the passthrough default.
 *
 * For canvas/PNG RGBA input (channels===4) the 4th channel is alpha (opacity),
 * not image signal, so it is EXCLUDED from the grayscale mean — otherwise a
 * fully-opaque image would blend a constant 255 into every pixel. A genuine
 * 4-channel image therefore should select channels explicitly rather than rely
 * on chan=0 (which would treat its 4th channel as alpha and drop it).
 *
 * Output is always a (3, H, W) Float32 array, channel-major.
 */

export interface ChannelMapOptions {
  /** Primary (cytoplasm) channel. 0 = grayscale (mean of color channels);
   *  k >= 1 selects source channel k-1 (0-based).
   *
   *  Leave BOTH `chan` and `chan2` unset for the default passthrough, which
   *  copies the first up-to-3 source channels through unchanged (matches
   *  upstream Cellpose v4). Setting either one switches to explicit selection. */
  chan?: number;
  /** Secondary (nuclear) channel, same indexing. 0 = none.
   *  Setting this switches out of passthrough mode; `chan` then defaults to 0. */
  chan2?: number;
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
  if (!Number.isInteger(channels) || channels < 1) {
    throw new Error(`buildCpsamChannels: channels must be a positive integer, got ${channels}`);
  }
  const hw = width * height;
  if (src.length !== hw * channels) {
    throw new Error(
      `buildCpsamChannels: expected ${hw * channels} values for ${channels}ch image, got ${src.length}`,
    );
  }
  const out = new Float32Array(3 * hw);

  // Neither option given → upstream v4 passthrough: copy the first up-to-3
  // source channels into output slots 0..2 and leave the rest zero.
  if (opts.chan === undefined && opts.chan2 === undefined) {
    const nCopy = Math.min(3, channels);
    for (let c = 0; c < nCopy; c++) {
      const offset = c * hw;
      for (let i = 0; i < hw; i++) {
        out[offset + i] = src[i * channels + c] as number;
      }
    }
    return out;
  }

  const { chan = 0, chan2 = 0 } = opts;

  // Source is pixel-interleaved (e.g. RGBA): src[i*ch + c].
  //   idx = 0  → grayscale = mean across color channels (alpha excluded for RGBA).
  //   idx >= 1 → source channel idx-1, selected directly (never averaged).
  const pickChannel = (idx: number): Float32Array => {
    if (!Number.isInteger(idx) || idx < 0) {
      throw new Error(`channel index must be a non-negative integer, got ${idx}`);
    }
    const buf = new Float32Array(hw);
    if (idx === 0) {
      // Mean over color channels. Canvas RGBA (channels===4) carries alpha in
      // the last slot; drop it so an opaque image isn't dragged toward 255.
      const colorCh = channels === 4 ? 3 : channels;
      const invN = 1 / colorCh;
      for (let i = 0; i < hw; i++) {
        const base = i * channels;
        let sum = 0;
        for (let c = 0; c < colorCh; c++) sum += src[base + c] as number;
        buf[i] = sum * invN;
      }
      return buf;
    }
    const srcIdx = idx - 1;
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

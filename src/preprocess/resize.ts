/**
 * Diameter-aware resize, mirroring cellpose.transforms.resize_image with
 * cv2.INTER_LINEAR (bilinear) as the interpolation.
 *
 * Browser bilinear resize is not bit-exact to OpenCV's — they use different
 * edge handling and rounding. We expect mean abs error ~1e-3 vs cv2; the
 * fixture-based parity tests use a wider tolerance for resize than for
 * pure-math ops like normalize.
 *
 * Implementation: use OffscreenCanvas where available (workers), HTMLCanvas
 * fallback for main-thread contexts. We resize one channel at a time as a
 * grayscale image (R = value, A = 255), read back as RGBA, and pull the R
 * channel.
 */

export interface ResizeResult {
  /** Resized pixels in CHW float32 layout. */
  data: Float32Array;
  /** New width. */
  width: number;
  /** New height. */
  height: number;
  /** Scale factor applied (new = original * scale). Used in postprocess to map masks back. */
  scale: number;
}

export interface DiameterResizeOptions {
  /** Source channel count (>= 1). */
  channels: number;
  /** Estimated cell diameter in source-image pixels. */
  diameter: number;
  /** Target diameter in resized pixels. CPSAM's training median is 30 px. */
  targetDiameter?: number;
}

/** Create a 2D canvas context. Prefers OffscreenCanvas for worker contexts. */
function createCanvas(w: number, h: number): {
  ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
} {
  if (typeof OffscreenCanvas !== 'undefined') {
    const c = new OffscreenCanvas(w, h);
    const ctx = c.getContext('2d');
    if (!ctx) throw new Error('OffscreenCanvas 2d context unavailable');
    return { ctx };
  }
  if (typeof document !== 'undefined') {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    if (!ctx) throw new Error('HTMLCanvas 2d context unavailable');
    return { ctx };
  }
  throw new Error('No canvas context available in this environment');
}

/**
 * Resize one channel using canvas bilinear.
 *
 * The channel data is packed into RGBA (R = value scaled to [0,255]), drawn,
 * resized, read back as RGBA, and the R channel is rescaled to the original
 * value range.
 *
 * This is lossy — values are quantized to uint8 in the canvas pipeline. For
 * percentile-normalized inputs in roughly [0, 1] this is acceptable (~1/255
 * quantization error). For raw images use the source-value range; callers
 * should normalize *after* resize if they need full precision.
 */
function resizeChannel(
  src: Float32Array,
  srcW: number, srcH: number,
  dstW: number, dstH: number
): Float32Array {
  // Find the channel's value range so quantization uses the full uint8.
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < src.length; i++) {
    const v = src[i] as number;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const span = (mx - mn) || 1;

  // Build the source ImageData.
  const srcRgba = new Uint8ClampedArray(srcW * srcH * 4);
  for (let i = 0; i < src.length; i++) {
    const u8 = Math.max(0, Math.min(255, Math.round(((src[i] as number) - mn) / span * 255)));
    srcRgba[i * 4]     = u8;
    srcRgba[i * 4 + 1] = u8;
    srcRgba[i * 4 + 2] = u8;
    srcRgba[i * 4 + 3] = 255;
  }
  const srcImageData = new ImageData(srcRgba, srcW, srcH);

  const { ctx: srcCtx } = createCanvas(srcW, srcH);
  srcCtx.putImageData(srcImageData, 0, 0);

  const { ctx: dstCtx } = createCanvas(dstW, dstH);
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.imageSmoothingQuality = 'high';
  // drawImage with a canvas source supports floating-point resizing via bilinear.
  // OffscreenCanvas accepts itself as a CanvasImageSource.
  dstCtx.drawImage(srcCtx.canvas as CanvasImageSource, 0, 0, dstW, dstH);

  const dstRgba = dstCtx.getImageData(0, 0, dstW, dstH).data;
  const out = new Float32Array(dstW * dstH);
  for (let i = 0; i < out.length; i++) {
    out[i] = (dstRgba[i * 4] as number) / 255 * span + mn;
  }
  return out;
}

/**
 * Resize a CHW Float32 image so that the estimated cell diameter matches
 * CPSAM's training median.
 *
 * @param chw      Source image in CHW Float32 layout.
 * @param width    Source width.
 * @param height   Source height.
 * @param opts     Channel count + diameter + (optional) target diameter.
 */
export function diameterResize(
  chw: Float32Array,
  width: number,
  height: number,
  opts: DiameterResizeOptions
): ResizeResult {
  const { channels, diameter, targetDiameter = 30 } = opts;
  if (!(diameter > 0)) {
    throw new Error(`diameterResize: diameter must be positive, got ${diameter}`);
  }
  const scale = targetDiameter / diameter;
  if (Math.abs(scale - 1) < 1e-3) {
    return { data: new Float32Array(chw), width, height, scale: 1 };
  }
  const dstW = Math.max(1, Math.round(width * scale));
  const dstH = Math.max(1, Math.round(height * scale));
  const hwOut = dstW * dstH;
  const hwIn = width * height;
  const out = new Float32Array(channels * hwOut);
  for (let c = 0; c < channels; c++) {
    const srcView = chw.subarray(c * hwIn, (c + 1) * hwIn);
    const resized = resizeChannel(srcView, width, height, dstW, dstH);
    out.set(resized, c * hwOut);
  }
  return { data: out, width: dstW, height: dstH, scale };
}

/**
 * Diameter-aware resize, mirroring cellpose.transforms.resize_image with
 * cv2.INTER_LINEAR (bilinear) as the interpolation.
 *
 * Pure-JS Float32 bilinear — no canvas roundtrip. Earlier versions of this
 * module pushed each channel through an OffscreenCanvas which quantized
 * values to uint8 (~1/255 relative error per pixel). The canvas path also
 * placed an environment dependency on OffscreenCanvas / HTMLCanvas, which
 * isn't available in Node/test contexts.
 *
 * Pixel-center mapping matches OpenCV's INTER_LINEAR:
 *   src_y = (dst_y + 0.5) * (srcH / dstH) - 0.5
 * with edge replication (BORDER_REPLICATE) outside [0, N-1].
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

/**
 * Resize one channel with bilinear interpolation on Float32 values.
 *
 * Pixel-center mapping is OpenCV's INTER_LINEAR convention:
 *   src_y = (dst_y + 0.5) * (srcH / dstH) - 0.5
 * Out-of-bounds samples replicate the nearest edge pixel (BORDER_REPLICATE).
 */
function resizeChannel(
  src: Float32Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Float32Array {
  const out = new Float32Array(dstW * dstH);
  const syScale = srcH / dstH;
  const sxScale = srcW / dstW;
  for (let dy = 0; dy < dstH; dy++) {
    const sy = (dy + 0.5) * syScale - 0.5;
    let y0 = Math.floor(sy);
    let y1 = y0 + 1;
    const fy = sy - y0;
    // Edge-replicate clamps. Hot-path branchless mins/maxes are not worth the
    // readability hit here — modern engines hoist these.
    if (y0 < 0) y0 = 0;
    else if (y0 > srcH - 1) y0 = srcH - 1;
    if (y1 < 0) y1 = 0;
    else if (y1 > srcH - 1) y1 = srcH - 1;
    const row0 = y0 * srcW;
    const row1 = y1 * srcW;
    const dstRow = dy * dstW;
    const wy1 = fy;
    const wy0 = 1 - fy;
    for (let dx = 0; dx < dstW; dx++) {
      const sx = (dx + 0.5) * sxScale - 0.5;
      let x0 = Math.floor(sx);
      let x1 = x0 + 1;
      const fx = sx - x0;
      if (x0 < 0) x0 = 0;
      else if (x0 > srcW - 1) x0 = srcW - 1;
      if (x1 < 0) x1 = 0;
      else if (x1 > srcW - 1) x1 = srcW - 1;
      const v00 = src[row0 + x0] as number;
      const v01 = src[row0 + x1] as number;
      const v10 = src[row1 + x0] as number;
      const v11 = src[row1 + x1] as number;
      const v0 = v00 * (1 - fx) + v01 * fx;
      const v1 = v10 * (1 - fx) + v11 * fx;
      out[dstRow + dx] = v0 * wy0 + v1 * wy1;
    }
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
  opts: DiameterResizeOptions,
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
  // Memory guard: each resized channel allocates dstW*dstH Float32 values.
  // For a 4096×4096 destination at 3 channels that's ~192 MB; anything
  // larger reliably OOMs on memory-constrained devices.
  const MAX_PIXELS = 4096 * 4096;
  if (dstW * dstH > MAX_PIXELS) {
    throw new Error(
      `diameterResize: requested output ${dstW}x${dstH} (scale ${scale.toFixed(2)}) ` +
        `exceeds the safe limit (4096x4096). Source is ${width}x${height}; estimated ` +
        `diameter is ${opts.diameter}. Increase diameter to downscale less aggressively.`,
    );
  }
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

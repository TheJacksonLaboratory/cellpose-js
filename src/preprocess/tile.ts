/**
 * Tiling + stitching helpers, mirroring cellpose.transforms.make_tiles
 * (non-augmented branch only — augmentation is a training concern).
 *
 * For an image with shape (C, Ly, Lx) and tile size B:
 *   ny = 1 if Ly <= B, else ceil((1 + 2*overlap) * Ly / B)
 *   nx = same for Lx
 *   ystart = linspace(0, Ly - B, ny).astype(int)
 *   xstart = linspace(0, Lx - B, nx).astype(int)
 *
 * Returns one TileRecord per tile, ready for the model.
 */

export interface TileRecord {
  /** (C, B, B) float32 tile data. */
  tile: Float32Array;
  /** Top-left X (inclusive) of this tile in source coordinates. */
  tx: number;
  /** Top-left Y (inclusive). */
  ty: number;
  /** Tile width / height (always equal). */
  bsize: number;
  /** Source image width. */
  srcW: number;
  /** Source image height. */
  srcH: number;
  /** Channel count. */
  channels: number;
}

export interface TileOptions {
  /** Tile size in pixels. CPSAM trains/serves at 256. */
  bsize?: number;
  /** Overlap fraction (clamped to [0.05, 0.5]). Cellpose default 0.1. */
  overlap?: number;
}

/**
 * `np.linspace(0, stop, n).astype(int)` equivalent.
 * Truncates each step toward zero — matches NumPy's behavior for non-negative
 * values.
 */
function linspaceInt(stop: number, n: number): number[] {
  if (n <= 0) return [];
  if (n === 1) return [0];
  const step = stop / (n - 1);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = Math.floor(i * step);
  return out;
}

/**
 * Split a CHW image into a list of (C, B, B) tiles with overlap.
 *
 * @param chw       (C, Ly, Lx) float32 source.
 * @param width     Lx
 * @param height    Ly
 * @param channels  C
 */
export function makeTiles(
  chw: Float32Array,
  width: number,
  height: number,
  channels: number,
  opts: TileOptions = {}
): TileRecord[] {
  const bsize = opts.bsize ?? 256;
  let overlap = opts.overlap ?? 0.1;
  overlap = Math.min(0.5, Math.max(0.05, overlap));

  if (chw.length !== channels * width * height) {
    throw new Error(`makeTiles: expected ${channels * width * height} floats, got ${chw.length}`);
  }

  // If the image is smaller than bsize, return a single zero-padded tile.
  if (width <= bsize && height <= bsize) {
    return [packTile(chw, width, height, channels, 0, 0, bsize, true)];
  }

  const ny = height <= bsize ? 1 : Math.ceil((1 + 2 * overlap) * height / bsize);
  const nx = width  <= bsize ? 1 : Math.ceil((1 + 2 * overlap) * width  / bsize);
  const ystart = linspaceInt(Math.max(0, height - bsize), ny);
  const xstart = linspaceInt(Math.max(0, width  - bsize), nx);

  const tiles: TileRecord[] = [];
  for (const ty of ystart) {
    for (const tx of xstart) {
      tiles.push(packTile(chw, width, height, channels, tx, ty, bsize, false));
    }
  }
  return tiles;
}

/** Extract a (C, bsize, bsize) tile starting at (tx, ty), zero-padded if needed. */
function packTile(
  chw: Float32Array,
  width: number, height: number, channels: number,
  tx: number, ty: number, bsize: number,
  zeroPad: boolean
): TileRecord {
  const tile = new Float32Array(channels * bsize * bsize);
  const hwSrc = width * height;
  const hwTile = bsize * bsize;
  const copyW = Math.min(bsize, width  - tx);
  const copyH = Math.min(bsize, height - ty);
  for (let c = 0; c < channels; c++) {
    const dstChanOff = c * hwTile;
    const srcChanOff = c * hwSrc;
    for (let y = 0; y < copyH; y++) {
      const srcRow = srcChanOff + (ty + y) * width + tx;
      const dstRow = dstChanOff + y * bsize;
      // Float32Array.copyWithin/set both work; .set is the bulk copy.
      tile.set(chw.subarray(srcRow, srcRow + copyW), dstRow);
    }
  }
  void zeroPad; // tile is already zero-initialized → any region not covered remains zero
  return { tile, tx, ty, bsize, srcW: width, srcH: height, channels };
}

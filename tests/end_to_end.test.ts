/**
 * End-to-end test for the postprocess pipeline (tile → average → dynamics).
 *
 * The inference step itself runs on WebGPU and isn't reachable from the
 * Node test runner, so we mock the per-tile model output: we synthesize a
 * flow field that points every pixel toward a known per-cell center, plus
 * a cellprob > threshold for those pixels. The pipeline should recover the
 * cells we planted.
 *
 * This catches integration bugs across averageTiles + computeMasks that
 * the module-level parity tests miss (e.g. wrong channel-order assumptions,
 * tile-coordinate vs full-image-coordinate mistakes).
 */
import { describe, it, expect } from 'vitest';
import { makeTiles } from '../src/preprocess/tile.js';
import { averageTiles, type TileInputForAveraging } from '../src/postprocess/average_tiles.js';
import { computeMasks } from '../src/postprocess/compute_masks.js';
import { instanceIoUs } from './util/iou.js';

interface Cell {
  cy: number;
  cx: number;
  radius: number;
}

/**
 * Synthesize a full-image (3, H, W) flow+cellprob tensor where every pixel
 * inside any cell's radius points toward its center, and cellprob inside
 * cells is 1.0 (0 elsewhere).
 *
 * Layout: [dy(0..hw), dx(hw..2hw), cellprob(2hw..3hw)] in CHW row-major.
 */
function synthesizeFullImageFlows(H: number, W: number, cells: Cell[]): Float32Array {
  const hw = H * W;
  const out = new Float32Array(3 * hw);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      let bestCell: Cell | null = null;
      let bestD = Infinity;
      for (const c of cells) {
        const d = Math.hypot(y - c.cy, x - c.cx);
        if (d <= c.radius && d < bestD) {
          bestCell = c;
          bestD = d;
        }
      }
      if (bestCell !== null) {
        const dy = bestCell.cy - y;
        const dx = bestCell.cx - x;
        const r = Math.hypot(dy, dx) || 1;
        out[i] = dy / r; // unit vector toward center; compute_masks scales by 1/5
        out[hw + i] = dx / r;
        out[2 * hw + i] = 1.0; // cellprob
      }
    }
  }
  return out;
}

/**
 * Slice the full-image (3, H, W) tensor into per-tile (3, B, B) tensors.
 * This mirrors what real inference produces: per-tile outputs in tile-local
 * coordinates, ready for averageTiles to stitch back together.
 */
function sliceFullImageToTiles(
  full: Float32Array,
  H: number,
  W: number,
  bsize: number,
  tileOrigins: Array<{ tx: number; ty: number }>,
): TileInputForAveraging[] {
  const hw = H * W;
  const tileHW = bsize * bsize;
  const out: TileInputForAveraging[] = [];
  for (const origin of tileOrigins) {
    const flowsCellprob = new Float32Array(3 * tileHW);
    for (let c = 0; c < 3; c++) {
      for (let dy = 0; dy < bsize; dy++) {
        const yy = origin.ty + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = 0; dx < bsize; dx++) {
          const xx = origin.tx + dx;
          if (xx < 0 || xx >= W) continue;
          flowsCellprob[c * tileHW + dy * bsize + dx] = full[c * hw + yy * W + xx] as number;
        }
      }
    }
    out.push({ flowsCellprob, tx: origin.tx, ty: origin.ty, bsize });
  }
  return out;
}

/**
 * Reference ground-truth label map: every pixel within a cell's radius gets
 * that cell's 1-indexed label.
 */
function buildGtLabels(H: number, W: number, cells: Cell[]): Uint32Array {
  const out = new Uint32Array(H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      for (let k = 0; k < cells.length; k++) {
        const c = cells[k]!;
        if (Math.hypot(y - c.cy, x - c.cx) <= c.radius) {
          out[y * W + x] = k + 1;
          break;
        }
      }
    }
  }
  return out;
}

describe('postprocess pipeline (averageTiles + computeMasks)', () => {
  it('single tile, 1 synthetic cell → 1 label that covers the cell', () => {
    const H = 96,
      W = 96;
    const cells: Cell[] = [{ cy: 48, cx: 48, radius: 12 }];
    const full = synthesizeFullImageFlows(H, W, cells);
    // No tiling — pass a single full-image "tile".
    const tiles: TileInputForAveraging[] = [
      { flowsCellprob: full, tx: 0, ty: 0, bsize: Math.max(H, W) },
    ];
    // averageTiles expects bsize-sized inputs, so reshape full → (3, bsize, bsize)
    // For this test we just call it with bsize=H=W (square).
    const avg = averageTiles(tiles, H, W);
    const hwFull = H * W;
    const dP = avg.data.subarray(0, 2 * hwFull) as Float32Array;
    const cp = avg.data.subarray(2 * hwFull, 3 * hwFull) as Float32Array;
    const r = computeMasks(dP, cp, H, W, { niter: 300 });
    expect(r.count).toBe(1);
    const gt = buildGtLabels(H, W, cells);
    const { mean } = instanceIoUs(gt, r.masks);
    expect(mean).toBeGreaterThan(0.85);
  });

  it('multi-tile 2x2 grid, 4 cells (one per quadrant) → 4 labels', () => {
    const H = 400,
      W = 400;
    const bsize = 256;
    const cells: Cell[] = [
      { cy: 100, cx: 100, radius: 18 },
      { cy: 100, cx: 300, radius: 18 },
      { cy: 300, cx: 100, radius: 18 },
      { cy: 300, cx: 300, radius: 18 },
    ];
    const full = synthesizeFullImageFlows(H, W, cells);
    // Use makeTiles on a dummy image just to get the tile origins.
    const dummy = new Float32Array(3 * H * W);
    const tileRecs = makeTiles(dummy, W, H, 3, { bsize, overlap: 0.1 });
    const tileInputs = sliceFullImageToTiles(
      full,
      H,
      W,
      bsize,
      tileRecs.map((t) => ({ tx: t.tx, ty: t.ty })),
    );
    const avg = averageTiles(tileInputs, H, W);
    const hwFull = H * W;
    const dP = avg.data.subarray(0, 2 * hwFull) as Float32Array;
    const cp = avg.data.subarray(2 * hwFull, 3 * hwFull) as Float32Array;
    const r = computeMasks(dP, cp, H, W, { niter: 300 });
    expect(r.count).toBe(4);
    const gt = buildGtLabels(H, W, cells);
    const { mean, per } = instanceIoUs(gt, r.masks);
    // Per-cell IoU should be high (each cell is well-isolated).
    expect(per.length).toBe(4);
    for (const iou of per) expect(iou).toBeGreaterThan(0.7);
    expect(mean).toBeGreaterThan(0.8);
  });

  it('cellprobThreshold filters out a cell when raised above its cellprob', () => {
    const H = 96,
      W = 96;
    const cells: Cell[] = [{ cy: 48, cx: 48, radius: 12 }];
    const full = synthesizeFullImageFlows(H, W, cells);
    const hwFull = H * W;
    const dP = full.subarray(0, 2 * hwFull) as Float32Array;
    const cp = full.subarray(2 * hwFull, 3 * hwFull) as Float32Array;
    // Cellprob is 1.0 inside the cell. Set threshold to 1.5 → no pixel passes.
    const r = computeMasks(dP, cp, H, W, { cellprobThreshold: 1.5 });
    expect(r.count).toBe(0);
    for (let i = 0; i < r.masks.length; i++) expect(r.masks[i]).toBe(0);
  });

  it('empty cellprob → zero masks (defensive path)', () => {
    const H = 64,
      W = 64;
    const dP = new Float32Array(2 * H * W);
    const cp = new Float32Array(H * W); // all zero
    const r = computeMasks(dP, cp, H, W);
    expect(r.count).toBe(0);
  });
});

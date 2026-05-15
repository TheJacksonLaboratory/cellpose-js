/**
 * Euler integration of cellpose flow fields, faithful port of
 * cellpose.dynamics.steps_interp.
 *
 * The Python implementation uses PyTorch's grid_sample with align_corners=False
 * but normalizes pixel coordinates as if align_corners=True (divides by N-1
 * rather than N). We mirror that exact quirk for behavioral parity.
 *
 * Algorithm per pixel (one of `inds`):
 *   Let g = 2*p/(N-1) - 1   ∈ [-1, 1]       (normalized coord)
 *   Pre-scale flow: flow_n = flow * 2/(N-1)
 *   For niter steps:
 *     dPt = bilinear_sample(flow_n, g; align_corners=False)
 *     g = clamp(g + dPt, -1, 1)
 *   Final pixel coord = (g + 1) * (N-1) / 2  (round to int for clustering)
 */

export interface FollowFlowsResult {
  /** Final pixel coordinates per seed pixel, length = 2 * n_seeds. Interleaved (y0, x0, y1, x1, ...). */
  pFinal: Int32Array;
  /** Seed pixel y indices in source image. */
  seedY: Int32Array;
  /** Seed pixel x indices in source image. */
  seedX: Int32Array;
}

/**
 * Run flow dynamics.
 *
 * @param dP            Flow field (2, H, W) row-major: dP[0]=dy, dP[1]=dx.
 *                      Caller must pre-multiply by `(cellprob > thresh) / 5` to match Python.
 * @param cellprob      Per-pixel cellprob (H, W) row-major. Used to determine which pixels to integrate.
 * @param H             Image height.
 * @param W             Image width.
 * @param cellprobThreshold Default 0.
 * @param niter         Iteration count. Default 200.
 */
export function followFlows(
  dP: Float32Array,
  cellprob: Float32Array,
  H: number, W: number,
  cellprobThreshold = 0.0,
  niter = 200
): FollowFlowsResult {
  const hw = H * W;
  if (dP.length !== 2 * hw)        throw new Error(`dP wrong size: ${dP.length} vs ${2 * hw}`);
  if (cellprob.length !== hw)      throw new Error(`cellprob wrong size: ${cellprob.length} vs ${hw}`);

  // Collect seed indices.
  const seedYArr: number[] = [];
  const seedXArr: number[] = [];
  for (let y = 0; y < H; y++) {
    const rowOff = y * W;
    for (let x = 0; x < W; x++) {
      if ((cellprob[rowOff + x] as number) > cellprobThreshold) {
        seedYArr.push(y);
        seedXArr.push(x);
      }
    }
  }
  const n = seedYArr.length;
  const seedY = Int32Array.from(seedYArr);
  const seedX = Int32Array.from(seedXArr);
  if (n === 0) return { pFinal: new Int32Array(0), seedY, seedX };

  // Working normalized coordinates: g[i] = (gy_i, gx_i) ∈ [-1, 1].
  // We store as parallel Float32Arrays so the inner loop stays tight.
  const gY = new Float32Array(n);
  const gX = new Float32Array(n);
  const invHm1 = 1 / Math.max(1, H - 1);
  const invWm1 = 1 / Math.max(1, W - 1);
  for (let i = 0; i < n; i++) {
    gY[i] = 2 * (seedY[i] as number) * invHm1 - 1;
    gX[i] = 2 * (seedX[i] as number) * invWm1 - 1;
  }

  // Flow scaling: dP is normalized so one Euler step moves the same fractional
  // distance regardless of image size. cellpose does `im[k] *= 2/(N-1)`.
  // We absorb that scale into the bilinear-sample math rather than mutating dP.
  const fyScale = 2 * invHm1;
  const fxScale = 2 * invWm1;

  // dP[0] = dy field, dP[1] = dx field, row-major.
  const dyField = dP.subarray(0, hw);
  const dxField = dP.subarray(hw, 2 * hw);

  // grid_sample(align_corners=False) inverse mapping:
  //   normalized coord g maps to pixel coord u = (g + 1) * N / 2 - 0.5
  // Then bilinear sample u in [-0.5, N-0.5], with zero-padding outside [0, N-1].
  const sampleAt = (gy: number, gx: number, out: { dy: number; dx: number }): void => {
    const uy = (gy + 1) * H * 0.5 - 0.5;
    const ux = (gx + 1) * W * 0.5 - 0.5;
    const y0 = Math.floor(uy);
    const x0 = Math.floor(ux);
    const fy = uy - y0;
    const fx = ux - x0;
    let dy = 0, dx = 0;
    // Four corners with bounds checks (zero padding outside).
    for (let by = 0; by < 2; by++) {
      const yy = y0 + by;
      if (yy < 0 || yy >= H) continue;
      const wy = by === 0 ? 1 - fy : fy;
      const yOff = yy * W;
      for (let bx = 0; bx < 2; bx++) {
        const xx = x0 + bx;
        if (xx < 0 || xx >= W) continue;
        const wx = bx === 0 ? 1 - fx : fx;
        const w = wy * wx;
        dy += w * (dyField[yOff + xx] as number);
        dx += w * (dxField[yOff + xx] as number);
      }
    }
    out.dy = dy * fyScale;
    out.dx = dx * fxScale;
  };

  const tmp = { dy: 0, dx: 0 };
  for (let t = 0; t < niter; t++) {
    for (let i = 0; i < n; i++) {
      sampleAt(gY[i] as number, gX[i] as number, tmp);
      let ny = (gY[i] as number) + tmp.dy;
      let nx = (gX[i] as number) + tmp.dx;
      if (ny >  1) ny =  1; else if (ny < -1) ny = -1;
      if (nx >  1) nx =  1; else if (nx < -1) nx = -1;
      gY[i] = ny;
      gX[i] = nx;
    }
  }

  // Un-normalize and round to int.
  const pFinal = new Int32Array(2 * n);
  for (let i = 0; i < n; i++) {
    pFinal[2 * i]     = Math.round(((gY[i] as number) + 1) * 0.5 * (H - 1));
    pFinal[2 * i + 1] = Math.round(((gX[i] as number) + 1) * 0.5 * (W - 1));
  }
  return { pFinal, seedY, seedX };
}

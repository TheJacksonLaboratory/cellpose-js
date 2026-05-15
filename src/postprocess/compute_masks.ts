/**
 * Top-level dynamics orchestrator: applies cellprob threshold, scales flows
 * by 1/5 (matches cellpose convention), runs Euler integration, then clusters.
 *
 * Mirrors cellpose.dynamics.compute_masks for the 2D path, minus the
 * (optional) flow-consistency filter (M4 follow-up) and hole-fill / min-size
 * cleanup (M4 follow-up).
 */
import { followFlows } from './follow_flows.js';
import { getMasks, type GetMasksResult } from './get_masks.js';

export interface ComputeMasksOptions {
  /** Cellprob threshold; pixels above this enter the dynamical system. Default 0. */
  cellprobThreshold?: number;
  /** Euler iteration count. Default 200. */
  niter?: number;
  /** Maximum fraction of image a single mask may cover. Default 0.4. */
  maxSizeFraction?: number;
}

/**
 * @param dP        Float32 (2, H, W) row-major; dP[0]=dy, dP[1]=dx.
 * @param cellprob  Float32 (H, W) row-major.
 * @param H         Image height.
 * @param W         Image width.
 */
export function computeMasks(
  dP: Float32Array,
  cellprob: Float32Array,
  H: number,
  W: number,
  opts: ComputeMasksOptions = {},
): GetMasksResult {
  const { cellprobThreshold = 0, niter = 200, maxSizeFraction = 0.4 } = opts;
  const hw = H * W;

  // Cellpose pre-scales: dP * (cellprob > thresh) / 5. Pre-multiply once.
  const dPScaled = new Float32Array(2 * hw);
  for (let i = 0; i < hw; i++) {
    const on = (cellprob[i] as number) > cellprobThreshold ? 1 : 0;
    dPScaled[i] = ((dP[i] as number) * on) / 5;
    dPScaled[hw + i] = ((dP[hw + i] as number) * on) / 5;
  }

  const { pFinal, seedY, seedX } = followFlows(dPScaled, cellprob, H, W, cellprobThreshold, niter);
  return getMasks(pFinal, seedY, seedX, H, W, maxSizeFraction);
}

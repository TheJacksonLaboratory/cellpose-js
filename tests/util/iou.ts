/**
 * IoU helpers for parity tests.
 *
 * Given two integer label maps of the same shape, for each ground-truth label
 * find the predicted label with maximum overlap, and compute IoU between them.
 * Mean of those IoUs is the per-instance match quality.
 */

export function instanceIoUs(
  gt: Uint16Array | Uint32Array,
  pred: Uint16Array | Uint32Array
): { mean: number; per: number[] } {
  if (gt.length !== pred.length) {
    throw new Error(`label map length mismatch: ${gt.length} vs ${pred.length}`);
  }
  // Collect label sets
  const gtLabels = new Set<number>();
  const predLabels = new Set<number>();
  for (let i = 0; i < gt.length; i++) {
    if ((gt[i] as number) > 0) gtLabels.add(gt[i] as number);
    if ((pred[i] as number) > 0) predLabels.add(pred[i] as number);
  }
  // Both empty → perfect.
  if (gtLabels.size === 0 && predLabels.size === 0) return { mean: 1.0, per: [] };
  // GT empty but pred non-empty → score 0 for each pred (false positive).
  if (gtLabels.size === 0) return { mean: 0, per: Array(predLabels.size).fill(0) };

  // For each GT label, build {predLabel: overlapCount}, then compute IoU.
  const gtSize: Record<number, number> = {};
  const predSize: Record<number, number> = {};
  for (let i = 0; i < gt.length; i++) {
    const g = gt[i] as number;
    const p = pred[i] as number;
    if (g > 0) gtSize[g] = (gtSize[g] ?? 0) + 1;
    if (p > 0) predSize[p] = (predSize[p] ?? 0) + 1;
  }
  const per: number[] = [];
  for (const g of gtLabels) {
    const overlaps: Record<number, number> = {};
    for (let i = 0; i < gt.length; i++) {
      if ((gt[i] as number) !== g) continue;
      const p = pred[i] as number;
      if (p > 0) overlaps[p] = (overlaps[p] ?? 0) + 1;
    }
    let bestIou = 0;
    for (const [pStr, overlap] of Object.entries(overlaps)) {
      const p = parseInt(pStr, 10);
      const union = (gtSize[g] ?? 0) + (predSize[p] ?? 0) - overlap;
      const iou = union > 0 ? overlap / union : 0;
      if (iou > bestIou) bestIou = iou;
    }
    per.push(bestIou);
  }
  const mean = per.reduce((a, b) => a + b, 0) / per.length;
  return { mean, per };
}

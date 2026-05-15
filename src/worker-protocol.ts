/**
 * Message contract between the public Cellpose API (main thread) and the
 * inference worker. Discriminated unions on `type` for both directions.
 *
 * Transferables policy:
 *   - `MainToWorker.init.modelBytes` is transferred (worker takes ownership).
 *   - `MainToWorker.runTile.tile` (Float32Array) — the underlying ArrayBuffer
 *     is transferred so the worker reads it without a copy.
 *   - `WorkerToMain.tileResult.flowsCellprob` — same: the worker transfers
 *     the FP32 output back so the main thread can stitch / postprocess
 *     without copying.
 */

export type MainToWorker =
  | { type: 'init'; modelBytes: ArrayBuffer; wasmPaths?: string }
  | { type: 'run-tile'; tileId: number; tile: Float32Array; bsize: number }
  | { type: 'dispose' };

export type WorkerToMain =
  | { type: 'ready'; adapterInfo: { vendor: string; architecture: string; device: string } | null }
  | { type: 'tile-result'; tileId: number; flowsCellprob: Float32Array; inferenceMs: number }
  | { type: 'error'; tileId: number | null; message: string };

/**
 * Message contract between the public Cellpose API (main thread) and the
 * inference worker. Discriminated unions on `type` for both directions.
 *
 * Two segmentation paths:
 *   - `segment` runs the WHOLE pipeline (preprocess → tile → inference →
 *     average → dynamics → inverse-resize) in the worker and returns the final
 *     label map. Nothing heavy runs on the main thread, so the UI never blocks.
 *   - `run-tile` runs a single tile's inference (legacy; the main thread did the
 *     orchestration). Kept for back-compat.
 *
 * Transferables policy:
 *   - `init.modelBytes` is transferred (worker takes ownership).
 *   - `run-tile.tile` / `tile-result.flowsCellprob` ArrayBuffers are transferred.
 *   - `segment.image.data` is cloned (left intact for the caller); the returned
 *     `segment-result.masks` is transferred back (zero-copy).
 */
import type { ChannelMapOptions } from './preprocess/channels.js';
import type { NormalizeOptions } from './preprocess/normalize.js';
import type { ComputeMasksOptions } from './postprocess/compute_masks.js';

/** Serializable subset of SegmentOptions safe to structured-clone to the worker
 *  (no functions, no AbortSignal). */
export interface WorkerSegmentOptions extends ChannelMapOptions {
  diameter?: number;
  tile?: number;
  overlap?: number;
  normalize?: NormalizeOptions;
  dynamics?: ComputeMasksOptions;
}

/** Per-tile timing diagnostics returned with a full-segment result. The flow
 *  tensors stay in the worker — returning them would defeat the offload. */
export interface WorkerTileDiag {
  tx: number;
  ty: number;
  bsize: number;
  inferenceMs: number;
}

export type MainToWorker =
  | { type: 'init'; modelBytes: ArrayBuffer; wasmPaths?: string }
  | { type: 'run-tile'; tileId: number; tile: Float32Array; bsize: number }
  | {
      type: 'segment';
      reqId: number;
      image: {
        data: Uint8ClampedArray | Uint8Array | Float32Array;
        width: number;
        height: number;
        channels: number;
      };
      opts: WorkerSegmentOptions;
    }
  | { type: 'dispose' };

export type WorkerToMain =
  | { type: 'ready'; adapterInfo: { vendor: string; architecture: string; device: string } | null }
  | { type: 'tile-result'; tileId: number; flowsCellprob: Float32Array; inferenceMs: number }
  | { type: 'error'; tileId: number | null; message: string }
  // Pre-ready progress strings: 'configuring ORT', 'creating session',
  // 'session created', 'describing adapter'. Optional; consumers wire via
  // `FromPretrainedOptions.onStatus`.
  | { type: 'status'; status: string }
  // Full-pipeline segment progress: per-tile inference, then the dynamics phase.
  | { type: 'segment-progress'; reqId: number; phase: 'inference' | 'dynamics'; done: number; total: number }
  | {
      type: 'segment-result';
      reqId: number;
      masks: Uint32Array;
      count: number;
      width: number;
      height: number;
      resizedWidth: number;
      resizedHeight: number;
      scale: number;
      tiles: WorkerTileDiag[];
      postprocessMs: number;
    }
  | { type: 'segment-error'; reqId: number; message: string };

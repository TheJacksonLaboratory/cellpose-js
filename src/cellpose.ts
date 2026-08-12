/**
 * Public Cellpose API. The full segmentation pipeline (preprocess → tile →
 * inference → average → dynamics → inverse-resize) runs in a Web Worker via the
 * `segment` message, so nothing heavy runs on the UI thread. AbortSignal-driven
 * cancellation and tile-level progress are supported.
 */
import { assertSupportedEnvironment, describeAdapter } from './env.js';
import { fetchModel, type FetchProgress } from './model-cache.js';
import { configureOrt, _getWasmPaths } from './session.js';
import { type ChannelMapOptions, type NormalizeOptions } from './preprocess/index.js';
import { type ComputeMasksOptions } from './postprocess/index.js';
import type { MainToWorker, WorkerToMain, WorkerSegmentOptions } from './worker-protocol.js';

export interface FromPretrainedOptions {
  /** Eagerly create the inference worker + ORT session at construct time. */
  preload?: boolean;
  /** Override ORT's WASM helper path (must be same-origin for dynamic .mjs imports). */
  wasmPaths?: string;
  /** Forwarded to the model fetcher. */
  onProgress?: (p: FetchProgress) => void;
  bypassCache?: boolean;
  /** Aborts the in-flight fetch AND any preload worker init. */
  signal?: AbortSignal;
  /** Optional callback for worker-init phase strings ('spawning worker',
   *  'creating ORT session', 'session ready'). Useful for showing a determinate
   *  status while the 588 MB FP16 model is being parsed and the WebGPU adapter
   *  is being initialized. */
  onStatus?: (status: string) => void;
}

export interface SegmentTileOutput {
  flows_cellprob: Float32Array;
  tx: number;
  ty: number;
  bsize: number;
  inferenceMs: number;
}

export interface SegmentInput {
  data: Uint8ClampedArray | Uint8Array | Float32Array;
  width: number;
  height: number;
  channels: number;
}

export interface SegmentOptions extends ChannelMapOptions {
  diameter?: number;
  /** Tile size fed to the network. CPSAM only supports 256 — its position
   *  embeddings and the ONNX export are both fixed at that size. Default 256. */
  tile?: number;
  overlap?: number;
  /**
   * Where the flow-dynamics step runs when `diameter` triggers a resize.
   *
   * - `false` (default): dynamics run at the resized (network) resolution and
   *   the label map is upscaled nearest-neighbor. Cheapest — the dynamical
   *   system only ever sees the downscaled image.
   * - `true`: the predicted flow field and cellprob are bilinear-upsampled back
   *   to source resolution first, and dynamics run there with a proportionally
   *   larger iteration count. This is upstream Cellpose's default
   *   (`models.py:_run_net`, `resample=True`); mask boundaries follow the flows
   *   instead of a blocky upscale, at the cost of running dynamics over the
   *   full-resolution image.
   *
   * No effect when `diameter` is omitted (no resize happens either way).
   */
  resample?: boolean;
  normalize?: NormalizeOptions;
  /** Dynamics postprocessing knobs. */
  dynamics?: ComputeMasksOptions;
  /** Fires after each tile finishes inference. */
  onTileProgress?: (done: number, total: number) => void;
  /** Abort the in-flight call. Terminates the worker; next call respawns. */
  signal?: AbortSignal;
}

export interface SegmentOutput {
  /** Full-image instance label map at SOURCE-image resolution (after inverse
   *  resize). 0 = background. Row-major Uint32Array of length sourceW * sourceH. */
  masks: Uint32Array;
  /** Number of distinct instances in `masks`. */
  count: number;
  /** Width / height at source resolution. */
  width: number;
  height: number;
  /** Per-tile diagnostics (timing). Flow tensors stay in the worker, so
   *  `flows_cellprob` is empty here — returning them would defeat the offload. */
  tiles: SegmentTileOutput[];
  /** Resized image dimensions (intermediate; pre-inverse-resize). */
  resizedWidth: number;
  resizedHeight: number;
  /** Scale factor applied (resized = source * scale). */
  scale: number;
  /** Total wall-clock ms for the full segment() call. */
  totalMs: number;
  /** Wall-clock ms in the average+dynamics step (excludes per-tile inference). */
  postprocessMs: number;
}

interface PendingTile {
  resolve: (msg: Extract<WorkerToMain, { type: 'tile-result' }>) => void;
  reject: (err: Error) => void;
}

interface PendingSegment {
  resolve: (out: SegmentOutput) => void;
  reject: (err: Error) => void;
  onTileProgress?: (done: number, total: number) => void;
  t0: number;
}

export class Cellpose {
  private _worker: Worker | null = null;
  private _workerReady: Promise<void> | null = null;
  private _adapterInfo: { vendor: string; architecture: string; device: string } | null = null;
  private _pending = new Map<number, PendingTile>();
  private _nextReqId = 0;
  private _pendingSegments = new Map<number, PendingSegment>();

  // _modelBytes is detached after the worker takes ownership of it. To respawn
  // after _abort() we re-fetch via fetchModel (cache hit -> instant).
  private _modelBytes: ArrayBuffer | null;
  private constructor(
    modelBytes: ArrayBuffer,
    private readonly _modelUrl: string,
  ) {
    this._modelBytes = modelBytes;
  }

  static async fromPretrained(
    modelUrl: string,
    opts: FromPretrainedOptions = {},
  ): Promise<Cellpose> {
    assertSupportedEnvironment();
    const fetchOpts = {
      ...(opts.onProgress !== undefined && { onProgress: opts.onProgress }),
      ...(opts.bypassCache !== undefined && { bypassCache: opts.bypassCache }),
      ...(opts.signal !== undefined && { signal: opts.signal }),
    };
    const bytes = await fetchModel(modelUrl, fetchOpts);
    if (opts.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    if (opts.wasmPaths) configureOrt({ wasmPaths: opts.wasmPaths });
    const cp = new Cellpose(bytes, modelUrl);
    if (opts.preload) {
      const workerOpts: { signal?: AbortSignal; onStatus?: (s: string) => void } = {};
      if (opts.signal !== undefined) workerOpts.signal = opts.signal;
      if (opts.onStatus !== undefined) workerOpts.onStatus = opts.onStatus;
      await cp._ensureWorker(workerOpts);
    }
    return cp;
  }

  async describeAdapter(): Promise<{
    vendor: string;
    architecture: string;
    device: string;
  } | null> {
    if (this._adapterInfo) return this._adapterInfo;
    return describeAdapter();
  }

  /** Lazy spawn + init. Idempotent. */
  private _ensureWorker(
    opts: { signal?: AbortSignal; onStatus?: (s: string) => void } = {},
  ): Promise<void> {
    if (this._workerReady) return this._workerReady;
    if (opts.signal?.aborted) {
      return Promise.reject(new DOMException('Aborted before worker init', 'AbortError'));
    }

    opts.onStatus?.('spawning worker');
    const worker = new Worker(new URL('./inference.worker.js', import.meta.url), {
      type: 'module',
    });
    this._worker = worker;
    // Track whether the worker has finished init. Before this flips true,
    // a worker `error` event has nowhere to be reported (pending queues are
    // empty), so we route it to the init promise's reject instead. The
    // ref is updated to a no-op once init completes.
    let rejectInit: (err: Error) => void = () => {};
    let initSettled = false;

    worker.addEventListener('message', (ev: MessageEvent<WorkerToMain>) => {
      const msg = ev.data;
      if (msg.type === 'tile-result') {
        const p = this._pending.get(msg.tileId);
        if (p) {
          this._pending.delete(msg.tileId);
          p.resolve(msg);
        }
      } else if (msg.type === 'error') {
        if (msg.tileId !== null) {
          const p = this._pending.get(msg.tileId);
          if (p) {
            this._pending.delete(msg.tileId);
            p.reject(new Error(msg.message));
          }
        } else if (!initSettled) {
          rejectInit(new Error(msg.message));
        }
      } else if (msg.type === 'segment-progress') {
        // Only per-tile inference maps onto the public onTileProgress callback;
        // the dynamics phase is signalled by the final (done===total) tile tick.
        if (msg.phase === 'inference') {
          this._pendingSegments.get(msg.reqId)?.onTileProgress?.(msg.done, msg.total);
        }
      } else if (msg.type === 'segment-result') {
        const p = this._pendingSegments.get(msg.reqId);
        if (p) {
          this._pendingSegments.delete(msg.reqId);
          p.resolve({
            masks: msg.masks,
            count: msg.count,
            width: msg.width,
            height: msg.height,
            tiles: msg.tiles.map((t) => ({ ...t, flows_cellprob: new Float32Array(0) })),
            resizedWidth: msg.resizedWidth,
            resizedHeight: msg.resizedHeight,
            scale: msg.scale,
            totalMs: performance.now() - p.t0,
            postprocessMs: msg.postprocessMs,
          });
        }
      } else if (msg.type === 'segment-error') {
        const p = this._pendingSegments.get(msg.reqId);
        if (p) {
          this._pendingSegments.delete(msg.reqId);
          p.reject(new Error(msg.message));
        }
      } else if (msg.type === 'status') {
        opts.onStatus?.(msg.status);
      }
    });
    // The Worker.onerror event fires when the worker module fails to load
    // or throws at top level. `ev.message` carries the underlying Error.
    worker.addEventListener('error', (ev) => {
      const detail =
        ev.message ||
        (ev.filename ? `worker error in ${ev.filename}:${ev.lineno}` : 'worker error (no detail)');
      if (!initSettled) {
        rejectInit(new Error(`worker module failed to load: ${detail}`));
      }
      this._rejectAllPending(new Error(detail));
    });
    // messageerror fires when structured-clone deserialization fails on a
    // message *received* by the worker (the failure is reported on the main
    // side). Surface it the same way as error.
    worker.addEventListener('messageerror', () => {
      const err = new Error('worker received malformed message (structured-clone failure)');
      if (!initSettled) rejectInit(err);
      this._rejectAllPending(err);
    });

    this._workerReady = new Promise<void>((resolve, reject) => {
      let abortListener: (() => void) | null = null;
      const cleanup = (): void => {
        initSettled = true;
        if (abortListener && opts.signal) {
          opts.signal.removeEventListener('abort', abortListener);
        }
        worker.removeEventListener('message', onReady);
      };
      rejectInit = (err) => {
        cleanup();
        reject(err);
      };
      const onReady = (ev: MessageEvent<WorkerToMain>) => {
        if (ev.data.type === 'ready') {
          this._adapterInfo = ev.data.adapterInfo;
          cleanup();
          resolve();
        } else if (ev.data.type === 'error' && ev.data.tileId === null) {
          cleanup();
          reject(new Error(ev.data.message));
        }
      };
      worker.addEventListener('message', onReady);

      // Abort handling: terminate the worker and reject. Honors caller abort
      // while ORT is doing its (potentially slow) WebGPU + WASM-sidecar init.
      if (opts.signal) {
        abortListener = () => {
          cleanup();
          this._worker?.terminate();
          this._worker = null;
          this._workerReady = null;
          reject(new DOMException('Aborted during worker init', 'AbortError'));
        };
        opts.signal.addEventListener('abort', abortListener);
      }

      // Resolve model bytes. If the previous worker took ownership (transfer
      // detached the buffer), refetch from IDB cache — typically <100 ms.
      const ensureBytes = async (): Promise<ArrayBuffer> => {
        if (this._modelBytes && this._modelBytes.byteLength > 0) return this._modelBytes;
        const bytes = await fetchModel(this._modelUrl);
        this._modelBytes = bytes;
        return bytes;
      };
      ensureBytes()
        .then((bytes) => {
          if (opts.signal?.aborted) return; // abort fired during fetch; listener already rejected
          this._modelBytes = null; // drop our ref since we're about to transfer ownership
          opts.onStatus?.('posting model to worker');
          const init: MainToWorker = {
            type: 'init',
            modelBytes: bytes,
            wasmPaths: _getWasmPaths(),
          };
          worker.postMessage(init, [bytes]);
        })
        .catch((err) => {
          cleanup();
          reject(err);
        });
    });
    return this._workerReady;
  }

  /** Reject every in-flight tile + segment promise (worker died/errored). */
  private _rejectAllPending(err: Error): void {
    for (const p of this._pending.values()) p.reject(err);
    this._pending.clear();
    for (const p of this._pendingSegments.values()) p.reject(err);
    this._pendingSegments.clear();
  }

  /** Aborts the worker mid-run; pending promises reject with AbortError. */
  private _abort(reason?: string): void {
    if (!this._worker) return;
    this._worker.terminate();
    this._worker = null;
    this._workerReady = null;
    this._rejectAllPending(new DOMException(reason ?? 'Operation aborted', 'AbortError'));
  }

  async segment(input: SegmentInput, opts: SegmentOptions = {}): Promise<SegmentOutput> {
    const t0 = performance.now();
    const signal = opts.signal;

    // Hook abort: terminate the worker, surface AbortError to the caller.
    let abortListener: (() => void) | null = null;
    if (signal) {
      if (signal.aborted) throw new DOMException('Aborted before start', 'AbortError');
      abortListener = () =>
        this._abort(signal.reason instanceof Error ? signal.reason.message : undefined);
      signal.addEventListener('abort', abortListener);
    }

    try {
      await this._ensureWorker();
      const worker = this._worker;
      if (!worker) throw new Error('worker not available');

      // Strip the non-serializable opts (AbortSignal + callbacks); the rest is
      // structured-cloned to the worker.
      const { signal: _omitSignal, onTileProgress, ...rest } = opts;
      void _omitSignal;
      const workerOpts = rest as WorkerSegmentOptions;
      const reqId = this._nextReqId++;

      return await new Promise<SegmentOutput>((resolve, reject) => {
        const pending: PendingSegment = { resolve, reject, t0 };
        if (onTileProgress) pending.onTileProgress = onTileProgress;
        this._pendingSegments.set(reqId, pending);
        const msg: MainToWorker = {
          type: 'segment',
          reqId,
          image: {
            data: input.data,
            width: input.width,
            height: input.height,
            channels: input.channels,
          },
          opts: workerOpts,
        };
        // image.data is cloned (left intact for the caller); the result masks
        // are transferred back from the worker.
        worker.postMessage(msg);
      });
    } finally {
      if (signal && abortListener) signal.removeEventListener('abort', abortListener);
    }
  }

  async dispose(): Promise<void> {
    if (!this._worker) return;
    this._worker.postMessage({ type: 'dispose' } satisfies MainToWorker);
    this._worker.terminate();
    this._worker = null;
    this._workerReady = null;
    this._pending.clear();
    this._pendingSegments.clear();
  }
}

/**
 * Public Cellpose API. Milestone 3: inference offloaded to a Web Worker,
 * AbortSignal-driven cancellation, tile-level progress.
 */
import { assertSupportedEnvironment, describeAdapter } from './env.js';
import { fetchModel, type FetchProgress } from './model-cache.js';
import { configureOrt, _getWasmPaths } from './session.js';
import {
  buildCpsamChannels, type ChannelMapOptions,
  diameterResize,
  normalizePerChannel, type NormalizeOptions,
  makeTiles, type TileRecord,
} from './preprocess/index.js';
import { computeMasks, type ComputeMasksOptions, averageTiles } from './postprocess/index.js';
import type { MainToWorker, WorkerToMain } from './worker-protocol.js';

export interface FromPretrainedOptions {
  /** Eagerly create the inference worker + ORT session at construct time. */
  preload?: boolean;
  /** Override ORT's WASM helper path (must be same-origin for dynamic .mjs imports). */
  wasmPaths?: string;
  /** Forwarded to the model fetcher. */
  onProgress?: (p: FetchProgress) => void;
  bypassCache?: boolean;
  signal?: AbortSignal;
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
  tile?: number;
  overlap?: number;
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
  /** Per-tile diagnostics. Per-tile masks are NOT included (single global label
   *  map is the v1 contract). */
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

/** @deprecated kept for legacy demo; use `segment()`. */
export interface SegmentMilestone1Output {
  flows_cellprob: Float32Array;
  height: number;
  width: number;
  inferenceMs: number;
}

interface PendingTile {
  resolve: (msg: Extract<WorkerToMain, { type: 'tile-result' }>) => void;
  reject: (err: Error) => void;
}

export class Cellpose {
  private _worker: Worker | null = null;
  private _workerReady: Promise<void> | null = null;
  private _adapterInfo: { vendor: string; architecture: string; device: string } | null = null;
  private _nextTileId = 0;
  private _pending = new Map<number, PendingTile>();

  // _modelBytes is detached after the worker takes ownership of it. To respawn
  // after _abort() we re-fetch via fetchModel (cache hit -> instant).
  private _modelBytes: ArrayBuffer | null;
  private constructor(
    modelBytes: ArrayBuffer,
    private readonly _modelUrl: string,
  ) {
    this._modelBytes = modelBytes;
  }

  static async fromPretrained(modelUrl: string, opts: FromPretrainedOptions = {}): Promise<Cellpose> {
    assertSupportedEnvironment();
    const fetchOpts = {
      ...(opts.onProgress  !== undefined && { onProgress: opts.onProgress }),
      ...(opts.bypassCache !== undefined && { bypassCache: opts.bypassCache }),
      ...(opts.signal      !== undefined && { signal: opts.signal }),
    };
    const bytes = await fetchModel(modelUrl, fetchOpts);
    if (opts.wasmPaths) configureOrt({ wasmPaths: opts.wasmPaths });
    const cp = new Cellpose(bytes, modelUrl);
    if (opts.preload) await cp._ensureWorker();
    return cp;
  }

  async describeAdapter(): Promise<{ vendor: string; architecture: string; device: string } | null> {
    if (this._adapterInfo) return this._adapterInfo;
    return describeAdapter();
  }

  /** Lazy spawn + init. Idempotent. */
  private _ensureWorker(): Promise<void> {
    if (this._workerReady) return this._workerReady;

    const worker = new Worker(
      new URL('./inference.worker.ts', import.meta.url),
      { type: 'module' }
    );
    this._worker = worker;

    worker.addEventListener('message', (ev: MessageEvent<WorkerToMain>) => {
      const msg = ev.data;
      if (msg.type === 'tile-result') {
        const p = this._pending.get(msg.tileId);
        if (p) { this._pending.delete(msg.tileId); p.resolve(msg); }
      } else if (msg.type === 'error') {
        if (msg.tileId !== null) {
          const p = this._pending.get(msg.tileId);
          if (p) { this._pending.delete(msg.tileId); p.reject(new Error(msg.message)); }
        }
      }
    });
    worker.addEventListener('error', (ev) => {
      const err = new Error(ev.message || 'worker error');
      for (const p of this._pending.values()) p.reject(err);
      this._pending.clear();
    });

    this._workerReady = new Promise<void>((resolve, reject) => {
      const onReady = (ev: MessageEvent<WorkerToMain>) => {
        if (ev.data.type === 'ready') {
          this._adapterInfo = ev.data.adapterInfo;
          worker.removeEventListener('message', onReady);
          resolve();
        } else if (ev.data.type === 'error' && ev.data.tileId === null) {
          worker.removeEventListener('message', onReady);
          reject(new Error(ev.data.message));
        }
      };
      worker.addEventListener('message', onReady);

      // Resolve model bytes. If the previous worker took ownership (transfer
      // detached the buffer), refetch from IDB cache — typically <100 ms.
      const ensureBytes = async (): Promise<ArrayBuffer> => {
        if (this._modelBytes && this._modelBytes.byteLength > 0) return this._modelBytes;
        const bytes = await fetchModel(this._modelUrl);
        this._modelBytes = bytes;
        return bytes;
      };
      ensureBytes().then((bytes) => {
        this._modelBytes = null; // drop our ref since we're about to transfer ownership
        const init: MainToWorker = { type: 'init', modelBytes: bytes, wasmPaths: _getWasmPaths() };
        worker.postMessage(init, [bytes]);
      }).catch(reject);
    });
    return this._workerReady;
  }

  /** Aborts the worker mid-run; pending tile promises reject with AbortError. */
  private _abort(reason?: string): void {
    if (!this._worker) return;
    this._worker.terminate();
    this._worker = null;
    this._workerReady = null;
    const err = new DOMException(reason ?? 'Operation aborted', 'AbortError');
    for (const p of this._pending.values()) p.reject(err);
    this._pending.clear();
  }

  private async _runTile(tile: Float32Array, bsize: number): Promise<Extract<WorkerToMain, { type: 'tile-result' }>> {
    await this._ensureWorker();
    const worker = this._worker;
    if (!worker) throw new Error('worker not available');
    const tileId = this._nextTileId++;
    return new Promise<Extract<WorkerToMain, { type: 'tile-result' }>>((resolve, reject) => {
      this._pending.set(tileId, { resolve, reject });
      const msg: MainToWorker = { type: 'run-tile', tileId, tile, bsize };
      worker.postMessage(msg, [tile.buffer]);
    });
  }

  async segment(input: SegmentInput, opts: SegmentOptions = {}): Promise<SegmentOutput> {
    const t0 = performance.now();
    const tileSize = opts.tile ?? 256;
    const signal = opts.signal;

    // Hook abort: terminate the worker, surface AbortError to the caller.
    let abortListener: (() => void) | null = null;
    if (signal) {
      if (signal.aborted) throw new DOMException('Aborted before start', 'AbortError');
      abortListener = () => this._abort(signal.reason instanceof Error ? signal.reason.message : undefined);
      signal.addEventListener('abort', abortListener);
    }

    try {
      let chw = buildCpsamChannels(input.data, input.width, input.height, input.channels, opts);
      let w = input.width, h = input.height, scale = 1;
      if (opts.diameter !== undefined) {
        const r = diameterResize(chw, w, h, { channels: 3, diameter: opts.diameter });
        chw = r.data; w = r.width; h = r.height; scale = r.scale;
      }
      chw = normalizePerChannel(chw, 3, w * h, opts.normalize ?? {});

      const tileOpts: { bsize: number; overlap?: number } = { bsize: tileSize };
      if (opts.overlap !== undefined) tileOpts.overlap = opts.overlap;
      const tiles: TileRecord[] = makeTiles(chw, w, h, 3, tileOpts);

      const out: SegmentTileOutput[] = [];
      for (let i = 0; i < tiles.length; i++) {
        const t = tiles[i]!;
        const r = await this._runTile(t.tile, tileSize);
        out.push({
          flows_cellprob: r.flowsCellprob,
          tx: t.tx, ty: t.ty, bsize: tileSize,
          inferenceMs: r.inferenceMs,
        });
        opts.onTileProgress?.(i + 1, tiles.length);
      }

      // Average overlapping tile predictions into a single full-image tensor,
      // then run dynamics once. See average_tiles.ts header for why this is the
      // right algorithm choice (vs per-tile dynamics + label stitching).
      const tPost = performance.now();
      const averaged = averageTiles(
        out.map(o => ({ flowsCellprob: o.flows_cellprob, tx: o.tx, ty: o.ty, bsize: o.bsize })),
        h, w,
      );
      const hwFull = h * w;
      const dPFull = averaged.data.subarray(0, 2 * hwFull) as Float32Array;
      const cpFull = averaged.data.subarray(2 * hwFull, 3 * hwFull) as Float32Array;
      const m = computeMasks(dPFull, cpFull, h, w, opts.dynamics ?? {});

      // Inverse-resize labels back to source resolution if a diameter resize
      // was applied (nearest-neighbor — labels must not be interpolated).
      let masksSrc = m.masks;
      let outW = w, outH = h;
      if (scale !== 1) {
        outW = input.width;
        outH = input.height;
        const resized = new Uint32Array(outW * outH);
        for (let yy = 0; yy < outH; yy++) {
          const sy = Math.min(h - 1, Math.max(0, Math.round(yy * scale)));
          const srcRow = sy * w;
          const dstRow = yy * outW;
          for (let xx = 0; xx < outW; xx++) {
            const sx = Math.min(w - 1, Math.max(0, Math.round(xx * scale)));
            resized[dstRow + xx] = m.masks[srcRow + sx] as number;
          }
        }
        masksSrc = resized;
      }
      const postprocessMs = performance.now() - tPost;
      return {
        masks: masksSrc,
        count: m.count,
        width: outW,
        height: outH,
        tiles: out,
        resizedWidth: w,
        resizedHeight: h,
        scale,
        totalMs: performance.now() - t0,
        postprocessMs,
      };
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
  }
}

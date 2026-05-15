/**
 * Public Cellpose API. Milestone 1: model loading + identity forward pass.
 * Pre/postprocessing (Milestone 2+) and dynamics (Milestone 4) live in
 * sibling modules added later.
 */
import * as ort from 'onnxruntime-web/webgpu';
import { assertSupportedEnvironment, describeAdapter } from './env.js';
import { fetchModel, type FetchProgress } from './model-cache.js';
import { CellposeSession, type SessionOptions } from './session.js';
import {
  buildCpsamChannels, type ChannelMapOptions,
  diameterResize,
  normalizePerChannel, type NormalizeOptions,
  makeTiles, type TileRecord,
} from './preprocess/index.js';

export interface FromPretrainedOptions extends SessionOptions {
  /** Eagerly create the ORT session at construct time (recommended).
   *  Trades ~3.5 s of latency now for a fast first segment() call. */
  preload?: boolean;
  /** Forwarded to the model fetcher. */
  onProgress?: (p: FetchProgress) => void;
  bypassCache?: boolean;
  signal?: AbortSignal;
}

/**
 * Per-tile inference output. Milestone 4 will collapse these into a single
 * instance label map; for M2 we expose the per-tile flows for inspection.
 */
export interface SegmentTileOutput {
  /** Raw model output `flows_cellprob` for this tile, shape (1, 3, bsize, bsize), FP32. */
  flows_cellprob: Float32Array;
  /** Tile origin in the resized image's pixel coordinates. */
  tx: number;
  ty: number;
  /** Tile size (square, matches model input). */
  bsize: number;
  /** Wall-clock ms in this tile's sess.run. */
  inferenceMs: number;
}

export interface SegmentInput {
  /** Pixel-interleaved source data. RGBA (length = 4*w*h), RGB (3*w*h), or grayscale (w*h). */
  data: Uint8ClampedArray | Uint8Array | Float32Array;
  width: number;
  height: number;
  /** Channel count of `data`. 1, 3, or 4. */
  channels: number;
}

export interface SegmentOptions extends ChannelMapOptions {
  /** Estimated cell diameter in source-image pixels. If provided, image is
   *  resized so target diameter = 30 px (CPSAM's training median). */
  diameter?: number;
  /** Tile size. Default 256 (matches CPSAM's training bsize). */
  tile?: number;
  /** Tile overlap fraction in [0.05, 0.5]. Default 0.1. */
  overlap?: number;
  /** Percentile normalization knobs. */
  normalize?: NormalizeOptions;
}

export interface SegmentOutput {
  /** Per-tile model outputs. M5 will stitch these into a single label map. */
  tiles: SegmentTileOutput[];
  /** Resized image dimensions (after diameter rescale; equal to input if no resize). */
  resizedWidth: number;
  resizedHeight: number;
  /** Scale factor applied (resized = source * scale). */
  scale: number;
  /** Wall-clock ms across the whole segment() call. */
  totalMs: number;
}

/** @deprecated kept for M1 demo; use `segment()` instead. */
export interface SegmentMilestone1Output {
  flows_cellprob: Float32Array;
  height: number;
  width: number;
  inferenceMs: number;
}

export class Cellpose {
  private _session: CellposeSession | null = null;

  private constructor(
    private readonly _modelBytes: ArrayBuffer,
    private readonly _sessionOpts: SessionOptions
  ) {}

  /** Construct a Cellpose instance from a URL pointing at an FP16 CPSAM ONNX file. */
  static async fromPretrained(modelUrl: string, opts: FromPretrainedOptions = {}): Promise<Cellpose> {
    assertSupportedEnvironment();

    const fetchOpts = {
      ...(opts.onProgress  !== undefined && { onProgress: opts.onProgress }),
      ...(opts.bypassCache !== undefined && { bypassCache: opts.bypassCache }),
      ...(opts.signal      !== undefined && { signal: opts.signal }),
    };
    const bytes = await fetchModel(modelUrl, fetchOpts);

    const sessionOpts: SessionOptions = {
      ...(opts.wasmPaths !== undefined && { wasmPaths: opts.wasmPaths }),
    };
    const cp = new Cellpose(bytes, sessionOpts);

    if (opts.preload) await cp._ensureSession();
    return cp;
  }

  /** Returns adapter info for diagnostics. */
  async describeAdapter(): Promise<{ vendor: string; architecture: string; device: string; } | null> {
    return describeAdapter();
  }

  /** Lazy session creation. Idempotent. */
  private async _ensureSession(): Promise<CellposeSession> {
    if (!this._session) {
      this._session = await CellposeSession.create(this._modelBytes, this._sessionOpts);
    }
    return this._session;
  }

  /**
   * Milestone 1 stub: run a single (1, 3, 256, 256) forward pass.
   *
   * `input` is FP32 in NCHW order with values already preprocessed by the
   * caller. Milestone 2 will move preprocessing inside this method.
   */
  async segmentRawTile(input: Float32Array, height = 256, width = 256): Promise<SegmentMilestone1Output> {
    if (input.length !== 1 * 3 * height * width) {
      throw new Error(`segmentRawTile: expected ${1 * 3 * height * width} floats, got ${input.length}`);
    }
    const sess = await this._ensureSession();

    // FP32 -> FP16 at the model boundary.
    const fp16 = new Float16Array(input.length);
    for (let i = 0; i < input.length; i++) fp16[i] = input[i] as number;
    const tensor = new ort.Tensor('float16', fp16 as unknown as Uint16Array, [1, 3, height, width]);

    const t0 = performance.now();
    const outputs = await sess.run({ [sess.inputNames[0] as string]: tensor });
    const inferenceMs = performance.now() - t0;

    const out = outputs['flows_cellprob'];
    if (!out) throw new Error(`Missing output 'flows_cellprob' (got: ${Object.keys(outputs).join(', ')})`);

    // FP16 -> FP32 at the model boundary. ort.Tensor.data for float16 is a
    // Uint16Array view of bit patterns; Float16Array re-interprets those
    // when constructed from the same ArrayBuffer.
    const outF16 = new Float16Array((out.data as Uint16Array).buffer,
                                    (out.data as Uint16Array).byteOffset,
                                    (out.data as Uint16Array).length);
    const outF32 = new Float32Array(outF16.length);
    for (let i = 0; i < outF16.length; i++) outF32[i] = outF16[i] as number;

    return { flows_cellprob: outF32, height, width, inferenceMs };
  }

  /**
   * Full preprocess + per-tile inference. Returns per-tile flows; tile
   * stitching into a global instance label map is M5 work.
   */
  async segment(input: SegmentInput, opts: SegmentOptions = {}): Promise<SegmentOutput> {
    const t0 = performance.now();
    const { data, width, height, channels } = input;
    const tileSize = opts.tile ?? 256;

    // 1) Build CPSAM's 3-channel CHW input.
    let chw = buildCpsamChannels(data, width, height, channels, opts);
    let w = width, h = height, scale = 1;

    // 2) Optional diameter-aware resize.
    if (opts.diameter !== undefined) {
      const r = diameterResize(chw, w, h, { channels: 3, diameter: opts.diameter });
      chw = r.data; w = r.width; h = r.height; scale = r.scale;
    }

    // 3) Per-channel percentile normalization.
    chw = normalizePerChannel(chw, 3, w * h, opts.normalize ?? {});

    // 4) Tile (always 3 channels at this point, always bsize x bsize tiles).
    const tileOpts: { bsize: number; overlap?: number } = { bsize: tileSize };
    if (opts.overlap !== undefined) tileOpts.overlap = opts.overlap;
    const tiles: TileRecord[] = makeTiles(chw, w, h, 3, tileOpts);

    // 5) Run model on each tile.
    const out: SegmentTileOutput[] = [];
    for (const t of tiles) {
      const r = await this.segmentRawTile(t.tile, tileSize, tileSize);
      out.push({
        flows_cellprob: r.flows_cellprob,
        tx: t.tx, ty: t.ty, bsize: tileSize,
        inferenceMs: r.inferenceMs,
      });
    }
    return { tiles: out, resizedWidth: w, resizedHeight: h, scale, totalMs: performance.now() - t0 };
  }

  /** Release GPU resources. */
  async dispose(): Promise<void> {
    await this._session?.dispose();
    this._session = null;
  }
}

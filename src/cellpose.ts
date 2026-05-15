/**
 * Public Cellpose API. Milestone 1: model loading + identity forward pass.
 * Pre/postprocessing (Milestone 2+) and dynamics (Milestone 4) live in
 * sibling modules added later.
 */
import * as ort from 'onnxruntime-web/webgpu';
import { assertSupportedEnvironment, describeAdapter } from './env.js';
import { fetchModel, type FetchProgress } from './model-cache.js';
import { CellposeSession, type SessionOptions } from './session.js';

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
 * Milestone-1 segment() output: the raw model tensors converted to FP32.
 * Milestone 4 will add `masks`, `count`, etc. once the flow-dynamics
 * postprocessing is wired up.
 */
export interface SegmentMilestone1Output {
  /** Raw model output `flows_cellprob`, shape (1, 3, H, W), FP32 (converted from FP16). */
  flows_cellprob: Float32Array;
  height: number;
  width: number;
  /** Milliseconds spent in `sess.run()` for this call. */
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

  /** Release GPU resources. */
  async dispose(): Promise<void> {
    await this._session?.dispose();
    this._session = null;
  }
}

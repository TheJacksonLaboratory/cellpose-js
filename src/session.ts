/**
 * ORT-WebGPU session wrapper. Thin layer over `ort.InferenceSession.create()`
 * so the rest of the package doesn't have to know about WASM paths, EP names,
 * or the UMD vs ESM split.
 */
import * as ort from 'onnxruntime-web/webgpu';

/** CDN base for ORT-web's WASM auxiliary files (used by the WebGPU EP for unsupported ops). */
const DEFAULT_WASM_PATHS = '/ort/';

let wasmPathsConfigured = false;

/** One-time global setup. Idempotent. */
export function configureOrt(opts: { wasmPaths?: string } = {}): void {
  if (wasmPathsConfigured) return;
  ort.env.wasm.wasmPaths = opts.wasmPaths ?? DEFAULT_WASM_PATHS;
  wasmPathsConfigured = true;
}

export interface SessionOptions {
  /** Override CDN base for WASM helper files. */
  wasmPaths?: string;
}

export class CellposeSession {
  private constructor(private readonly _sess: ort.InferenceSession) {}

  static async create(modelBytes: ArrayBuffer, opts: SessionOptions = {}): Promise<CellposeSession> {
    configureOrt(opts.wasmPaths ? { wasmPaths: opts.wasmPaths } : {});
    const sess = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });
    return new CellposeSession(sess);
  }

  get inputNames(): readonly string[]  { return this._sess.inputNames; }
  get outputNames(): readonly string[] { return this._sess.outputNames; }

  async run(feed: Record<string, ort.Tensor>): Promise<Record<string, ort.Tensor>> {
    return this._sess.run(feed);
  }

  /** Release GPU/CPU resources. Subsequent run() calls will throw. */
  async dispose(): Promise<void> {
    // ort.InferenceSession.release() exists on the C++ binding but isn't on the
    // public TS type yet. Best-effort cast.
    const r = (this._sess as unknown as { release?: () => Promise<void> }).release;
    if (typeof r === 'function') await r.call(this._sess);
  }
}

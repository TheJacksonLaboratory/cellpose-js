/**
 * Public-facing ORT-web configuration.
 *
 * Library consumers must serve ort-web's WASM/JSEP sidecar files (notably
 * `ort-wasm-simd-threaded.asyncify.mjs` and the matching .wasm files)
 * from a same-origin URL — cross-origin dynamic-import is blocked. Call
 * `configureOrt({ wasmPaths: '/your/path/' })` once at app startup with
 * the URL prefix that serves those files.
 *
 * The default '/ort/' matches our demo's Vite proxy
 * (see examples/demo/vite.config.ts).
 */

export interface ConfigureOrtOptions {
  /** URL prefix where ORT-web's WASM/JSEP sidecar files are served. */
  wasmPaths?: string;
}

let _wasmPaths: string = '/ort/';

export function configureOrt(opts: ConfigureOrtOptions = {}): void {
  if (opts.wasmPaths) _wasmPaths = opts.wasmPaths;
}

/** Internal: current wasmPaths value, used when initializing the inference worker. */
export function _getWasmPaths(): string {
  return _wasmPaths;
}

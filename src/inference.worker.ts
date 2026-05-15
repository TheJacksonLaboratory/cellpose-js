/**
 * Inference worker — hosts the ORT-WebGPU session so per-tile inference
 * (~280 ms each) doesn't block the UI thread.
 *
 * Lifecycle:
 *   main thread: new Worker(...) -> postMessage({type:'init', modelBytes})
 *   worker:      ORT session create -> postMessage({type:'ready', adapterInfo})
 *   main thread: postMessage({type:'run-tile', tileId, tile, bsize}, [tile.buffer])
 *   worker:      sess.run(...) -> postMessage({type:'tile-result', tileId, flowsCellprob}, [flowsCellprob.buffer])
 *
 * Cancellation: main thread calls worker.terminate(); worker dies mid-run.
 * No graceful unwind needed — the session is gone with the worker.
 */
/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/webgpu';
import type { MainToWorker, WorkerToMain } from './worker-protocol.js';

let session: ort.InferenceSession | null = null;
let wasmPathsConfigured = false;

declare const self: DedicatedWorkerGlobalScope;

function configureOrt(wasmPaths: string | undefined): void {
  if (wasmPathsConfigured) return;
  if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
  wasmPathsConfigured = true;
}

async function describeAdapter(): Promise<{ vendor: string; architecture: string; device: string } | null> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const info = adapter.info ?? ({} as GPUAdapterInfo);
  return {
    vendor: info.vendor ?? '?',
    architecture: info.architecture ?? '?',
    device: info.device ?? '?',
  };
}

async function handleInit(msg: Extract<MainToWorker, { type: 'init' }>): Promise<void> {
  configureOrt(msg.wasmPaths);
  session = await ort.InferenceSession.create(msg.modelBytes, {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
  });
  const adapterInfo = await describeAdapter();
  postReply({ type: 'ready', adapterInfo });
}

async function handleRunTile(msg: Extract<MainToWorker, { type: 'run-tile' }>): Promise<void> {
  const sess = session;
  if (!sess) {
    postReply({ type: 'error', tileId: msg.tileId, message: 'worker not initialized' });
    return;
  }

  // FP32 -> FP16. Float16Array auto-rounds on store.
  const fp16 = new Float16Array(msg.tile.length);
  for (let i = 0; i < msg.tile.length; i++) fp16[i] = msg.tile[i] as number;
  // ort-web 1.26's TS types don't yet accept Float16Array directly; cast.
  const tensor = new ort.Tensor(
    'float16',
    fp16 as unknown as Uint16Array,
    [1, 3, msg.bsize, msg.bsize]
  );

  const t0 = performance.now();
  const outputs = await sess.run({ [sess.inputNames[0] as string]: tensor });
  const inferenceMs = performance.now() - t0;

  const out = outputs['flows_cellprob'];
  if (!out) {
    postReply({ type: 'error', tileId: msg.tileId, message: `missing 'flows_cellprob' output` });
    return;
  }

  // FP16 -> FP32. Reinterpret the Uint16Array as Float16Array on the same
  // buffer, then copy into FP32.
  const u16 = out.data as Uint16Array;
  const outF16 = new Float16Array(u16.buffer, u16.byteOffset, u16.length);
  const outF32 = new Float32Array(outF16.length);
  for (let i = 0; i < outF16.length; i++) outF32[i] = outF16[i] as number;

  postReply(
    { type: 'tile-result', tileId: msg.tileId, flowsCellprob: outF32, inferenceMs },
    [outF32.buffer]
  );
}

function postReply(msg: WorkerToMain, transfer?: Transferable[]): void {
  if (transfer && transfer.length > 0) self.postMessage(msg, transfer);
  else self.postMessage(msg);
}

self.addEventListener('message', async (ev: MessageEvent<MainToWorker>) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') await handleInit(msg);
    else if (msg.type === 'run-tile') await handleRunTile(msg);
    else if (msg.type === 'dispose') {
      session = null;
      self.close();
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    postReply({ type: 'error', tileId: msg.type === 'run-tile' ? msg.tileId : null, message });
  }
});

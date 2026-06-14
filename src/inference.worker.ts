/**
 * Inference worker — hosts the ORT-WebGPU session AND (via the `segment`
 * message) runs the entire Cellpose pipeline off the UI thread: preprocess →
 * tile → per-tile inference → tile averaging → flow dynamics → inverse-resize.
 * Only the final label map is posted back. This keeps the main thread free —
 * preprocess and dynamics used to run there and froze the UI.
 *
 * Lifecycle:
 *   main: new Worker(...) -> postMessage({type:'init', modelBytes})
 *   worker: ORT session create -> postMessage({type:'ready', adapterInfo})
 *   main: postMessage({type:'segment', reqId, image, opts})
 *   worker: postMessage({type:'segment-progress', ...})* ->
 *           postMessage({type:'segment-result', reqId, masks, ...}, [masks.buffer])
 *
 * The legacy `run-tile` message (single-tile inference, main thread orchestrates)
 * is kept for back-compat.
 *
 * Cancellation: main thread calls worker.terminate(); worker dies mid-run.
 */
/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/webgpu';
import type { MainToWorker, WorkerToMain } from './worker-protocol.js';
import {
  buildCpsamChannels,
  normalizePerChannel,
  diameterResize,
  makeTiles,
  type TileRecord,
} from './preprocess/index.js';
import {
  averageTiles,
  computeMasks,
  type ComputeMasksOptions,
  type TileInputForAveraging,
} from './postprocess/index.js';

let session: ort.InferenceSession | null = null;
let wasmPathsConfigured = false;

declare const self: DedicatedWorkerGlobalScope;

function configureOrt(wasmPaths: string | undefined): void {
  if (wasmPathsConfigured) return;
  if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
  wasmPathsConfigured = true;
}

async function describeAdapter(): Promise<{
  vendor: string;
  architecture: string;
  device: string;
} | null> {
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
  postReply({ type: 'status', status: 'configuring ORT (loading WASM sidecars)' });
  configureOrt(msg.wasmPaths);
  postReply({ type: 'status', status: 'creating ORT session (parsing 588 MB ONNX graph)' });
  const t0 = performance.now();
  session = await ort.InferenceSession.create(msg.modelBytes, {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
  });
  postReply({
    type: 'status',
    status: `session created in ${(performance.now() - t0).toFixed(0)} ms; describing adapter`,
  });
  const adapterInfo = await describeAdapter();
  postReply({ type: 'ready', adapterInfo });
}

/** Run one (C,B,B) tile through the model; returns FP32 flows+cellprob. */
async function runTile(
  sess: ort.InferenceSession,
  tile: Float32Array,
  bsize: number,
): Promise<{ flowsCellprob: Float32Array; inferenceMs: number }> {
  // FP32 -> FP16. Float16Array auto-rounds on store.
  const fp16 = new Float16Array(tile.length);
  for (let i = 0; i < tile.length; i++) fp16[i] = tile[i] as number;
  // ort-web's TS types don't yet accept Float16Array directly; cast.
  const tensor = new ort.Tensor('float16', fp16 as unknown as Uint16Array, [1, 3, bsize, bsize]);

  const t0 = performance.now();
  const outputs = await sess.run({ [sess.inputNames[0] as string]: tensor });
  const inferenceMs = performance.now() - t0;

  const out = outputs['flows_cellprob'];
  if (!out) throw new Error(`missing 'flows_cellprob' output`);

  // FP16 -> FP32. Reinterpret the Uint16Array as Float16Array on the same
  // buffer, then copy into FP32.
  const u16 = out.data as Uint16Array;
  const outF16 = new Float16Array(u16.buffer, u16.byteOffset, u16.length);
  const outF32 = new Float32Array(outF16.length);
  for (let i = 0; i < outF16.length; i++) outF32[i] = outF16[i] as number;
  return { flowsCellprob: outF32, inferenceMs };
}

async function handleRunTile(msg: Extract<MainToWorker, { type: 'run-tile' }>): Promise<void> {
  const sess = session;
  if (!sess) {
    postReply({ type: 'error', tileId: msg.tileId, message: 'worker not initialized' });
    return;
  }
  const r = await runTile(sess, msg.tile, msg.bsize);
  postReply({ type: 'tile-result', tileId: msg.tileId, flowsCellprob: r.flowsCellprob, inferenceMs: r.inferenceMs }, [
    r.flowsCellprob.buffer,
  ]);
}

/** Full pipeline in the worker — keeps preprocess + dynamics off the UI thread. */
async function handleSegment(msg: Extract<MainToWorker, { type: 'segment' }>): Promise<void> {
  const sess = session;
  if (!sess) {
    postReply({ type: 'segment-error', reqId: msg.reqId, message: 'worker not initialized' });
    return;
  }
  const { reqId, image, opts } = msg;
  const tileSize = opts.tile ?? 256;

  // Preprocess: channels -> normalize -> (optional) diameter resize.
  let chw = buildCpsamChannels(image.data, image.width, image.height, image.channels, opts);
  let w = image.width,
    h = image.height,
    scale = 1;
  chw = normalizePerChannel(chw, 3, w * h, opts.normalize ?? {});
  if (opts.diameter !== undefined) {
    const r = diameterResize(chw, w, h, { channels: 3, diameter: opts.diameter });
    chw = r.data;
    w = r.width;
    h = r.height;
    scale = r.scale;
  }

  const tileOpts: { bsize: number; overlap?: number } = { bsize: tileSize };
  if (opts.overlap !== undefined) tileOpts.overlap = opts.overlap;
  const tiles: TileRecord[] = makeTiles(chw, w, h, 3, tileOpts);

  const flows: TileInputForAveraging[] = [];
  const tileDiags: { tx: number; ty: number; bsize: number; inferenceMs: number }[] = [];
  for (let i = 0; i < tiles.length; i++) {
    const t = tiles[i]!;
    const r = await runTile(sess, t.tile, tileSize);
    flows.push({ flowsCellprob: r.flowsCellprob, tx: t.tx, ty: t.ty, bsize: tileSize });
    tileDiags.push({ tx: t.tx, ty: t.ty, bsize: tileSize, inferenceMs: r.inferenceMs });
    postReply({ type: 'segment-progress', reqId, phase: 'inference', done: i + 1, total: tiles.length });
  }

  // Average overlapping tile predictions, then run dynamics once.
  postReply({ type: 'segment-progress', reqId, phase: 'dynamics', done: 0, total: 1 });
  const tPost = performance.now();
  const averaged = averageTiles(flows, h, w);
  const hwFull = h * w;
  const dPFull = averaged.data.subarray(0, 2 * hwFull) as Float32Array;
  const cpFull = averaged.data.subarray(2 * hwFull, 3 * hwFull) as Float32Array;
  const dynOpts: ComputeMasksOptions = { ...(opts.dynamics ?? {}) };
  if (dynOpts.niter === undefined && opts.diameter !== undefined && scale !== 1) {
    dynOpts.niter = Math.max(1, Math.floor(200 / scale));
  }
  const m = computeMasks(dPFull, cpFull, h, w, dynOpts);

  // Inverse-resize labels back to source resolution (nearest-neighbor).
  let masksSrc = m.masks;
  let outW = w,
    outH = h;
  if (scale !== 1) {
    outW = image.width;
    outH = image.height;
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
  postReply(
    {
      type: 'segment-result',
      reqId,
      masks: masksSrc,
      count: m.count,
      width: outW,
      height: outH,
      resizedWidth: w,
      resizedHeight: h,
      scale,
      tiles: tileDiags,
      postprocessMs,
    },
    [masksSrc.buffer],
  );
}

function postReply(msg: WorkerToMain, transfer?: Transferable[]): void {
  if (transfer && transfer.length > 0) self.postMessage(msg, transfer);
  else self.postMessage(msg);
}

// Beacon: confirms the worker module loaded successfully (i.e. the
// `import * as ort` at the top didn't throw).
postReply({ type: 'status', status: 'worker module loaded' });

self.addEventListener('message', async (ev: MessageEvent<MainToWorker>) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      postReply({ type: 'status', status: 'init message received' });
      await handleInit(msg);
    } else if (msg.type === 'run-tile') await handleRunTile(msg);
    else if (msg.type === 'segment') await handleSegment(msg);
    else if (msg.type === 'dispose') {
      session = null;
      self.close();
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (msg.type === 'segment') postReply({ type: 'segment-error', reqId: msg.reqId, message });
    else postReply({ type: 'error', tileId: msg.type === 'run-tile' ? msg.tileId : null, message });
  }
});

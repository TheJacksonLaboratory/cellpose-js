import { Cellpose, clearCachedModel, UnsupportedEnvironmentError,
         type SegmentInput } from '../../src/index.js';

const $  = <T extends HTMLElement = HTMLInputElement>(id: string) =>
            document.getElementById(id) as T;
const logEl = document.getElementById('log')!;
const log = (msg: string, cls?: string) => {
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = msg + '\n';
  logEl.appendChild(span);
  logEl.scrollTop = logEl.scrollHeight;
};
const fmt = (ms: number) => (ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms/1000).toFixed(2)} s`);

let cp: Cellpose | null = null;
let currentImage: SegmentInput | null = null;

async function ensureModel(): Promise<Cellpose> {
  if (cp) return cp;
  const modelUrl    = $('modelUrl').value;
  const preload     = ($('preload') as HTMLInputElement).checked;
  const bypassCache = ($('bypassCache') as HTMLInputElement).checked;
  log(`fromPretrained('${modelUrl}', { preload: ${preload}, bypassCache: ${bypassCache} })...`);
  const t0 = performance.now();
  let lastPctLogged = -1;
  cp = await Cellpose.fromPretrained(modelUrl, {
    preload, bypassCache,
    onProgress: ({ loaded, total }) => {
      if (!total) return;
      const pct = Math.floor((loaded / total) * 100);
      if (pct >= lastPctLogged + 10 || pct === 100) {
        log(`  fetch: ${pct}%`);
        lastPctLogged = pct;
      }
    },
  });
  log(`model ready in ${fmt(performance.now() - t0)}`);
  return cp;
}

function imageDataToSegmentInput(img: ImageData): SegmentInput {
  return { data: img.data, width: img.width, height: img.height, channels: 4 };
}

function drawToCanvas(canvas: HTMLCanvasElement, image: SegmentInput): void {
  canvas.width  = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d')!;
  if (image.channels === 4 && image.data instanceof Uint8ClampedArray) {
    ctx.putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
  } else {
    // Convert generic source to RGBA for display
    const rgba = new Uint8ClampedArray(image.width * image.height * 4);
    for (let i = 0; i < image.width * image.height; i++) {
      const v = Number(image.data[i * image.channels] ?? 0);
      rgba[i*4] = rgba[i*4+1] = rgba[i*4+2] = v;
      rgba[i*4+3] = 255;
    }
    ctx.putImageData(new ImageData(rgba, image.width, image.height), 0, 0);
  }
}

function drawHeatmap(canvas: HTMLCanvasElement, data: Float32Array, w: number, h: number): void {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  // Auto-range to data min/max for display
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i]!;
    if (Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
  }
  const span = (mx - mn) || 1;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const t = ((data[i] as number) - mn) / span;        // [0,1]
    const u8 = Math.max(0, Math.min(255, Math.round(t * 255)));
    // Simple turbo-ish: cool→warm via (255-u8, u8/2, u8) — readable enough for diagnostics.
    rgba[i*4]     = 255 - u8;
    rgba[i*4 + 1] = Math.round(u8 / 2);
    rgba[i*4 + 2] = u8;
    rgba[i*4 + 3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
}

async function loadImageFile(file: File): Promise<SegmentInput> {
  const bitmap = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  const id = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  return imageDataToSegmentInput(id);
}

function makeSynthetic(w = 400, h = 400): SegmentInput {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      // Sum of a few Gaussians to fake some blobs
      let v = 30;
      for (const [cx, cy, r] of [[100, 100, 30], [250, 180, 40], [320, 320, 25], [80, 300, 35]]) {
        const dx = x - (cx as number), dy = y - (cy as number);
        v += 200 * Math.exp(-(dx*dx + dy*dy) / (2 * (r as number) * (r as number)));
      }
      data[i] = data[i+1] = data[i+2] = Math.min(255, v);
      data[i+3] = 255;
    }
  }
  return { data, width: w, height: h, channels: 4 };
}

let activeAbort: AbortController | null = null;

async function run() {
  log('--- run ---', 'muted');
  if (!currentImage) { log('no image loaded', 'fail'); return; }
  ($('go') as HTMLButtonElement).disabled = true;
  ($('abort') as HTMLButtonElement).disabled = false;
  activeAbort = new AbortController();
  try {
    const model = await ensureModel();
    const adapter = await model.describeAdapter();
    log(`adapter: vendor=${adapter?.vendor} arch=${adapter?.architecture}`);

    const diameterStr = ($('diameter') as HTMLInputElement).value.trim();
    const opts: Parameters<Cellpose['segment']>[1] = {
      tile: parseInt($('tile').value, 10),
      chan:  parseInt($('chan').value, 10)  as 0|1|2|3,
      chan2: parseInt($('chan2').value, 10) as 0|1|2|3,
    };
    if (diameterStr) opts.diameter = parseFloat(diameterStr);

    opts.signal = activeAbort.signal;
    const totalTiles = { v: 0 };
    opts.onTileProgress = (done, total) => {
      totalTiles.v = total;
      ($('progress') as HTMLSpanElement).textContent = `tile ${done}/${total}`;
    };
    log(`segment(${currentImage.width}x${currentImage.height}, opts.tile=${opts.tile}, chan=${opts.chan}, chan2=${opts.chan2}, diameter=${opts.diameter ?? 'auto'})...`);
    const r = await model.segment(currentImage, opts);
    log(`tiles: ${r.tiles.length}  resized: ${r.resizedWidth}x${r.resizedHeight}  scale: ${r.scale.toFixed(3)}`);
    const infTimes = r.tiles.map(t => t.inferenceMs).sort((a, b) => a - b);
    const median = infTimes[Math.floor(infTimes.length/2)] ?? 0;
    log(`per-tile inference median: ${fmt(median)}  total: ${fmt(r.totalMs)}`);

    // Visualize tile 0 channels
    const t0 = r.tiles[0]!;
    const B = t0.bsize;
    const hw = B * B;
    drawHeatmap($('flowYCanvas') as HTMLCanvasElement, t0.flows_cellprob.subarray(0, hw),        B, B);
    drawHeatmap($('flowXCanvas') as HTMLCanvasElement, t0.flows_cellprob.subarray(hw, 2*hw),     B, B);
    drawHeatmap($('cellprobCanvas') as HTMLCanvasElement, t0.flows_cellprob.subarray(2*hw, 3*hw), B, B);
    log('GATE: PASS (preprocess + inference complete; postprocess is M4/M5)', 'pass');
  } catch (err: unknown) {
    const e = err as Error;
    if (e instanceof UnsupportedEnvironmentError) log(`ENV: ${e.message}`, 'fail');
    else if (e?.name === 'AbortError')             log(`ABORTED: ${e.message ?? 'cancelled'}`, 'warn');
    else                                            log(`ERROR: ${e.message ?? e}`, 'fail');
    console.error(err);
  } finally {
    ($('go') as HTMLButtonElement).disabled = false;
    ($('abort') as HTMLButtonElement).disabled = true;
    ($('progress') as HTMLSpanElement).textContent = '';
    activeAbort = null;
  }
}

$('imageFile').addEventListener('change', async (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (!f) return;
  currentImage = await loadImageFile(f);
  drawToCanvas($('inputCanvas') as HTMLCanvasElement, currentImage);
  ($('go') as HTMLButtonElement).disabled = false;
  log(`loaded ${f.name}: ${currentImage.width}x${currentImage.height}, ch=${currentImage.channels}`);
});
$('useSynthetic').addEventListener('click', () => {
  currentImage = makeSynthetic();
  drawToCanvas($('inputCanvas') as HTMLCanvasElement, currentImage);
  ($('go') as HTMLButtonElement).disabled = false;
  log(`loaded synthetic: ${currentImage.width}x${currentImage.height}`);
});
$('go').addEventListener('click', run);
$('abort').addEventListener('click', () => {
  if (activeAbort) {
    const t = performance.now();
    activeAbort.abort();
    log(`abort fired (will measure latency on next tick)`);
    requestAnimationFrame(() => log(`abort UI tick at ${(performance.now() - t).toFixed(0)} ms`));
  }
});
const clearBtn = document.createElement('button');
clearBtn.textContent = 'Clear cached model';
clearBtn.addEventListener('click', async () => {
  await clearCachedModel($('modelUrl').value);
  if (cp) { await cp.dispose(); cp = null; }
  log('cleared cache and disposed session.');
});
$('go').parentElement!.appendChild(clearBtn);

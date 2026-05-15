import { Cellpose, clearCachedModel, UnsupportedEnvironmentError } from '../../src/index.js';

const $ = (id: string) => document.getElementById(id) as HTMLInputElement;
const logEl = document.getElementById('log')!;
const log = (msg: string, cls?: string) => {
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = msg + '\n';
  logEl.appendChild(span);
};
const fmt = (ms: number) => (ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms/1000).toFixed(2)} s`);

async function run() {
  logEl.textContent = '';
  const goBtn = $('go'); goBtn.disabled = true;
  try {
    const modelUrl    = $('modelUrl').value;
    const preload     = $('preload').checked;
    const bypassCache = $('bypassCache').checked;

    log(`fromPretrained('${modelUrl}', { preload: ${preload}, bypassCache: ${bypassCache} })...`);
    let lastPct = -1;
    const t0 = performance.now();
    const cp = await Cellpose.fromPretrained(modelUrl, {
      preload,
      bypassCache,
      onProgress: ({ loaded, total }) => {
        if (total) {
          const pct = Math.floor((loaded / total) * 100);
          if (pct !== lastPct && pct % 5 === 0) { log(`  fetch: ${pct}%`); lastPct = pct; }
        } else if ((loaded & ((1 << 22) - 1)) === 0) {
          log(`  fetch: ${(loaded/1024/1024).toFixed(0)} MB`);
        }
      },
    });
    log(`ready in ${fmt(performance.now() - t0)}`);

    const adapter = await cp.describeAdapter();
    log(`adapter: vendor=${adapter?.vendor} arch=${adapter?.architecture}`);

    // Synthetic input tile (will be replaced with real preprocessing in Milestone 2).
    const N = 1 * 3 * 256 * 256;
    const input = new Float32Array(N);
    for (let i = 0; i < N; i++) input[i] = Math.random() * 2 - 1;

    log('running 3 warmup + 5 timed forward passes...');
    for (let i = 0; i < 3; i++) {
      const r = await cp.segmentRawTile(input);
      log(`  warmup ${i}: inference ${fmt(r.inferenceMs)}`);
    }
    const times: number[] = [];
    let last: Float32Array | null = null;
    for (let i = 0; i < 5; i++) {
      const r = await cp.segmentRawTile(input);
      times.push(r.inferenceMs);
      last = r.flows_cellprob;
      log(`  iter ${i}: ${fmt(r.inferenceMs)}`);
    }
    times.sort((a, b) => a - b);
    const median = times[Math.floor(times.length / 2)] as number;
    log(`\ninference median: ${fmt(median)}`);

    if (!last) { log('no output captured', 'fail'); return; }
    let mn = Infinity, mx = -Infinity, sum = 0, nonFinite = 0;
    for (let i = 0; i < last.length; i++) {
      const v = last[i] as number;
      if (Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
      else nonFinite++;
    }
    log(`output: shape=(1, 3, 256, 256) len=${last.length} min=${mn.toFixed(3)} max=${mx.toFixed(3)} mean=${(sum/last.length).toFixed(3)} non-finite=${nonFinite}`);

    if (nonFinite === 0 && median < 2000) log('GATE: PASS', 'pass');
    else if (median < 5000)               log('GATE: MARGINAL', 'warn');
    else                                  log('GATE: FAIL', 'fail');
  } catch (err: unknown) {
    const e = err as Error;
    if (e instanceof UnsupportedEnvironmentError) log(`ENV: ${e.message}`, 'fail');
    else log(`ERROR: ${e.message ?? e}`, 'fail');
    console.error(err);
  } finally {
    goBtn.disabled = false;
  }
}

$('go').addEventListener('click', run);
$('clear').addEventListener('click', async () => {
  await clearCachedModel($('modelUrl').value);
  log('cleared cached model entry.');
});

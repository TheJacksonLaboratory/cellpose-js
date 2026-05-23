import { defineConfig, type Plugin } from 'vite';
import fs from 'node:fs';

/**
 * The model itself is served from examples/demo/public/cpsam_fp16.onnx
 * (a symlink to ~/cellpose-js-spike/cpsam_fp16.onnx in dev).
 *
 * NOTE: the @1.26.0 in the rewrite URL must match the onnxruntime-web
 * peerDependency pin in package.json. Keep them in lock-step.
 *
 * ORT-web's WebGPU backend dynamically imports several .mjs sidecar files.
 * We cannot put them in public/ because Vite refuses to serve public files
 * to module-import requests (the `?import` query path is intercepted by
 * Vite's plugin pipeline). Cross-origin dynamic import to jsdelivr is
 * also blocked. Solution: proxy /ort/* to jsdelivr — the browser sees
 * same-origin URLs while bytes come from jsdelivr's CORS-enabled CDN.
 */

/**
 * Vite's worker plugin transforms `new URL('./inference.worker.js',
 * import.meta.url)` into a URL ending in `inference.worker.js?worker_file`.
 * The library source uses `.js` so the tsc-built prod artifact resolves
 * correctly. But in dev mode the file on disk is `.ts`, and the worker
 * plugin doesn't auto-swap extensions like the regular resolver does — the
 * request 404s, then falls through to the SPA fallback (200 OK index.html),
 * and the browser silently fails to parse HTML as a JS module.
 *
 * This middleware rewrites `…/foo.js?worker_file…` → `…/foo.ts?worker_file…`
 * when the `.ts` file exists on disk. Dev-only; the prod build path is
 * unaffected.
 */
const remapWorkerTsExt = (): Plugin => ({
  name: 'cellpose-demo-remap-worker-ts-ext',
  configureServer(server) {
    server.middlewares.use((req, _res, next) => {
      const url = req.url;
      if (url && url.includes('worker_file') && url.includes('.js?')) {
        const [pathPart, queryPart = ''] = url.split('?');
        if (pathPart && pathPart.endsWith('.js')) {
          const tsPath = pathPart.replace(/\.js$/, '.ts');
          const onDisk = tsPath.replace(/^\/@fs/, '');
          if (fs.existsSync(onDisk)) {
            req.url = `${tsPath}?${queryPart}`;
          }
        }
      }
      next();
    });
  },
});

export default defineConfig({
  plugins: [remapWorkerTsExt()],
  server: {
    proxy: {
      '/ort': {
        target: 'https://cdn.jsdelivr.net',
        changeOrigin: true,
        rewrite: (path) => `/npm/onnxruntime-web@1.26.0/dist${path.replace(/^\/ort/, '')}`,
      },
    },
  },
});

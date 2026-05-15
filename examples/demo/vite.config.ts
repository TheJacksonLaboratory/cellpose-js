import { defineConfig } from 'vite';

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
export default defineConfig({
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

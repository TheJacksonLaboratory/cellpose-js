import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'CellposeJs',
      formats: ['es'],
      fileName: () => 'cellpose-js.js',
    },
    rollupOptions: {
      // Don't bundle peer deps; consumers bring their own.
      external: ['onnxruntime-web', 'onnxruntime-web/webgpu'],
    },
    sourcemap: true,
    target: 'es2022',
  },
  // Worker chunks get their own Rollup build. Without this `external` rule
  // ort-web gets inlined into the worker, producing a ~63 MB chunk. With it
  // the worker chunk stays under ~5 kB and the consumer's bundler resolves
  // ort-web at runtime.
  worker: {
    format: 'es',
    rollupOptions: {
      external: ['onnxruntime-web', 'onnxruntime-web/webgpu'],
    },
  },
});

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
});

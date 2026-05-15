# cellpose-js

Browser-side cellular segmentation via [Cellpose-SAM](https://github.com/MouseLand/cellpose), running on WebGPU.

> **Status:** early development — Milestone 1 (skeleton + identity forward pass). See the implementation plan
> in the consuming jit-ui repo (`apps/jit-ui/src/app/main/components/processing-pipeline/CELLPOSE-JS-PLAN.md`).

## Requirements

- Chrome ≥135 or Safari ≥17.4 (native `Float16Array` required)
- WebGPU available (`'gpu' in navigator`)
- `onnxruntime-web` ~1.26.0 as a peer dependency
- The CPSAM FP16 ONNX model (~588 MB) hosted somewhere reachable from the browser

## Quick start

```ts
import { Cellpose } from 'cellpose-js';

const cp = await Cellpose.fromPretrained('https://your-cdn/cpsam_fp16.onnx', {
  preload: true,
});

const out = await cp.segment(rgbaImage, {
  diameter: 30,
  tile: 256,
  cellprob_threshold: 0,
  flow_threshold: 0.4,
});
```

## License

MIT — see [LICENSE](./LICENSE).

# Security

## Reporting a vulnerability

If you believe you've found a vulnerability in `cellpose-js`, please **do not** open a public issue. Email the maintainer directly or file a [private security advisory](https://github.com/belkassaby/Cellpose.js/security/advisories/new). We'll respond within a week.

## Threat model and input-handling surface

`cellpose-js` is a thin TypeScript/WebGPU client wrapping [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web). It runs entirely in a browser context — there is no Node.js code path in the published package. The published tarball contains only `dist/*.{js,d.ts,map}` plus README/LICENSE; no scripts run at install or import time.

### What the package does at runtime

1. **`fetch(modelUrl)`** — downloads the model file from a URL the caller supplies to `Cellpose.fromPretrained(modelUrl)`.
2. **IndexedDB write/read** — caches the model bytes locally to skip the network on subsequent visits.
3. **Dynamic `import()`** — `onnxruntime-web`'s WebGPU backend imports its own WASM/JSEP sidecar `.mjs` files from a path set by `configureOrt({ wasmPaths })`.
4. **WebGPU + Web Worker** — runs the ONNX graph on the GPU inside a dedicated worker.

### Inputs that affect what gets fetched

| Input | Default | Risk if attacker-controlled |
|---|---|---|
| `modelUrl` argument to `Cellpose.fromPretrained(modelUrl, …)` | none — caller must supply | An attacker who controls this string can cause the browser to download arbitrary content (up to 588 MB and beyond). Same-origin policy and CORS still apply, so the response can't be read cross-origin without permission, but a malicious URL can: (1) waste bandwidth, (2) point at a malformed ONNX that crashes the inference worker, (3) cause IndexedDB to fill with attacker-supplied bytes. |
| `wasmPaths` argument to `configureOrt({ wasmPaths })` | `/ort/` | Same risk class as `modelUrl`. The path is used for dynamic ESM `import()`, which requires same-origin or proper CORS — so cross-origin substitution is hard. Same-origin substitution would require an existing site compromise. |

### What the package does **not** do

- No Node.js `fs` reads. The browser build never touches the filesystem.
- No `eval`, `Function()`, or other dynamic code execution from string content.
- No telemetry, analytics, or beaconing.
- No persistent storage other than the model byte cache in IndexedDB (consumer-cleared via `clearCachedModel(url)`).

### Recommended consumer practices

If you're integrating `cellpose-js` into a product:

1. **Hardcode the model URL** or restrict the user-facing input to a vetted allowlist (HuggingFace Hub, your own CDN). Don't accept arbitrary URLs from untrusted users.
2. **Serve `cellpose-js` and its ORT WASM sidecars same-origin** to avoid CORS and dynamic-import edge cases. The reference setup is in [`examples/demo/vite.config.ts`](./examples/demo/vite.config.ts) and is also documented in the README.
3. **Validate model integrity** if your threat model includes a compromised HF Hub URL. `onnxruntime-web`'s session-create will reject malformed graphs, but won't detect a substituted-but-valid model. If integrity matters, check the model's SHA-256 client-side before passing it to `Cellpose.fromPretrained()`.
4. **Lock the `onnxruntime-web` version**. `cellpose-js@0.1.x` declares it as a `~1.26.0` peer dep. You may want to pin it to a specific patch in your own `package.json`.

## Build & supply chain

- **Source:** [github.com/belkassaby/Cellpose.js](https://github.com/belkassaby/Cellpose.js) (public, MIT). `main` is protected; all changes via PR.
- **Build:** plain `tsc` emits to `dist/`. No bundler, no minification, no transformation beyond TypeScript transpilation. The published JS is reviewable diff-for-diff against the source.
- **CI:** every push and PR runs `tsc --noEmit`, ESLint, Prettier, Vitest, and a build — see [`.github/workflows/ci-cd.yaml`](./.github/workflows/ci-cd.yaml).
- **Publish provenance:** every npm release ships with a signed [sigstore provenance attestation](https://docs.npmjs.com/generating-provenance-statements) linking the tarball back to the GitHub Actions run that built it. Verify with `npm view cellpose-js dist.attestations`.
- **No install scripts.** The published `package.json` has no `preinstall` / `postinstall` / `prepare` hooks.

## Known scanner findings

`onnxruntime-web` (cellpose-js's peer dependency) is sometimes flagged by static-analysis scanners with labels like *"Obfuscated code"* or *"Supply-chain risk"*. These are typically structural false positives:

- ort-web ships minified bundles and binary `.wasm` blobs by design — large minified JS looks "obfuscated" to a scanner but is just minified.
- ort-web also accepts caller-supplied URLs/paths (it's a generic ONNX runtime), which scanners flag as a permissive input surface. That's the documented behavior of any model-loader library; mitigations belong at the consumer layer (allowlists, hardcoded defaults) — see §Recommended consumer practices above.

A 2026-05-15 scan of ort-web found no evidence of `eval`, dynamic code execution, credential harvesting, exfiltration, or persistence — the reviewer's note explicitly classifies it as "likely non-malicious library code with a contextual input-handling risk." `cellpose-js` itself adds no new such surface.

/**
 * Runtime environment checks. Called eagerly from Cellpose.fromPretrained()
 * so we fail fast with a clear error instead of cryptic ORT or tensor messages.
 */

export class UnsupportedEnvironmentError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UnsupportedEnvironmentError';
  }
}

/** Throws if WebGPU or Float16Array is unavailable. */
export function assertSupportedEnvironment(): void {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    throw new UnsupportedEnvironmentError(
      'WebGPU is not available. cellpose-js requires Chrome >=135 or Safari >=17.4 ' +
        'with WebGPU enabled.',
    );
  }
  if (typeof Float16Array === 'undefined') {
    throw new UnsupportedEnvironmentError(
      'Native Float16Array is not available. cellpose-js requires Chrome >=135 ' +
        '(Feb 2025) or Safari >=17.4 to consume the FP16 model.',
    );
  }
}

/** Returns adapter info for diagnostics. Does not throw. */
export async function describeAdapter(): Promise<{
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

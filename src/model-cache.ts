/**
 * Fetch ONNX model bytes from a URL, with an IndexedDB cache.
 *
 * Cache keying: we key by URL **plus** an HTTP ETag (or Last-Modified) probed
 * via HEAD so a re-uploaded model invalidates automatically. If the server
 * returns neither header, we fall back to URL-only keying (cache hit is then
 * "first byte wins; clear cache manually to refresh").
 */
import { get as idbGet, set as idbSet, del as idbDel } from 'idb-keyval';

export interface FetchProgress {
  loaded: number;
  total: number | null; // null if Content-Length is absent
}

export interface FetchModelOptions {
  /** Called with byte counts while streaming. */
  onProgress?: (p: FetchProgress) => void;
  /** Force a network fetch and overwrite any cached copy. */
  bypassCache?: boolean;
  /** AbortSignal to cancel the fetch. */
  signal?: AbortSignal;
}

const CACHE_VERSION = 1;
const cacheKey = (url: string, etag: string | null): string =>
  `cellpose-js:v${CACHE_VERSION}:${url}#${etag ?? 'no-etag'}`;

async function probeEtag(url: string, signal?: AbortSignal): Promise<string | null> {
  try {
    const res = await fetch(url, signal ? { method: 'HEAD', signal } : { method: 'HEAD' });
    if (!res.ok) return null;
    return res.headers.get('ETag') ?? res.headers.get('Last-Modified') ?? null;
  } catch {
    return null;
  }
}

/**
 * Fetch ONNX model bytes for a given URL, using IndexedDB to skip the
 * network on subsequent visits. Returns the raw ArrayBuffer ready for
 * `ort.InferenceSession.create(buf, …)`.
 */

/** Cache write is best-effort: we never want a quota / browser-quirk failure
 *  to block the actual model load. */
async function tryCacheWrite(key: string, buf: ArrayBuffer): Promise<void> {
  try {
    await idbSet(key, buf);
  } catch (err) {
    console.warn('[cellpose-js] IndexedDB cache write failed; will refetch next time.', err);
  }
}

export async function fetchModel(
  url: string,
  opts: FetchModelOptions = {}
): Promise<ArrayBuffer> {
  const { onProgress, bypassCache, signal } = opts;

  const etag = await probeEtag(url, signal);
  const key = cacheKey(url, etag);

  if (!bypassCache) {
    const cached = await idbGet<ArrayBuffer>(key);
    if (cached) return cached;
    // Etag-keyed miss: try to evict any older entries for the same URL.
    // (Best-effort; idb-keyval doesn't support prefix iteration, so this is
    // a no-op unless the consumer calls clearCache().)
  }

  const res = await fetch(url, signal ? { signal } : {});
  if (!res.ok) {
    throw new Error(`Failed to fetch model from ${url}: HTTP ${res.status}`);
  }
  const totalHdr = res.headers.get('Content-Length');
  const total = totalHdr ? parseInt(totalHdr, 10) : null;

  // Stream so we can report progress; collect chunks then assemble.
  const reader = res.body?.getReader();
  if (!reader) {
    // No streaming reader (rare). Fall back to a single arrayBuffer().
    const buf = await res.arrayBuffer();
    onProgress?.({ loaded: buf.byteLength, total: buf.byteLength });
    await tryCacheWrite(key, buf);
    return buf;
  }
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      onProgress?.({ loaded, total });
    }
  }
  const merged = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    merged.set(c, offset);
    offset += c.byteLength;
  }
  const buf = merged.buffer;
  await tryCacheWrite(key, buf);
  return buf;
}

/** Remove a single cached model (any version/etag) — best-effort. */
export async function clearCachedModel(url: string): Promise<void> {
  // We don't know the etag at clear-time, so try both known-key forms.
  const etag = await probeEtag(url);
  await idbDel(cacheKey(url, etag));
  await idbDel(cacheKey(url, null));
}

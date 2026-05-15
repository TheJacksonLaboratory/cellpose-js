/** Minimal .npy reader for float32 little-endian C-order arrays. */
import fs from 'node:fs';

export interface NpyArray {
  data: Float32Array | Uint16Array | Uint32Array;
  shape: number[];
}

export function readNpy(path: string): NpyArray {
  const bytes = fs.readFileSync(path);
  // magic: \x93NUMPY
  if (bytes[0] !== 0x93 || bytes.toString('ascii', 1, 6) !== 'NUMPY') {
    throw new Error(`not a .npy file: ${path}`);
  }
  const major = bytes[6];
  const headerLen = major === 1
    ? bytes.readUInt16LE(8)
    : bytes.readUInt32LE(8);
  const headerStart = major === 1 ? 10 : 12;
  const header = bytes.toString('ascii', headerStart, headerStart + headerLen);
  // Parse the python dict by ad-hoc regex — sufficient for our fixtures.
  const dtypeMatch = header.match(/'descr':\s*'([^']+)'/);
  const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
  const forderMatch = header.match(/'fortran_order':\s*(True|False)/);
  if (!dtypeMatch || !shapeMatch || !forderMatch) throw new Error(`malformed npy header: ${header}`);
  const dtype = dtypeMatch[1];
  if (!['<f4', '<u2', '<u4'].includes(dtype as string)) throw new Error(`unsupported dtype: ${dtype}`);
  if (forderMatch[1] !== 'False') throw new Error(`fortran-order arrays not supported`);
  const shape = (shapeMatch[1] || '').split(',')
    .map((s) => s.trim()).filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));
  const dataStart = headerStart + headerLen;
  const rawBuf = bytes.buffer.slice(bytes.byteOffset + dataStart, bytes.byteOffset + bytes.byteLength);
  const data = dtype === '<f4' ? new Float32Array(rawBuf)
             : dtype === '<u2' ? new Uint16Array(rawBuf)
             : new Uint32Array(rawBuf);
  const expected = shape.reduce((a, b) => a * b, 1);
  if (data.length !== expected) {
    throw new Error(`shape ${JSON.stringify(shape)} mismatches data length ${data.length}`);
  }
  return { data, shape };
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs((a[i] as number) - (b[i] as number));
    if (d > max) max = d;
  }
  return max;
}

/// <reference types="@webgpu/types" />

// Float16Array is part of TC39 Stage-3, shipped in Chrome 135 / Safari 17.4
// but not yet in TypeScript's stock lib (as of TS 5.6). Minimal ambient shim
// covering only what we use.
declare global {
  interface Float16ArrayConstructor {
    new (length: number): Float16Array;
    new (array: ArrayLike<number> | ArrayBufferLike): Float16Array;
    new (buffer: ArrayBufferLike, byteOffset?: number, length?: number): Float16Array;
    readonly BYTES_PER_ELEMENT: number;
    readonly prototype: Float16Array;
  }
  interface Float16Array {
    readonly length: number;
    readonly buffer: ArrayBufferLike;
    readonly byteLength: number;
    readonly byteOffset: number;
    [index: number]: number;
    set(array: ArrayLike<number>, offset?: number): void;
    subarray(begin?: number, end?: number): Float16Array;
  }
  const Float16Array: Float16ArrayConstructor;
}

export {};

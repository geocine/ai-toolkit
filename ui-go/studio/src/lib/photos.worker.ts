// Measures one photo in the browser before upload (PHOTO-PLAN section 3,
// metrics_version 1): the same copies and metrics as
// internal/photocheck/metrics.go, so the server's judge reads browser and
// server measurements alike. Values differ a little because the decoders
// and the first downscale differ (the browser draws the S copy on a canvas);
// the box filter and the dHash are the same arithmetic, pinned by
// internal/photocheck/testdata/dhash_golden.json (studio/scripts/check-dhash.mjs).
//
// As a worker it answers WorkIn messages. photos.ts imports the same code to
// measure on the main thread where workers have no OffscreenCanvas.

import { headSize, injectExif, sniff, type Sniffed } from './exif';
import type { PhotoMetrics } from './photoTypes';

export type FixWhat = 'upright' | 'gray16' | 'flattened';
export type WorkIn = { id: number; file: File; sha: boolean };
export type WorkOut =
  | { id: number; metrics: PhotoMetrics; fixed?: { blob: Blob; name: string; what: FixWhat } }
  | { id: number; error: string };

// Sizes of the section 3 copies.
const sharpEdge = 1024; // S: area-equivalent edge
const workLong = 512; // W: long side
const colourLong = 128; // C: long side
const maxPixels = 100_000_000; // thumbs.MaxPixels: never decoded past this
const transparentShare = 0.005; // flattened on white above this

/**
 * Measures file; with fix, a JPEG, PNG or WebP stored sideways is turned
 * upright, a 16-bit grey PNG becomes 8-bit and transparency over 0.5% is
 * flattened on white (JPEG, same stem). The metrics then describe the fixed
 * file. A file that does not decode comes back with decoded false and
 * error set; it throws only when no canvas can turn a sideways photo upright
 * (the server then measures it).
 */
export async function measure(file: File, opts: { sha: boolean; fix: boolean }): Promise<Omit<Extract<WorkOut, { metrics: PhotoMetrics }>, 'id'>> {
  const head = new Uint8Array(await file.slice(0, headSize).arrayBuffer());
  const sn = sniff(head);
  const m = headerMetrics(sn, file);
  if (opts.sha) m.sha256 = await sha256(file);
  if ((sn.width ?? 0) * (sn.height ?? 0) > maxPixels) {
    m.error = 'too large to analyse';
    return { metrics: m };
  }
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    m.error = 'cannot decode this file';
    return { metrics: m };
  }
  try {
    if (bmp.width * bmp.height > maxPixels) {
      m.error = 'too large to analyse';
      return { metrics: m };
    }
    let full = true;
    if (sn.orientation >= 2 && sn.orientation <= 8 && !(await orientationApplied(bmp, sn))) {
      const r = await orientBitmap(bmp, sn.orientation);
      bmp.close();
      bmp = r.bmp;
      full = r.full;
    }
    measurePixels(bmp, m, full ? undefined : [m.width, m.height]);
    // Upright only as a smaller copy: measured, and the server's
    // bake_orientation turns the file itself upright.
    if (!full) return { metrics: m };
    if (!opts.fix || !['jpeg', 'png', 'webp'].includes(sn.format)) return { metrics: m };
    let fixed: { blob: Blob; name: string; what: FixWhat } | null = null;
    try {
      fixed = await fixOf(bmp, file.name, sn, m);
    } catch {
      fixed = null; // a canvas too large for this browser: upload as it is
    }
    if (!fixed) return { metrics: m };
    const again = await measure(new File([fixed.blob], fixed.name, { type: fixed.blob.type, lastModified: file.lastModified }), {
      sha: opts.sha,
      fix: false,
    });
    // The fixed file is a canvas encode: it keeps the capture time and camera
    // only as a JPEG, and never the Software tag, PNG text chunks or C2PA. What
    // the original said about where it came from is carried over.
    const fm = again.metrics;
    if (m.generated) fm.generated = true;
    if (m.exif) fm.exif = { ...m.exif, ...fm.exif };
    return { metrics: fm, fixed };
  } finally {
    bmp.close();
  }
}

/** What the header alone says (photocheck.headerMetrics). */
function headerMetrics(sn: Sniffed, file: File): PhotoMetrics {
  const o = sn.orientation >= 1 && sn.orientation <= 8 ? sn.orientation : 1;
  let w = sn.width ?? 0, h = sn.height ?? 0;
  if (o >= 5) [w, h] = [h, w];
  const m: PhotoMetrics = {
    version: 1,
    file: file.name,
    bytes: file.size,
    format: sn.format,
    width: w,
    height: h,
    orientation: o,
    decoded: false,
    source: 'browser',
  };
  // The decoded pixel model, as Go's decoders would name it.
  if (sn.cmyk) m.mode = 'cmyk';
  else if (sn.format === 'jpeg' && sn.jpegComponents === 1) m.mode = 'gray';
  else if (sn.format === 'png' && sn.pngColorType === 0) m.mode = sn.pngBitDepth === 16 ? 'gray16' : 'gray';
  else if (sn.format === 'png' && sn.pngColorType === 3) m.mode = 'paletted';
  if (sn.animated) m.animated = true;
  if (sn.jpegQ) m.jpeg_q = sn.jpegQ;
  if (sn.pngColorType !== undefined && sn.format === 'png') {
    if (sn.pngColorType) m.png_color_type = sn.pngColorType;
    if (sn.pngBitDepth) m.png_bit_depth = sn.pngBitDepth;
  }
  if (sn.exif) m.exif = sn.exif;
  if (sn.generated) m.generated = true;
  return m;
}

/**
 * Fills the pixel metrics of an upright bitmap into m. size is the photo's
 * upright size when bmp is a smaller copy of it.
 */
function measurePixels(bmp: ImageBitmap, m: PhotoMetrics, size?: [number, number]) {
  const [w, h] = size ?? [bmp.width, bmp.height];
  m.width = w;
  m.height = h;
  m.decoded = true;
  m.mode ??= 'rgb';

  // S: luma of the canvas-drawn copy; alpha alongside.
  const [sw, sh] = fitArea(w, h, sharpEdge);
  const sp = pixels(bmp, sw, sh);
  const n = sw * sh;
  const s = new Float32Array(n);
  const alpha = new Float32Array(n);
  let opaque = true;
  for (let i = 0; i < n; i++) {
    const r = sp[4 * i], g = sp[4 * i + 1], b = sp[4 * i + 2], a = sp[4 * i + 3];
    s[i] = (19595 * r + 38470 * g + 7471 * b + 32768) >>> 16;
    alpha[i] = a;
    if (a !== 255) opaque = false;
  }
  const [ww, wh] = fitLong(sw, sh, workLong);
  const wk = boxDown(s, sw, sh, ww, wh);

  m.sharp = r4(sharpness(s, sw, sh));
  const d = dhash(wk, ww, wh);
  m.dhash = d.h;
  m.dhash_flip = d.flip;
  m.thumb32 = thumb32(wk, ww, wh);
  const ex = exposure(wk, ww, wh);
  m.luma_mean = r4(ex.mean);
  m.luma_std = r4(ex.std);
  m.clip_lo = r4(ex.clipLo);
  m.clip_hi = r4(ex.clipHi);
  m.clip_hi_center = r4(ex.clipHiCenter);
  m.flat = r4(flatShare(wk));
  m.bars = r4(bars(wk, ww, wh));
  if (uiBand(wk, ww, wh)) m.ui_band = true;

  // C: drawn straight to a 128 canvas.
  const [cw, ch] = fitLong(w, h, colourLong);
  const cp = pixels(bmp, cw, ch);
  const cn = cw * ch;
  const cr = new Float32Array(cn), cg = new Float32Array(cn), cb = new Float32Array(cn);
  for (let i = 0; i < cn; i++) {
    cr[i] = cp[4 * i];
    cg[i] = cp[4 * i + 1];
    cb[i] = cp[4 * i + 2];
  }
  m.colorful = r4(colorfulness(cr, cg, cb));
  m.chroma_spread = r4(chromaSpread(cr, cg, cb));
  m.layout = toBase64(layout(cr, cg, cb, cw, ch));

  if (!opaque) {
    const a = boxDown(alpha, sw, sh, ww, wh);
    let t = 0;
    for (const v of a) if (v < 250) t++;
    m.alpha_transparent = r4(t / a.length);
  }
}

/** The silent fix this photo needs, if any (PHOTO-PLAN 13.3 step 3). */
async function fixOf(bmp: ImageBitmap, name: string, sn: Sniffed, m: PhotoMetrics): Promise<{ blob: Blob; name: string; what: FixWhat } | null> {
  const stem = name.replace(/\.[^.]+$/, '');
  const exif = { time: sn.raw?.time, subsec: sn.raw?.subsec, make: sn.raw?.make, model: sn.raw?.model };
  if ((m.alpha_transparent ?? 0) > transparentShare) {
    const jpeg = await encode(bmp, 'image/jpeg', 0.95, '#fff');
    return { blob: await withExif(jpeg, exif), name: stem + '.jpg', what: 'flattened' };
  }
  if (sn.format === 'png' && sn.pngColorType === 0 && sn.pngBitDepth === 16) {
    return { blob: await encode(bmp, 'image/png'), name, what: 'gray16' };
  }
  if (sn.orientation !== 1) {
    if (sn.format === 'jpeg') {
      const jpeg = await encode(bmp, 'image/jpeg', 0.95);
      return { blob: await withExif(jpeg, exif), name, what: 'upright' };
    }
    return { blob: await encode(bmp, 'image/png'), name: stem + '.png', what: 'upright' };
  }
  return null;
}

/**
 * Whether createImageBitmap applied the EXIF orientation. imageOrientation
 * 'from-image' is not honoured for every format (Chrome leaves WebP as
 * stored), so it is checked: by the size for orientations 5-8 of a photo
 * that is not square, else by decoding a tiny probe of the same format.
 */
async function orientationApplied(bmp: ImageBitmap, sn: Sniffed): Promise<boolean> {
  const w = sn.width ?? 0, h = sn.height ?? 0;
  if (sn.orientation >= 5 && w && h && w !== h) {
    if (bmp.width === h && bmp.height === w) return true;
    if (bmp.width === w && bmp.height === h) return false;
  }
  return probeOrientation(sn.format);
}

// 2x1 images with EXIF Orientation 6: an upright decode is 1x2.
const orientationProbes: Partial<Record<Sniffed['format'], [string, string]>> = {
  jpeg: [
    'image/jpeg',
    '/9j/4AAQSkZJRgABAQAAAQABAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAYAAAAAAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCAABAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwAooooA/9k=',
  ],
  png: [
    'image/png',
    'iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAGmVYSWZNTQAqAAAACAABARIAAwAAAAEABgAAAAAAANZnS2kAAAAPSURBVHicY2xoaGBgYAAAB44BguWmdUQAAAAASUVORK5CYII=',
  ],
  webp: ['image/webp', 'UklGRlIAAABXRUJQVlA4WAoAAAAIAAAAAQAAAAAAVlA4TBEAAAAvAQAAAAdQwAIWsP+BiOh/AABFWElGGgAAAE1NACoAAAAIAAEBEgADAAAAAQAGAAAAAAAA'],
};
const probed = new Map<string, Promise<boolean>>();

/** Whether this browser applies EXIF orientation to format (once per format; true when unknown). */
function probeOrientation(format: Sniffed['format']): Promise<boolean> {
  let p = probed.get(format);
  if (!p) {
    const probe = orientationProbes[format];
    p = !probe
      ? Promise.resolve(true)
      : (async () => {
          try {
            const bytes = Uint8Array.from(atob(probe[1]), c => c.charCodeAt(0));
            const b = await createImageBitmap(new Blob([bytes], { type: probe[0] }), { imageOrientation: 'from-image' });
            const ok = b.width === 1 && b.height === 2;
            b.close();
            return ok;
          } catch {
            return true;
          }
        })();
    probed.set(format, p);
  }
  return p;
}

/** The canvas transform that draws a w x h image stored with EXIF orientation o upright (photocheck.orient). */
function orientMatrix(o: number, w: number, h: number): [number, number, number, number, number, number] {
  switch (o) {
    case 2: return [-1, 0, 0, 1, w, 0]; // mirror
    case 3: return [-1, 0, 0, -1, w, h]; // rotate 180
    case 4: return [1, 0, 0, -1, 0, h]; // flip
    case 5: return [0, 1, 1, 0, 0, 0]; // transpose
    case 6: return [0, 1, -1, 0, h, 0]; // rotate 90 clockwise
    case 7: return [0, -1, -1, 0, h, w]; // transverse
    case 8: return [0, -1, 1, 0, 0, w]; // rotate 90 counter-clockwise
    default: return [1, 0, 0, 1, 0, 0];
  }
}

/**
 * bmp (as stored) drawn upright through EXIF orientation o. full is false
 * when the browser had no canvas that large and the copy is at most
 * 4096 px on its long side.
 */
async function orientBitmap(bmp: ImageBitmap, o: number): Promise<{ bmp: ImageBitmap; full: boolean }> {
  const draw = async (f: number) => {
    const w = Math.max(1, Math.round(bmp.width * f)), h = Math.max(1, Math.round(bmp.height * f));
    const { c, ctx } = canvas(o >= 5 ? h : w, o >= 5 ? w : h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.setTransform(...orientMatrix(o, w, h));
    ctx.drawImage(bmp, 0, 0, w, h);
    return createImageBitmap(c);
  };
  try {
    return { bmp: await draw(1), full: true };
  } catch {
    return { bmp: await draw(Math.min(1, 4096 / Math.max(bmp.width, bmp.height))), full: false };
  }
}

async function withExif(jpeg: Blob, e: Parameters<typeof injectExif>[1]): Promise<Blob> {
  const out = injectExif(new Uint8Array(await jpeg.arrayBuffer()), e);
  return new Blob([out as Uint8Array<ArrayBuffer>], { type: 'image/jpeg' });
}

type Canvas2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

/** A w x h canvas (OffscreenCanvas when there is one) and its 2D context. */
function canvas(w: number, h: number): { c: OffscreenCanvas | HTMLCanvasElement; ctx: Canvas2D } {
  let c: OffscreenCanvas | HTMLCanvasElement;
  if (typeof OffscreenCanvas !== 'undefined') c = new OffscreenCanvas(w, h);
  else {
    c = document.createElement('canvas');
    c.width = w;
    c.height = h;
  }
  const ctx = c.getContext('2d', { willReadFrequently: true }) as Canvas2D | null;
  if (!ctx) throw new Error('no 2d canvas');
  return { c, ctx };
}

/** RGBA of bmp drawn at w x h with high-quality smoothing. */
function pixels(bmp: ImageBitmap, w: number, h: number): Uint8ClampedArray {
  const { ctx } = canvas(w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

/** bmp at full size as an image file, over a background colour when one is given. */
async function encode(bmp: ImageBitmap, type: string, quality?: number, background?: string): Promise<Blob> {
  const { c, ctx } = canvas(bmp.width, bmp.height);
  if (background) {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, bmp.width, bmp.height);
  }
  ctx.drawImage(bmp, 0, 0);
  if ('convertToBlob' in c) return c.convertToBlob({ type, quality });
  const blob = await new Promise<Blob | null>(res => c.toBlob(res, type, quality));
  if (!blob) throw new Error('encode failed');
  return blob;
}

/** Lowercase hex sha256 of the file, or undefined where crypto.subtle is missing (plain http). */
async function sha256(file: File): Promise<string | undefined> {
  try {
    if (!globalThis.crypto?.subtle) return undefined;
    const d = new Uint8Array(await crypto.subtle.digest('SHA-256', await file.arrayBuffer()));
    return Array.from(d, b => b.toString(16).padStart(2, '0')).join('');
  } catch {
    return undefined;
  }
}

// ---- section 3 arithmetic (photocheck/resample.go, dhash.go, metrics.go) ----

/**
 * Area-averages a w x h float plane (row-major) to dw x dh: every output
 * pixel is the mean of the source area it covers, with fractional weights on
 * its edges; rows first into a float32 buffer, then columns, sums in float64.
 * The same arithmetic as photocheck.BoxDown, so both give the same bits.
 */
export function boxDown(src: Float32Array, w: number, h: number, dw: number, dh: number): Float32Array {
  if (w <= 0 || h <= 0 || dw <= 0 || dh <= 0 || src.length < w * h) return new Float32Array(0);
  const ax = boxAxis(w, dw), ay = boxAxis(h, dh);
  const tmp = new Float32Array(h * dw);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let i = 0; i < dw; i++) {
      let j = row + ax.start[i];
      let s = 0;
      for (let k = ax.off[i]; k < ax.off[i + 1]; k++) s += ax.wts[k] * src[j++];
      tmp[y * dw + i] = s;
    }
  }
  const dst = new Float32Array(dw * dh);
  const acc = new Float64Array(dw);
  for (let i = 0; i < dh; i++) {
    acc.fill(0);
    let y = ay.start[i];
    for (let k = ay.off[i]; k < ay.off[i + 1]; k++) {
      const wt = ay.wts[k];
      const row = y * dw;
      for (let x = 0; x < dw; x++) acc[x] += wt * tmp[row + x];
      y++;
    }
    dst.set(acc, i * dw);
  }
  return dst;
}

/** The weight table of one axis (photocheck.newBoxAxis). */
function boxAxis(n: number, m: number) {
  const start = new Int32Array(m);
  const off = new Int32Array(m + 1);
  const wts: number[] = [];
  const inv = m / n;
  for (let i = 0; i < m; i++) {
    const lo = (i * n) / m;
    const hi = ((i + 1) * n) / m;
    const j0 = Math.floor(lo);
    let j1 = Math.min(Math.ceil(hi), n);
    if (j1 <= j0) j1 = j0 + 1;
    start[i] = j0;
    for (let j = j0; j < j1; j++) wts.push((Math.min(hi, j + 1) - Math.max(lo, j)) * inv);
    off[i + 1] = wts.length;
  }
  return { start, off, wts };
}

/**
 * The 64-bit difference hash of a working copy as 16 hex chars: the copy
 * area-averaged to 9x8, bit y*8+x (from the top bit) set when P(x,y) >
 * P(x+1,y); flip is the same over the column-mirrored grid.
 */
export function dhash(w: Float32Array, ww: number, wh: number): { h: string; flip: string } {
  const p = boxDown(w, ww, wh, 9, 8);
  if (p.length !== 72) return { h: '0000000000000000', flip: '0000000000000000' };
  let h = 0n, flip = 0n;
  for (let y = 0; y < 8; y++) {
    const r = y * 9;
    for (let x = 0; x < 8; x++) {
      const bit = 1n << BigInt(63 - (y * 8 + x));
      if (p[r + x] > p[r + x + 1]) h |= bit;
      if (p[r + 8 - x] > p[r + 7 - x]) flip |= bit;
    }
  }
  return { h: h.toString(16).padStart(16, '0'), flip: flip.toString(16).padStart(16, '0') };
}

/** The 32x32 luma thumbnail (aspect squashed), rounded to uint8, as base64. */
export function thumb32(w: Float32Array, ww: number, wh: number): string {
  const p = boxDown(w, ww, wh, 32, 32);
  const out = new Uint8Array(p.length);
  for (let i = 0; i < p.length; i++) out[i] = clamp8(p[i]);
  return toBase64(out);
}

/** The size whose area-equivalent edge is at most edge; never grows. */
export function fitArea(w: number, h: number, edge: number): [number, number] {
  const f = Math.min(1, edge / Math.sqrt(w * h));
  return [Math.max(1, Math.round(w * f)), Math.max(1, Math.round(h * f))];
}

/** The size whose long side is at most long; never grows. */
export function fitLong(w: number, h: number, long: number): [number, number] {
  const l = Math.max(w, h);
  if (l <= long) return [w, h];
  const f = long / l;
  return [Math.max(1, Math.round(w * f)), Math.max(1, Math.round(h * f))];
}

/** Nearest-rank p90 of per-tile Laplacian variance on S after a 3x3 Gaussian (valid regions only). */
function sharpness(s: Float32Array, w: number, h: number): number {
  if (w < 5 || h < 5) return 0;
  const gw = w - 2, gh = h - 2;
  const t = new Float32Array(gw * h);
  for (let y = 0; y < h; y++) {
    const r = y * w;
    for (let x = 0; x < gw; x++) t[y * gw + x] = (s[r + x] + 2 * s[r + x + 1] + s[r + x + 2]) / 4;
  }
  const g = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) g[y * gw + x] = (t[y * gw + x] + 2 * t[(y + 1) * gw + x] + t[(y + 2) * gw + x]) / 4;
  }
  const lw = gw - 2, lh = gh - 2;
  const nx = Math.min(Math.max(Math.floor(w / 32), 1), 8), ny = Math.min(Math.max(Math.floor(h / 32), 1), 8);
  const vals: number[] = [];
  for (let ty = 0; ty < ny; ty++) {
    const y0 = Math.floor((ty * lh) / ny), y1 = Math.floor(((ty + 1) * lh) / ny);
    for (let tx = 0; tx < nx; tx++) {
      const x0 = Math.floor((tx * lw) / nx), x1 = Math.floor(((tx + 1) * lw) / nx);
      let sum = 0, sq = 0, n = 0;
      for (let y = y0; y < y1; y++) {
        const c = (y + 1) * gw;
        for (let x = x0; x < x1; x++) {
          const i = c + x + 1;
          const l = g[i - 1] + g[i + 1] + g[i - gw] + g[i + gw] - 4 * g[i];
          sum += l;
          sq += l * l;
          n++;
        }
      }
      if (n > 0) {
        const mean = sum / n;
        vals.push(Math.max(0, sq / n - mean * mean));
      }
    }
  }
  if (!vals.length) return 0;
  vals.sort((a, b) => a - b);
  return vals[Math.ceil(0.9 * vals.length) - 1];
}

/** Mean, std, shares at or below 5 and at or above 250, and the last inside the central 60% x 60%. */
function exposure(wk: Float32Array, w: number, h: number) {
  const e = { mean: 0, std: 0, clipLo: 0, clipHi: 0, clipHiCenter: 0 };
  const n = wk.length;
  if (!n) return e;
  let sum = 0, sq = 0, lo = 0, hi = 0;
  for (const v of wk) {
    sum += v;
    sq += v * v;
    if (v <= 5) lo++;
    if (v >= 250) hi++;
  }
  e.mean = sum / n;
  e.std = Math.sqrt(Math.max(0, sq / n - e.mean * e.mean));
  e.clipLo = lo / n;
  e.clipHi = hi / n;
  const [x0, x1] = centre(w), [y0, y1] = centre(h);
  let hc = 0, nc = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (wk[y * w + x] >= 250) hc++;
      nc++;
    }
  }
  if (nc) e.clipHiCenter = hc / nc;
  return e;
}

function centre(n: number): [number, number] {
  const a = Math.round(0.2 * n), b = Math.round(0.8 * n);
  return b <= a ? [Math.floor(n / 2), Math.floor(n / 2) + 1] : [a, b];
}

/** Share of W in its 6 largest of 32 luma bins (Y>>3). */
function flatShare(wk: Float32Array): number {
  if (!wk.length) return 0;
  const bins = new Array<number>(32).fill(0);
  for (const v of wk) bins[clamp8(v) >> 3]++;
  bins.sort((a, b) => a - b);
  return bins.slice(26).reduce((a, b) => a + b, 0) / wk.length;
}

// Bar detection (section 3 "bars").
const bar = { std: 3, dark: 24, light: 232, maxFrac: 0.25, pairMin: 0.02, pairTol: 0.25, oneMin: 0.04, step: 40 };

type BarScan = { frac: number; n: number; light: boolean; mean: number; inner: number };

/** How far uniform dark or light lines reach in from one edge; line(i) is the i-th line's mean and std. */
function scanBar(total: number, line: (i: number) => [number, number]): BarScan {
  const s: BarScan = { frac: 0, n: 0, light: false, mean: 0, inner: NaN };
  const limit = Math.floor(bar.maxFrac * total);
  let sum = 0;
  for (let i = 0; i < limit; i++) {
    const [m, sd] = line(i);
    if (sd >= bar.std || (m >= bar.dark && m <= bar.light)) break;
    const light = m > bar.light;
    if (i === 0) s.light = light;
    else if (light !== s.light) break;
    sum += m;
    s.n++;
  }
  s.frac = s.n / total;
  if (s.n > 0) {
    s.mean = sum / s.n;
    if (s.n + 2 <= total) s.inner = (line(s.n)[0] + line(s.n + 1)[0]) / 2;
  }
  return s;
}

/** Area share of W under counted edge bars: a matching pair, or one edge with a sharp step inward. */
function bars(wk: Float32Array, w: number, h: number): number {
  if (w < 4 || h < 4) return 0;
  const row = (y: number) => lineStats(wk, y * w, 1, w);
  const col = (x: number) => lineStats(wk, x, w, h);
  const [t, b] = countBars(scanBar(h, row), scanBar(h, i => row(h - 1 - i)));
  const [l, r] = countBars(scanBar(w, col), scanBar(w, i => col(w - 1 - i)));
  return 1 - (1 - t - b) * (1 - l - r);
}

function countBars(a: BarScan, b: BarScan): [number, number] {
  if (a.frac >= bar.pairMin && b.frac >= bar.pairMin && a.light === b.light && Math.abs(a.frac - b.frac) <= bar.pairTol * Math.max(a.frac, b.frac)) {
    return [a.frac, b.frac];
  }
  const single = (s: BarScan) => (s.frac >= bar.oneMin && !Number.isNaN(s.inner) && Math.abs(s.inner - s.mean) >= bar.step ? s.frac : 0);
  return [single(a), single(b)];
}

function lineStats(v: Float32Array, start: number, step: number, n: number): [number, number] {
  let sum = 0, sq = 0;
  for (let i = 0; i < n; i++) {
    const f = v[start + i * step];
    sum += f;
    sq += f * f;
  }
  const m = sum / n;
  return [m, Math.sqrt(Math.max(0, sq / n - m * m))];
}

/** A phone status bar along the top of W (photocheck.uiBand). */
function uiBand(wk: Float32Array, w: number, h: number): boolean {
  const rows = Math.min(h, Math.max(8, Math.round(0.04 * h)));
  const band = wk.subarray(0, rows * w);
  if (!band.length) return false;
  const sorted = Float32Array.from(band).sort();
  const half = sorted.length >> 1;
  const med = sorted.length % 2 === 0 ? (sorted[half - 1] + sorted[half]) / 2 : sorted[half];
  const dirty = new Uint8Array(w);
  let d = 0;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < w; x++) {
      if (Math.abs(band[y * w + x] - med) > 48) {
        d++;
        dirty[x] = 1;
      }
    }
  }
  const share = d / band.length;
  if (share < 0.01 || 1 - share < 0.85) return false;
  let runs = 0, gap = 2;
  for (const dx of dirty) {
    if (dx) {
      if (gap >= 2) runs++;
      gap = 0;
    } else gap++;
  }
  return runs >= 3;
}

/** Hasler-Süsstrunk M3 on C. */
function colorfulness(r: Float32Array, g: Float32Array, b: Float32Array): number {
  const n = r.length;
  if (!n) return 0;
  let srg = 0, syb = 0, qrg = 0, qyb = 0;
  for (let i = 0; i < n; i++) {
    const rg = r[i] - g[i];
    const yb = (r[i] + g[i]) / 2 - b[i];
    srg += rg;
    syb += yb;
    qrg += rg * rg;
    qyb += yb * yb;
  }
  const mrg = srg / n, myb = syb / n;
  return Math.sqrt(Math.max(0, qrg / n - mrg * mrg) + Math.max(0, qyb / n - myb * myb)) + 0.3 * Math.sqrt(mrg * mrg + myb * myb);
}

/** sqrt(var(Cb) + var(Cr)) on C, BT.601 full range. */
function chromaSpread(r: Float32Array, g: Float32Array, b: Float32Array): number {
  const n = r.length;
  if (!n) return 0;
  let scb = 0, scr = 0, qcb = 0, qcr = 0;
  for (let i = 0; i < n; i++) {
    const cb = 128 - 0.168736 * r[i] - 0.331264 * g[i] + 0.5 * b[i];
    const cr = 128 + 0.5 * r[i] - 0.418688 * g[i] - 0.081312 * b[i];
    scb += cb;
    scr += cr;
    qcb += cb * cb;
    qcr += cr * cr;
  }
  const mcb = scb / n, mcr = scr / n;
  return Math.sqrt(Math.max(0, qcb / n - mcb * mcb) + Math.max(0, qcr / n - mcr * mcr));
}

const srgbLinear = Array.from({ length: 256 }, (_, i) => {
  const c = i / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
});

function linear(v: number): number {
  const f = Math.min(255, Math.max(0, v));
  const i = Math.floor(f);
  if (i >= 255) return srgbLinear[255];
  const t = f - i;
  return srgbLinear[i] * (1 - t) + srgbLinear[i + 1] * t;
}

function labF(t: number): number {
  const d = 6 / 29;
  return t > d * d * d ? Math.cbrt(t) : t / (3 * d * d) + 4 / 29;
}

/** The 4x4 grid of mean CIELAB (sRGB, D65) on C: round(L*2.55), a+128, b+128 per cell. */
function layout(r: Float32Array, g: Float32Array, b: Float32Array, w: number, h: number): Uint8Array {
  const n = r.length;
  const L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const R = linear(r[i]), G = linear(g[i]), Bl = linear(b[i]);
    const x = (0.4124564 * R + 0.3575761 * G + 0.1804375 * Bl) / 0.95047;
    const y = 0.2126729 * R + 0.7151522 * G + 0.072175 * Bl;
    const z = (0.0193339 * R + 0.119192 * G + 0.9503041 * Bl) / 1.08883;
    const fx = labF(x), fy = labF(y), fz = labF(z);
    L[i] = 116 * fy - 16;
    A[i] = 500 * (fx - fy);
    B[i] = 200 * (fy - fz);
  }
  const gl = boxDown(L, w, h, 4, 4), ga = boxDown(A, w, h, 4, 4), gb = boxDown(B, w, h, 4, 4);
  const out = new Uint8Array(3 * gl.length);
  for (let i = 0; i < gl.length; i++) {
    out[3 * i] = clamp8(Math.fround(gl[i] * 2.55));
    out[3 * i + 1] = clamp8(ga[i] + 128);
    out[3 * i + 2] = clamp8(gb[i] + 128);
  }
  return out;
}

function clamp8(v: number): number {
  return v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v);
}

/** Rounds to 4 decimals, as the server stores metrics. */
function r4(v: number): number {
  return Math.round(v * 1e4) / 1e4;
}

function toBase64(b: Uint8Array): string {
  let s = '';
  for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
  return btoa(s);
}

// As a dedicated worker: measure each file posted, with the silent fixes.
type WorkerScope = { addEventListener(t: 'message', fn: (e: MessageEvent<WorkIn>) => void): void; postMessage(m: WorkOut): void };
const g = globalThis as unknown as { WorkerGlobalScope?: new () => unknown };
if (typeof g.WorkerGlobalScope === 'function' && globalThis instanceof g.WorkerGlobalScope) {
  const scope = globalThis as unknown as WorkerScope;
  scope.addEventListener('message', async e => {
    const { id, file, sha } = e.data;
    try {
      scope.postMessage({ id, ...(await measure(file, { sha, fix: true })) });
    } catch (err) {
      scope.postMessage({ id, error: err instanceof Error ? err.message : String(err) });
    }
  });
}

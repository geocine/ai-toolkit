// What the trainer reads (toolkit/data_loader.py), and getting dropped photos
// ready to upload: formats it cannot read (HEIC, AVIF, ...) become JPEGs in
// the browser, every photo is measured for the server's photo check, and the
// fixes nobody needs to be asked about happen on the way (PHOTO-PLAN 13.3).

import { headSize, injectExif, parseTIFF, sniff, type Exif } from './exif';
import { heifExif } from './heif';
import type { PhotoMetrics } from './photoTypes';
import { measureFiles } from './photos';

export const trainable = /\.(jpe?g|png|webp)$/i;
const convertible = /\.(avif|heic|heif|bmp|gif|tiff?)$/i;

/** What prepareImages did to a file, for the quiet note under the drop. */
export type PrepNote = { file: string; what: 'converted' | 'upright' | 'gray16' | 'flattened' | 'renamed' | 'server_convert'; to?: string };

const stemOf = (name: string) => name.replace(/\.[^.]+$/, '');
const extOf = (name: string) => (/\.[^.]+$/.exec(name)?.[0] ?? '');

/** The review.convert name of a file's format (tif is tiff). */
const formatOf = (name: string) => extOf(name).slice(1).toLowerCase().replace(/^tif$/, 'tiff');

/** Whether the server converts a file, by review.convert (photocheck judge.converts: heif also when heic is listed). */
const serverConverts = (server: Set<string>, name: string) => {
  const f = formatOf(name);
  return server.has(f) || (f === 'heif' && server.has('heic'));
};

/** EXIF worth keeping across a conversion: the capture time and the camera (HEIF: from its Exif item). */
async function exifOf(f: File): Promise<Exif | null> {
  try {
    const bytes = new Uint8Array(await f.arrayBuffer());
    const sn = sniff(bytes.subarray(0, headSize));
    if (sn.raw) return sn.raw;
    if (sn.format === 'heic' || sn.format === 'heif' || sn.format === 'avif') {
      const tiff = heifExif(bytes);
      return tiff ? parseTIFF(tiff) : null;
    }
  } catch {
    /* unreadable: converted without EXIF */
  }
  return null;
}

/** JPEG copy of an image the browser can decode (HEIC: Safari only), EXIF rotation applied, date and camera kept. */
async function toJpeg(f: File): Promise<File> {
  const bmp = await createImageBitmap(f, { imageOrientation: 'from-image' });
  const canvas = document.createElement('canvas');
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('no canvas');
  ctx.fillStyle = '#fff'; // transparent GIF/AVIF areas
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const blob = await new Promise<Blob | null>(res => canvas.toBlob(res, 'image/jpeg', 0.92));
  if (!blob) throw new Error('encode failed');
  let out: Blob = blob;
  const e = await exifOf(f);
  if (e) {
    const jpeg = injectExif(new Uint8Array(await blob.arrayBuffer()), { time: e.time, subsec: e.subsec, make: e.make, model: e.model });
    out = new Blob([jpeg as Uint8Array<ArrayBuffer>], { type: 'image/jpeg' });
  }
  return new File([out], stemOf(f.name) + '.jpg', { type: 'image/jpeg', lastModified: f.lastModified });
}

/**
 * Gets picked files ready to upload, in four steps:
 * 1. formats the trainer skips become JPEG where this browser decodes them;
 *    otherwise they go to serverConvert when the server converts that format
 *    (opts.serverConverts, from review.convert), else to skipped;
 * 2. every photo is measured (metrics, in the order of ok's photos);
 * 3. silent fixes: upright, 16-bit grey to 8-bit, transparency on white;
 * 4. a name already in the selection or in opts.taken becomes name_2.ext,
 *    and its caption .txt follows it.
 * Each change is listed in notes.
 */
export async function prepareImages(
  files: File[],
  opts: { taken: Set<string>; serverConverts: string[]; onProgress?(done: number, total: number): void } = { taken: new Set(), serverConverts: [] },
): Promise<{ ok: File[]; skipped: string[]; serverConvert: File[]; notes: PrepNote[]; metrics: PhotoMetrics[] }> {
  const notes: PrepNote[] = [];
  const skipped: string[] = [];
  const photos: File[] = [];
  const captions: File[] = [];
  let serverConvert: File[] = [];
  const server = new Set(opts.serverConverts.map(s => s.toLowerCase()));

  // 1. Formats.
  for (const f of files) {
    if (/\.txt$/i.test(f.name)) captions.push(f);
    else if (trainable.test(f.name)) photos.push(f);
    else if (convertible.test(f.name)) {
      try {
        const j = await toJpeg(f);
        photos.push(j);
        notes.push({ file: f.name, what: 'converted', to: j.name });
      } catch {
        if (serverConverts(server, f.name)) {
          serverConvert.push(f);
          notes.push({ file: f.name, what: 'server_convert' });
        } else skipped.push(f.name);
      }
    } else skipped.push(f.name);
  }

  // 2 and 3. Measure, with the silent fixes.
  const results = await measureFiles(photos, (d, t) => opts.onProgress?.(d, t));
  const metrics: (PhotoMetrics | null)[] = [];
  const ready = photos.map((f, i) => {
    const r = results[i];
    if (!r || 'error' in r) {
      metrics.push(null);
      return f;
    }
    metrics.push(r.metrics);
    if (!r.fixed) return f;
    notes.push({ file: f.name, what: r.fixed.what, ...(r.fixed.name !== f.name ? { to: r.fixed.name } : {}) });
    return new File([r.fixed.blob], r.fixed.name, { type: r.fixed.blob.type, lastModified: f.lastModified });
  });

  // 4. Names. Photos (and raw uploads) first, in order; then each caption
  // takes the name of the photo with its stem, the k-th caption the k-th photo.
  const used = new Set(opts.taken);
  const finalStems = new Map<string, string[]>();
  const rename = (f: File): File => {
    let name = f.name;
    const stem = stemOf(name), ext = extOf(name);
    for (let k = 2; used.has(name); k++) name = `${stem}_${k}${ext}`;
    used.add(name);
    finalStems.set(stem, [...(finalStems.get(stem) ?? []), stemOf(name)]);
    if (name === f.name) return f;
    notes.push({ file: f.name, what: 'renamed', to: name });
    return new File([f], name, { type: f.type, lastModified: f.lastModified });
  };
  const ok = ready.map((f, i) => {
    const g = rename(f);
    const m = metrics[i];
    if (m) m.file = g.name;
    return g;
  });
  serverConvert = serverConvert.map(rename);
  const seen = new Map<string, number>();
  for (const c of captions) {
    const stem = stemOf(c.name);
    const k = seen.get(stem) ?? 0;
    seen.set(stem, k + 1);
    const to = finalStems.get(stem)?.[k];
    if (to === undefined || to === stem) {
      ok.push(c);
      continue;
    }
    const name = to + extOf(c.name);
    notes.push({ file: c.name, what: 'renamed', to: name });
    ok.push(new File([c], name, { type: c.type, lastModified: c.lastModified }));
  }
  return { ok, skipped, serverConvert, notes, metrics: metrics.filter((m): m is PhotoMetrics => m !== null) };
}

/** Batches of at most maxFiles files and maxBytes bytes (a proxy in front of a pod caps request bodies). */
export function batches(files: File[], maxFiles = 25, maxBytes = 64 << 20): File[][] {
  const out: File[][] = [];
  let cur: File[] = [];
  let size = 0;
  for (const f of files) {
    if (cur.length && (cur.length >= maxFiles || size + f.size > maxBytes)) {
      out.push(cur);
      cur = [];
      size = 0;
    }
    cur.push(f);
    size += f.size;
  }
  if (cur.length) out.push(cur);
  return out;
}

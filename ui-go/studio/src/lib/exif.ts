// What the first bytes of a photo say about it, read in the browser before
// upload: the same reader as internal/imgmeta (sniff.go, exif.go, dqt.go)
// and the same EXIF writer (write.go). EXIF orientation, capture time,
// camera, software; PNG pixel format and text keys; WebP animation; JPEG
// components and quality; the C2PA "made by a model" marker.
//
// The bytes are untrusted: every read is bounds-checked and nothing throws.

import { heifExif, heifFormat, heifSize } from './heif';
import type { PhotoExif, PhotoMetrics } from './photoTypes';

/** How much of a file sniff expects (imgmeta.HeadSize). */
export const headSize = 256 << 10;

/** EXIF as imgmeta.Exif: time is "2006-01-02T15:04:05" (camera-local), subsec its digits. */
export interface Exif {
  orientation: number;
  time: string;
  subsec: string;
  make: string;
  model: string;
  software: string;
}

export interface Sniffed {
  format: PhotoMetrics['format'];
  orientation: number; // 1..8, 1 when absent
  exif?: PhotoExif; // time carries a .123 fraction when SubSecTimeOriginal is set
  pngColorType?: number;
  pngBitDepth?: number;
  animated?: boolean;
  generated?: boolean;
  jpegQ?: number; // 0 or absent: unknown
  cmyk?: boolean;
  jpegComponents?: number; // SOF component count: 1 grey, 3 colour, 4 CMYK
  width?: number; // as stored, before orientation
  height?: number;
  raw?: Exif; // the parsed EXIF, subsec separate
}

const tag = {
  width: 0x0100,
  length: 0x0101,
  make: 0x010f,
  model: 0x0110,
  orientation: 0x0112,
  software: 0x0131,
  exifIFD: 0x8769,
  dateTimeOriginal: 0x9003,
  subSecOriginal: 0x9291,
};
const typeASCII = 2, typeShort = 3, typeLong = 4;
const maxIFDEntries = 512;
const maxStringLen = 255;

const generatorSoftware = /stable diffusion|midjourney|dall.?e|comfyui|firefly|imagen|flux|novelai/i;
const generatorKeys = new Set(['parameters', 'prompt', 'workflow']);

/** Reads format, size and metadata from the first bytes of a file (headSize is enough). */
export function sniff(head: Uint8Array): Sniffed {
  const s: Sniffed = { format: 'other', orientation: 1 };
  let raw: Exif | null = null;
  let c2pa = false;
  let software = '';
  if (head.length >= 3 && head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) {
    s.format = 'jpeg';
    const jumbf: Uint8Array[] = [];
    let jumbfLen = 0;
    walkJPEG(head, (m, seg) => {
      if (m === 0xe1 && !raw && startsWith(seg, exifPrefix)) {
        raw = parseTIFF(seg.subarray(6));
      } else if (m === 0xeb && seg.length > 8 && seg[0] === 0x4a && seg[1] === 0x50) {
        // APP11 JUMBF: "JP", box instance, sequence number, then the box
        // bytes; a manifest spans several segments.
        if (jumbfLen < headSize) {
          jumbf.push(seg.subarray(8));
          jumbfLen += seg.length - 8;
        }
      } else if (m === 0xdb && !s.jpegQ) {
        s.jpegQ = jpegQuality(seg);
      } else if (isSOF(m) && !s.jpegComponents && seg.length >= 6) {
        s.height = (seg[1] << 8) | seg[2];
        s.width = (seg[3] << 8) | seg[4];
        s.jpegComponents = seg[5];
        s.cmyk = seg[5] === 4;
      }
      return true;
    });
    c2pa = c2paGenerated(concat(jumbf));
  } else if (startsWith(head, pngSig)) {
    s.format = 'png';
    walkPNG(head, (type, data) => {
      switch (type) {
        case 'IHDR':
          if (data.length >= 10) {
            s.width = u32be(data, 0) & 0x7fffffff;
            s.height = u32be(data, 4) & 0x7fffffff;
            s.pngBitDepth = data[8];
            s.pngColorType = data[9];
          }
          break;
        case 'acTL': // APNG: only the first frame trains
          s.animated = true;
          break;
        case 'eXIf':
          raw = parseTIFF(data) ?? raw;
          break;
        case 'tEXt':
        case 'iTXt':
        case 'zTXt': {
          const nul = data.indexOf(0);
          const key = latin1(nul >= 0 ? data.subarray(0, nul) : data);
          let text: Uint8Array | null = nul >= 0 ? data.subarray(nul + 1) : null;
          if (generatorKeys.has(key.toLowerCase())) s.generated = true;
          // NovelAI writes no EXIF: its name sits in a "Software" text chunk.
          // zTXt and compressed iTXt text are not inflated.
          if (text && key.toLowerCase() === 'software' && type !== 'zTXt') {
            if (type === 'iTXt') text = itxtText(text);
            if (text && generatorSoftware.test(latin1(text))) s.generated = true;
          }
          break;
        }
        case 'caBX': // C2PA in PNG
          if (c2paGenerated(data)) c2pa = true;
          break;
      }
      return true;
    });
  } else if (head.length >= 12 && ascii(head, 0, 4) === 'RIFF' && ascii(head, 8, 4) === 'WEBP') {
    s.format = 'webp';
    walkWebP(head, (fourcc, data) => {
      switch (fourcc) {
        case 'VP8X':
          if (data.length >= 10) {
            if (data[0] & 0x02) s.animated = true;
            s.width = (data[4] | (data[5] << 8) | (data[6] << 16)) + 1;
            s.height = (data[7] | (data[8] << 8) | (data[9] << 16)) + 1;
          }
          break;
        case 'ANIM':
        case 'ANMF':
          s.animated = true;
          break;
        case 'VP8 ':
          // Frame tag (3 bytes), start code 9d 01 2a, then 14-bit sizes.
          if (!s.width && data.length >= 10 && data[3] === 0x9d && data[4] === 0x01 && data[5] === 0x2a) {
            s.width = (data[6] | (data[7] << 8)) & 0x3fff;
            s.height = (data[8] | (data[9] << 8)) & 0x3fff;
          }
          break;
        case 'VP8L':
          // Signature 0x2f, then width-1 and height-1 in 14 bits each.
          if (!s.width && data.length >= 5 && data[0] === 0x2f) {
            const v = data[1] | (data[2] << 8) | (data[3] << 16) | (data[4] << 24);
            s.width = (v & 0x3fff) + 1;
            s.height = ((v >>> 14) & 0x3fff) + 1;
          }
          break;
        case 'EXIF':
          raw = parseTIFF(startsWith(data, exifPrefix) ? data.subarray(6) : data) ?? raw;
          break;
        case 'C2PA':
          if (c2paGenerated(data)) c2pa = true;
          break;
      }
      return true;
    });
  } else if (ascii(head, 0, 6) === 'GIF87a' || ascii(head, 0, 6) === 'GIF89a') {
    s.format = 'gif';
    if (head.length >= 10) {
      s.width = head[6] | (head[7] << 8);
      s.height = head[8] | (head[9] << 8);
    }
  } else if (head.length >= 2 && head[0] === 0x42 && head[1] === 0x4d) {
    s.format = 'bmp';
    if (head.length >= 26) {
      const dv = new DataView(head.buffer, head.byteOffset, head.byteLength);
      s.width = Math.abs(dv.getInt32(18, true));
      s.height = Math.abs(dv.getInt32(22, true)); // negative = top-down
    }
  } else if (ascii(head, 0, 4) === 'II*\0' || ascii(head, 0, 4) === 'MM\0*') {
    s.format = 'tiff';
    const t = readTIFF(head);
    if (t) {
      raw = t.exif;
      s.width = t.w;
      s.height = t.h;
    }
  } else if (heifFormat(head)) {
    s.format = heifFormat(head) as Sniffed['format'];
    const size = heifSize(head);
    s.width = size.width;
    s.height = size.height;
    const tiff = heifExif(head);
    if (tiff) raw = parseTIFF(tiff);
  } else if ((head[0] === 0xff && head[1] === 0x0a) || ascii(head, 0, 12) === '\0\0\0\x0cJXL \r\n\x87\n') {
    s.format = 'jxl';
  }
  const e = raw as Exif | null;
  if (e) {
    s.raw = e;
    s.orientation = e.orientation;
    software = e.software;
    const x: PhotoExif = {};
    if (e.time) x.time = e.subsec ? `${e.time}.${(e.subsec + '000').slice(0, 3)}` : e.time;
    if (e.make) x.make = e.make;
    if (e.model) x.model = e.model;
    if (e.software) x.software = e.software;
    if (Object.keys(x).length) s.exif = x;
  }
  if (generatorSoftware.test(software) || c2pa) s.generated = true;
  return s;
}

/** Parses a TIFF-structured EXIF block (both byte orders), or null when it is not one. */
export function parseTIFF(b: Uint8Array): Exif | null {
  return readTIFF(b)?.exif ?? null;
}

function readTIFF(b: Uint8Array): { exif: Exif; w: number; h: number } | null {
  const e: Exif = { orientation: 1, time: '', subsec: '', make: '', model: '', software: '' };
  if (b.length < 8) return null;
  let le: boolean;
  if (b[0] === 0x49 && b[1] === 0x49 && b[2] === 42 && b[3] === 0) le = true;
  else if (b[0] === 0x4d && b[1] === 0x4d && b[2] === 0 && b[3] === 42) le = false;
  else return null;
  const u16 = (p: number) => (le ? b[p] | (b[p + 1] << 8) : (b[p] << 8) | b[p + 1]);
  const u32 = (p: number) => (le ? (b[p] | (b[p + 1] << 8) | (b[p + 2] << 16)) + b[p + 3] * 2 ** 24 : u32be(b, p));
  let w = 0, h = 0;
  const walk = (off: number, fn: (tag: number, typ: number, count: number, val: number) => void) => {
    if (off < 8 || off + 2 > b.length) return;
    const n = Math.min(u16(off), maxIFDEntries);
    let p = off + 2;
    for (let i = 0; i < n; i++) {
      if (p + 12 > b.length) return;
      fn(u16(p), u16(p + 2), u32(p + 4), p + 8);
      p += 12;
    }
  };
  const uint = (typ: number, count: number, val: number) => {
    if (count !== 1) return 0;
    if (typ === typeShort) return u16(val);
    if (typ === typeLong) {
      const v = u32(val);
      return v > 2 ** 30 ? 0 : v;
    }
    return 0;
  };
  const text = (typ: number, count: number, val: number) => {
    if (typ !== typeASCII || count === 0) return '';
    let s: Uint8Array;
    if (count <= 4) s = b.subarray(val, val + count);
    else {
      const off = u32(val);
      if (off + count > b.length) return '';
      s = b.subarray(off, off + count);
    }
    const nul = s.indexOf(0);
    if (nul >= 0) s = s.subarray(0, nul);
    if (s.length > maxStringLen) s = s.subarray(0, maxStringLen);
    return new TextDecoder('utf-8', { fatal: false }).decode(s).replace(/�/g, '').trim();
  };
  const ifd0 = u32(4);
  let exifIFD = -1;
  walk(ifd0, (t, typ, count, val) => {
    switch (t) {
      case tag.orientation: {
        const o = uint(typ, count, val);
        if (o >= 1 && o <= 8) e.orientation = o;
        break;
      }
      case tag.make:
        e.make = text(typ, count, val);
        break;
      case tag.model:
        e.model = text(typ, count, val);
        break;
      case tag.software:
        e.software = text(typ, count, val);
        break;
      case tag.exifIFD:
        if (typ === typeLong && count === 1) exifIFD = u32(val);
        break;
      case tag.width:
        w = uint(typ, count, val);
        break;
      case tag.length:
        h = uint(typ, count, val);
        break;
    }
  });
  if (exifIFD > 0 && exifIFD !== ifd0) {
    walk(exifIFD, (t, typ, count, val) => {
      if (t === tag.dateTimeOriginal) e.time = exifToISO(text(typ, count, val));
      else if (t === tag.subSecOriginal) e.subsec = /^\d*/.exec(text(typ, count, val))![0];
    });
  }
  return { exif: e, w, h };
}

/** "2006:01:02 15:04:05" to "2006-01-02T15:04:05"; "" for anything that is not a valid date. */
function exifToISO(s: string): string {
  const m = /^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})/.exec(s);
  if (!m) return '';
  const [y, mo, d, h, mi, se] = m.slice(1).map(Number);
  const days = new Date(Date.UTC(y, mo, 0)).getUTCDate();
  if (y < 1 || mo < 1 || mo > 12 || d < 1 || d > days || h > 23 || mi > 59 || se > 59) return '';
  return `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}`;
}

// IJG (Annex K) luminance table in zigzag order, the order DQT stores tables in.
const stdLuma = [
  16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40, 26, 24, 22, 22, 24, 49, 35, 37, 29, 40, 58, 51, 61, 60, 57, 51,
  56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56, 80, 109, 81, 87, 95, 98, 103, 104, 103, 62, 77, 113, 121, 112, 100, 120, 92, 101, 103, 99,
];

/** IJG quality (1..100) from a DQT payload's luma table, or 0 (imgmeta.JPEGQuality). */
export function jpegQuality(dqt: Uint8Array): number {
  let p = 0;
  while (p < dqt.length) {
    const pq = dqt[p] >> 4, tq = dqt[p] & 0x0f;
    p++;
    if (pq > 1) return 0;
    const size = pq === 1 ? 128 : 64;
    if (p + size > dqt.length) return 0;
    if (tq !== 0) {
      p += size;
      continue;
    }
    let sum = 0;
    for (let i = 0; i < 64; i++) {
      const v = pq === 1 ? (dqt[p + 2 * i] << 8) | dqt[p + 2 * i + 1] : dqt[p + i];
      if (v === 0) return 0; // not a valid quantiser
      sum += (100 * v) / stdLuma[i];
    }
    const s = sum / 64;
    const q = s <= 100 ? (200 - s) / 2 : 5000 / s;
    return Math.max(1, Math.min(100, roundHalfAway(q)));
  }
  return 0;
}

/**
 * Returns jpeg with a small Exif APP1 (little-endian TIFF: IFD0 Make and
 * Model, Exif IFD DateTimeOriginal and SubSecTimeOriginal; never Orientation,
 * Software or GPS) in place of any Exif it had, after SOI or a leading APP0.
 * time may carry a ".123" fraction, used when subsec is empty. With nothing
 * to write, or a JPEG it cannot walk, it returns jpeg unchanged.
 */
export function injectExif(jpeg: Uint8Array, e: PhotoExif & { subsec?: string }): Uint8Array {
  const app1 = buildAPP1(e);
  if (!app1 || jpeg.length < 4 || jpeg[0] !== 0xff || jpeg[1] !== 0xd8) return jpeg;
  const segs: { start: number; end: number; marker: number }[] = [];
  let sos = -1;
  let p = 2;
  while (p + 1 < jpeg.length && sos < 0) {
    if (jpeg[p] !== 0xff) return jpeg;
    const m = jpeg[p + 1];
    if (m === 0xff) p++;
    else if (m === 0xda) sos = p;
    else if (m === 0xd9) return jpeg;
    else if (m === 0x01 || (m >= 0xd0 && m <= 0xd7)) {
      segs.push({ start: p, end: p + 2, marker: m });
      p += 2;
    } else {
      if (p + 4 > jpeg.length) return jpeg;
      const end = p + 2 + ((jpeg[p + 2] << 8) | jpeg[p + 3]);
      if (end < p + 4 || end > jpeg.length) return jpeg;
      segs.push({ start: p, end, marker: m });
      p = end;
    }
  }
  if (sos < 0) return jpeg;
  const parts: Uint8Array[] = [new Uint8Array([0xff, 0xd8])];
  let placed = false;
  segs.forEach((s, i) => {
    if (!placed && !(i === 0 && s.marker === 0xe0)) {
      parts.push(app1);
      placed = true;
    }
    if (s.marker === 0xe1 && startsWith(jpeg.subarray(Math.min(s.start + 4, s.end), s.end), exifPrefix)) return;
    parts.push(jpeg.subarray(s.start, s.end));
  });
  if (!placed) parts.push(app1);
  parts.push(jpeg.subarray(sos));
  return concat(parts);
}

/** The APP1 segment for injectExif, or null when e has no make, model or valid time. */
function buildAPP1(e: PhotoExif & { subsec?: string }): Uint8Array | null {
  const time = exifTime(e.time ?? '');
  let subsec = /^\d*/.exec(e.subsec ?? '')![0];
  if (time && !subsec) {
    const i = (e.time ?? '').indexOf('.');
    if (i >= 0) subsec = /^\d*/.exec(e.time!.slice(i + 1))![0];
  }
  type Entry = { tag: number; type: number; count: number; data?: Uint8Array; off?: number };
  const asciiEntry = (t: number, s: string): Entry => {
    const enc = new TextEncoder().encode(s).subarray(0, maxStringLen);
    const d = new Uint8Array(enc.length + 1);
    d.set(enc);
    return { tag: t, type: typeASCII, count: d.length, data: d };
  };
  const ifd0: Entry[] = [];
  const sub: Entry[] = [];
  if (e.make) ifd0.push(asciiEntry(tag.make, e.make));
  if (e.model) ifd0.push(asciiEntry(tag.model, e.model));
  if (time) sub.push(asciiEntry(tag.dateTimeOriginal, time));
  if (time && subsec) sub.push(asciiEntry(tag.subSecOriginal, subsec));
  if (sub.length) ifd0.push({ tag: tag.exifIFD, type: typeLong, count: 1 });
  if (!ifd0.length) return null;

  const ifdLen = (n: number) => 2 + 12 * n + 4;
  const subOff = 8 + ifdLen(ifd0.length);
  const dataOff = subOff + (sub.length ? ifdLen(sub.length) : 0);
  const data: number[] = [];
  for (const x of [...ifd0, ...sub]) {
    if (x.data && x.data.length > 4) {
      x.off = dataOff + data.length;
      data.push(...x.data);
      if (data.length & 1) data.push(0); // values start on a word boundary
    }
  }
  const out: number[] = [0x49, 0x49, 42, 0, 8, 0, 0, 0];
  const u16 = (v: number) => out.push(v & 0xff, (v >> 8) & 0xff);
  const u32 = (v: number) => out.push(v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >>> 24) & 0xff);
  const writeIFD = (es: Entry[]) => {
    u16(es.length);
    for (const x of es) {
      u16(x.tag);
      u16(x.type);
      u32(x.count);
      if (x.tag === tag.exifIFD) u32(subOff);
      else if (x.data && x.data.length <= 4) for (let i = 0; i < 4; i++) out.push(x.data[i] ?? 0);
      else u32(x.off ?? 0);
    }
    u32(0); // no next IFD
  };
  writeIFD(ifd0);
  if (sub.length) writeIFD(sub);
  out.push(...data);
  const n = 2 + exifPrefix.length + out.length;
  if (n > 0xffff) return null;
  return new Uint8Array([0xff, 0xe1, n >> 8, n & 0xff, ...exifPrefix, ...out]);
}

/** Exif time (ISO or EXIF layout, any fraction ignored) to "2006:01:02 15:04:05", or "". */
function exifTime(s: string): string {
  const m = /^(\d{4})[-:](\d{2})[-:](\d{2})[T ](\d{2}):(\d{2}):(\d{2})/.exec(s);
  if (!m) return '';
  return exifToISO(`${m[1]}:${m[2]}:${m[3]} ${m[4]}:${m[5]}:${m[6]}`) ? `${m[1]}:${m[2]}:${m[3]} ${m[4]}:${m[5]}:${m[6]}` : '';
}

const exifPrefix = new Uint8Array([0x45, 0x78, 0x69, 0x66, 0, 0]); // "Exif\0\0"
const pngSig = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** Each marker segment's payload up to the first SOS or EOI; a cut-off segment is passed with what is there. */
function walkJPEG(head: Uint8Array, fn: (marker: number, seg: Uint8Array) => boolean) {
  if (head.length < 4 || head[0] !== 0xff || head[1] !== 0xd8) return;
  let p = 2;
  while (p + 1 < head.length) {
    if (head[p] !== 0xff) return; // lost sync
    const m = head[p + 1];
    if (m === 0xff) {
      p++; // fill byte
      continue;
    }
    p += 2;
    if (m === 0xd9 || m === 0xda) return;
    if (m === 0x01 || (m >= 0xd0 && m <= 0xd7)) continue;
    if (p + 2 > head.length) return;
    const n = (head[p] << 8) | head[p + 1];
    if (n < 2) return;
    if (!fn(m, head.subarray(p + 2, Math.min(p + n, head.length)))) return;
    p += n;
  }
}

function walkPNG(head: Uint8Array, fn: (type: string, data: Uint8Array) => boolean) {
  if (!startsWith(head, pngSig)) return;
  let p = pngSig.length;
  while (p + 8 <= head.length) {
    const n = u32be(head, p);
    const type = ascii(head, p + 4, 4);
    const start = p + 8;
    if (!fn(type, head.subarray(start, Math.min(start + n, head.length))) || type === 'IEND') return;
    const next = start + n + 4; // data + CRC
    if (next > head.length) return;
    p = next;
  }
}

function walkWebP(head: Uint8Array, fn: (fourcc: string, data: Uint8Array) => boolean) {
  if (head.length < 12 || ascii(head, 0, 4) !== 'RIFF' || ascii(head, 8, 4) !== 'WEBP') return;
  let p = 12;
  while (p + 8 <= head.length) {
    const fourcc = ascii(head, p, 4);
    const n = (head[p + 4] | (head[p + 5] << 8) | (head[p + 6] << 16)) + head[p + 7] * 2 ** 24;
    const start = p + 8;
    if (!fn(fourcc, head.subarray(start, Math.min(start + n, head.length)))) return;
    const next = start + n + (n & 1); // chunks are padded to an even size
    if (next > head.length) return;
    p = next;
  }
}

/** iTXt text after the keyword's NUL, or null when compressed or short. */
function itxtText(b: Uint8Array): Uint8Array | null {
  if (b.length < 2 || b[0] !== 0) return null;
  b = b.subarray(2);
  for (let i = 0; i < 2; i++) {
    const nul = b.indexOf(0);
    if (nul < 0) return null;
    b = b.subarray(nul + 1);
  }
  return b;
}

function isSOF(m: number): boolean {
  return m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc;
}

/** A C2PA manifest that names the trainedAlgorithmicMedia source type (camera phones sign real photos too). */
function c2paGenerated(b: Uint8Array): boolean {
  if (!b.length) return false;
  const s = latin1(b);
  return s.includes('c2pa') && s.includes('trainedAlgorithmicMedia');
}

function startsWith(b: Uint8Array, prefix: Uint8Array): boolean {
  if (b.length < prefix.length) return false;
  for (let i = 0; i < prefix.length; i++) if (b[i] !== prefix[i]) return false;
  return true;
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let p = 0;
  for (const part of parts) {
    out.set(part, p);
    p += part.length;
  }
  return out;
}

function u32be(b: Uint8Array, p: number): number {
  if (p + 4 > b.length) return 0;
  return b[p] * 2 ** 24 + ((b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]);
}

function ascii(b: Uint8Array, p: number, n: number): string {
  let s = '';
  for (let i = p; i < p + n && i < b.length; i++) s += String.fromCharCode(b[i]);
  return s;
}

function latin1(b: Uint8Array): string {
  let s = '';
  for (let i = 0; i < b.length; i += 8192) s += String.fromCharCode(...b.subarray(i, i + 8192));
  return s;
}

/** Go's math.Round for positive and negative values (half away from zero). */
function roundHalfAway(v: number): number {
  return v < 0 ? -Math.round(-v) : Math.round(v);
}

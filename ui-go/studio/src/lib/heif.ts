// EXIF out of a HEIF container (HEIC, HEIF, AVIF), the same reader as
// internal/imgmeta/heif.go. The browser's HEIC to JPEG conversion goes
// through a canvas and drops EXIF; this finds the Exif item so the JPEG can
// carry the capture time and camera again (injectExif in exif.ts).
//
// The bytes are untrusted: every read is bounds-checked and nothing throws.

const maxItems = 4096;
const maxExifSize = 4 << 20;

/** The image format a ftyp box at the start of head declares: heic, avif, heif, or ''. */
export function heifFormat(head: Uint8Array): '' | 'heic' | 'heif' | 'avif' {
  if (head.length < 16 || str(head, 4, 4) !== 'ftyp') return '';
  let n = u32(head, 0);
  if (n < 16 || n > head.length) n = Math.min(head.length, 256);
  let heic = false, avif = false, heif = false;
  const check = (b: string) => {
    if (['heic', 'heix', 'heim', 'heis', 'hevc', 'hevx', 'hevm', 'hevs'].includes(b)) heic = true;
    else if (b === 'avif' || b === 'avis') avif = true;
    else if (b === 'mif1' || b === 'msf1' || b === 'mif2') heif = true;
  };
  const major = str(head, 8, 4);
  check(major);
  for (let p = 16; p + 4 <= n; p += 4) check(str(head, p, 4));
  // The major brand decides when it names a codec; an AVIF often lists
  // mif1 and a HEIC lists mif1 too.
  if (major === 'avif' || major === 'avis') return 'avif';
  if (heic) return 'heic';
  if (avif) return 'avif';
  if (heif) return 'heif';
  return '';
}

/** Item of meta/iinf with its location from meta/iloc and its ispe size. */
interface Item {
  type: string;
  method: number; // iloc construction_method: 0 file offset, 1 idat offset
  base: number;
  extents: [number, number][];
  located: boolean;
  w: number;
  h: number;
}

interface Meta {
  primary: number;
  items: Map<number, Item>;
  idat: Uint8Array | null;
}

/**
 * The TIFF bytes of a HEIF file's Exif item (starting "II" or "MM"), or null:
 * ftyp, then meta/iinf (infe v2/v3 of type 'Exif'), then meta/iloc (v0-2)
 * extents from the file or the meta box's idat, then the 4-byte TIFF header
 * offset (and any "Exif\0\0") skipped.
 */
export function heifExif(bytes: Uint8Array): Uint8Array | null {
  const m = readMeta(bytes);
  return m ? exifOf(m, bytes) : null;
}

/** The primary item's stored size (ispe), or zeros. */
export function heifSize(bytes: Uint8Array): { width: number; height: number } {
  const m = readMeta(bytes);
  const it = m?.items.get(m.primary);
  return { width: it?.w ?? 0, height: it?.h ?? 0 };
}

function readMeta(b: Uint8Array): Meta | null {
  if (!heifFormat(b)) return null;
  let off = 0;
  for (let i = 0; i < 64 && off + 8 <= b.length; i++) {
    let size = u32(b, off);
    const type = str(b, off + 4, 4);
    let hdr = 8;
    if (size === 0) size = b.length - off;
    else if (size === 1) {
      if (off + 16 > b.length) return null;
      size = u64(b, off + 8);
      hdr = 16;
    }
    if (size < hdr || size > b.length - off) return null;
    if (type === 'meta') return parseMeta(b.subarray(off + hdr, off + size));
    off += size;
  }
  return null;
}

function parseMeta(meta: Uint8Array): Meta | null {
  if (meta.length < 4) return null;
  const m: Meta = { primary: 0, items: new Map(), idat: null };
  const props: Uint8Array[] = []; // ipco children with their type, 1-based in ipma
  let ipma: Uint8Array | null = null;
  eachBox(meta.subarray(4), (type, body) => {
    switch (type) {
      case 'pitm': {
        const c = new Cursor(body);
        const v = c.u8();
        c.skip(3);
        m.primary = v === 0 ? c.u16() : c.u32();
        break;
      }
      case 'iinf':
        parseIINF(m, body);
        break;
      case 'iloc':
        parseILOC(m, body);
        break;
      case 'idat':
        m.idat = body;
        break;
      case 'iprp':
        eachBox(body, (t, b) => {
          if (t === 'ipco') {
            eachBox(b, (pt, pb) => {
              const p = new Uint8Array(4 + pb.length);
              for (let i = 0; i < 4; i++) p[i] = pt.charCodeAt(i);
              p.set(pb, 4);
              props.push(p);
              return props.length < maxItems;
            });
          } else if (t === 'ipma') ipma = b;
          return true;
        });
        break;
    }
    return true;
  });
  if (ipma) parseIPMA(m, ipma, props);
  return m;
}

function item(m: Meta, id: number): Item {
  let it = m.items.get(id);
  if (!it) {
    it = { type: '', method: 0, base: 0, extents: [], located: false, w: 0, h: 0 };
    if (m.items.size < maxItems) m.items.set(id, it); // over the cap: a throwaway
  }
  return it;
}

function parseIINF(m: Meta, body: Uint8Array) {
  const c = new Cursor(body);
  const v = c.u8();
  c.skip(3);
  if (v === 0) c.u16();
  else c.u32();
  if (c.bad) return;
  eachBox(c.rest(), (type, b) => {
    if (type !== 'infe') return true;
    const e = new Cursor(b);
    const ver = e.u8();
    e.skip(3);
    if (ver < 2) return true; // v0/v1 entries carry no item type
    const id = ver === 2 ? e.u16() : e.u32();
    e.u16(); // item_protection_index
    const t = e.n(4);
    if (!e.bad) item(m, id).type = str(t, 0, 4);
    return m.items.size < maxItems;
  });
}

function parseILOC(m: Meta, body: Uint8Array) {
  const c = new Cursor(body);
  const ver = c.u8();
  c.skip(3);
  if (ver > 2) return;
  const a = c.u8(), b = c.u8();
  const offSize = a >> 4, lenSize = a & 0x0f, baseSize = b >> 4;
  const idxSize = ver === 1 || ver === 2 ? b & 0x0f : 0;
  const count = ver < 2 ? c.u16() : c.u32();
  for (let i = 0; i < count && !c.bad && i < maxItems; i++) {
    const id = ver < 2 ? c.u16() : c.u32();
    const method = ver === 1 || ver === 2 ? c.u16() & 0x0f : 0;
    c.u16(); // data_reference_index
    const base = c.uintN(baseSize);
    let n = c.u16();
    // Bound the extent loop by the bytes left: with every size 0 an extent
    // reads nothing, so keep at most one (it carries no data).
    const per = idxSize + offSize + lenSize;
    if (per === 0) n = Math.min(n, 1);
    else if (n * per > c.rest().length) return;
    const ext: [number, number][] = [];
    for (let j = 0; j < n && !c.bad; j++) {
      c.uintN(idxSize);
      const off = c.uintN(offSize);
      const l = c.uintN(lenSize);
      if (ext.length < 64) ext.push([off, l]);
    }
    if (c.bad) return;
    const it = item(m, id);
    it.method = method;
    it.base = base;
    it.extents = ext;
    it.located = true;
  }
}

function parseIPMA(m: Meta, body: Uint8Array, props: Uint8Array[]) {
  const c = new Cursor(body);
  const ver = c.u8();
  const flags = (c.u8() << 16) | c.u16();
  const count = c.u32();
  for (let i = 0; i < count && !c.bad && i < maxItems; i++) {
    const id = ver < 1 ? c.u16() : c.u32();
    const n = c.u8();
    for (let j = 0; j < n && !c.bad; j++) {
      const idx = flags & 1 ? c.u16() & 0x7fff : c.u8() & 0x7f;
      if (c.bad || idx < 1 || idx > props.length) continue;
      const p = props[idx - 1];
      if (str(p, 0, 4) !== 'ispe' || p.length < 16) continue;
      // ispe: type, version+flags, then width and height.
      const it = item(m, id);
      it.w = u32(p, 8) & 0x3fffffff;
      it.h = u32(p, 12) & 0x3fffffff;
    }
  }
}

function exifOf(m: Meta, file: Uint8Array): Uint8Array | null {
  let it: Item | null = null;
  let best = 0;
  for (const [id, x] of m.items) {
    if (x.type === 'Exif' && x.located && x.extents.length > 0 && (it === null || id < best)) {
      it = x;
      best = id;
    }
  }
  if (!it) return null;
  const parts: Uint8Array[] = [];
  let total = 0;
  for (const [eo, el] of it.extents) {
    const src = it.method === 0 ? file : it.method === 1 ? m.idat : null;
    if (!src) return null;
    const off = it.base + eo;
    let n = el;
    if (n === 0 && it.extents.length === 1) n = src.length - off; // 0 = to the end
    if (off < 0 || n < 0 || off > src.length || n > src.length - off || total + n > maxExifSize) return null;
    parts.push(src.subarray(off, off + n));
    total += n;
  }
  const data = new Uint8Array(total);
  let p = 0;
  for (const part of parts) {
    data.set(part, p);
    p += part.length;
  }
  // The item starts with exif_tiff_header_offset (big-endian u32), the
  // number of bytes before the TIFF header.
  if (data.length < 4) return null;
  const skip = u32(data, 0);
  if (skip > data.length - 4) return null;
  let tiff = data.subarray(4 + skip);
  if (str(tiff, 0, 6) === 'Exif\0\0') tiff = tiff.subarray(6);
  const sig = str(tiff, 0, 4);
  if (tiff.length < 8 || (sig !== 'II*\0' && sig !== 'MM\0*')) return null;
  return tiff;
}

/** Calls fn with the type and body of each box in b; stops at the first malformed header. */
function eachBox(b: Uint8Array, fn: (type: string, body: Uint8Array) => boolean) {
  let p = 0;
  while (b.length - p >= 8) {
    let n = u32(b, p);
    const type = str(b, p + 4, 4);
    let hdr = 8;
    if (n === 0) n = b.length - p;
    else if (n === 1) {
      if (b.length - p < 16) return;
      n = u64(b, p + 8);
      hdr = 16;
    }
    if (n < hdr || n > b.length - p) return;
    if (!fn(type, b.subarray(p + hdr, p + n))) return;
    p += n;
  }
}

/** Reads big-endian fields; once a read runs past the end it sets bad and returns zeros. */
class Cursor {
  p = 0;
  bad = false;
  constructor(private b: Uint8Array) {}
  n(k: number): Uint8Array {
    if (this.bad || k < 0 || this.p + k > this.b.length) {
      this.bad = true;
      return new Uint8Array(Math.max(k, 0));
    }
    const s = this.b.subarray(this.p, this.p + k);
    this.p += k;
    return s;
  }
  skip(k: number) {
    this.n(k);
  }
  u8(): number {
    return this.n(1)[0];
  }
  u16(): number {
    const s = this.n(2);
    return (s[0] << 8) | s[1];
  }
  u32(): number {
    return u32(this.n(4), 0);
  }
  rest(): Uint8Array {
    return this.b.subarray(Math.min(this.p, this.b.length));
  }
  /** A 0, 4 or 8 byte unsigned field (iloc sizes); other sizes mark the cursor bad. */
  uintN(k: number): number {
    if (k === 0) return 0;
    if (k === 4) return this.u32();
    if (k === 8) {
      const v = u64(this.n(8), 0);
      if (!Number.isSafeInteger(v)) this.bad = true;
      return this.bad ? 0 : v;
    }
    this.bad = true;
    return 0;
  }
}

function u32(b: Uint8Array, p: number): number {
  if (p + 4 > b.length) return 0;
  return ((b[p] << 24) >>> 0) + ((b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]);
}

function u64(b: Uint8Array, p: number): number {
  return u32(b, p) * 2 ** 32 + u32(b, p + 4);
}

function str(b: Uint8Array, p: number, n: number): string {
  let s = '';
  for (let i = p; i < p + n && i < b.length; i++) s += String.fromCharCode(b[i]);
  return s;
}

// What the trainer reads (toolkit/data_loader.py), and turning phone photos it
// cannot read (HEIC, AVIF, ...) into JPEGs in the browser before they upload.

export const trainable = /\.(jpe?g|png|webp)$/i;
const convertible = /\.(avif|heic|heif|bmp|gif|tiff?)$/i;

/** JPEG copy of an image the browser can decode (HEIC: Safari only), EXIF rotation applied. */
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
  return new File([blob], f.name.replace(/\.[^.]+$/, '') + '.jpg', { type: 'image/jpeg', lastModified: f.lastModified });
}

/** Splits picked files into what can upload (converted where needed) and what cannot. */
export async function prepareImages(files: File[]): Promise<{ ok: File[]; skipped: string[] }> {
  const ok: File[] = [];
  const skipped: string[] = [];
  for (const f of files) {
    if (trainable.test(f.name) || /\.txt$/i.test(f.name)) ok.push(f);
    else if (convertible.test(f.name)) {
      try {
        ok.push(await toJpeg(f));
      } catch {
        skipped.push(f.name);
      }
    } else skipped.push(f.name);
  }
  return { ok, skipped };
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

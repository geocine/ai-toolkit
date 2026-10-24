// Photo checks from the studio: measuring photos in the browser before they
// upload (a pool of photos.worker.ts workers), and the /v1 photo routes that
// judge them (docs/headless.md, PHOTO-PLAN section 11). The browser only
// measures; every rule and threshold lives in the server's judge.

import { ApiError, call, del, get, post, type Job, type Server } from './api';
import type { PhotoAssess, PhotoFix, PhotoFixOp, PhotoKind, PhotoReview, PhotoSettingsPatch } from './photoTypes';
import { measure, type WorkIn, type WorkOut } from './photos.worker';

export type { WorkIn, WorkOut } from './photos.worker';
/** The section 3 box filter and dHash, exported for studio/scripts/check-dhash.mjs. */
export { boxDown, dhash, thumb32 } from './photos.worker';

/** Workers to measure with: one core is left for the page. */
function poolSize(): number {
  return Math.min(4, Math.max(1, (globalThis.navigator?.hardwareConcurrency ?? 2) - 1));
}

/**
 * Workers can measure only where they have a 2D OffscreenCanvas (Safari
 * before 16.4 has none); the main thread then measures, a few at a time.
 */
function workersMeasure(): boolean {
  try {
    return typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined' && new OffscreenCanvas(1, 1).getContext('2d') !== null;
  } catch {
    return false;
  }
}

/**
 * Measures each file (section 3 metrics, plus the silent fixes: upright,
 * 16-bit grey to 8-bit, transparency flattened on white). Results are in the
 * order of files; onProgress counts finished files, and onResult gets each
 * file's result as it finishes (in finishing order; r.id is its index).
 */
export async function measureFiles(
  files: File[],
  onProgress: (done: number, total: number) => void,
  onResult?: (r: WorkOut) => void,
): Promise<WorkOut[]> {
  const out: WorkOut[] = new Array(files.length);
  const sha = !!globalThis.crypto?.subtle;
  let done = 0;
  const finish = (r: WorkOut) => {
    out[r.id] = r;
    onResult?.(r);
    onProgress(++done, files.length);
  };
  const onMain = async (id: number) => {
    try {
      finish({ id, ...(await measure(files[id], { sha, fix: true })) });
    } catch (e) {
      finish({ id, error: e instanceof Error ? e.message : String(e) });
    }
  };
  let next = 0;
  const take = () => (next < files.length ? next++ : -1);

  if (files.length && workersMeasure()) {
    const lane = async () => {
      let w: Worker | null = null;
      try {
        w = new Worker(new URL('./photos.worker.ts', import.meta.url), { type: 'module' });
      } catch {
        w = null;
      }
      try {
        for (let id = take(); id >= 0; id = take()) {
          if (!w) {
            await onMain(id);
            continue;
          }
          const r = await new Promise<WorkOut | null>(resolve => {
            const ww = w!;
            const msg = (e: MessageEvent<WorkOut>) => e.data.id === id && settle(e.data);
            const err = () => settle(null);
            const settle = (v: WorkOut | null) => {
              ww.removeEventListener('message', msg);
              ww.removeEventListener('error', err);
              resolve(v);
            };
            ww.addEventListener('message', msg);
            ww.addEventListener('error', err);
            ww.postMessage({ id, file: files[id], sha } satisfies WorkIn);
          });
          if (r) finish(r);
          else {
            // The worker failed to load or died: the rest of this lane runs here.
            w.terminate();
            w = null;
            await onMain(id);
          }
        }
      } finally {
        w?.terminate();
      }
    };
    await Promise.all(Array.from({ length: Math.min(poolSize(), files.length) }, lane));
    return out;
  }

  // Main thread: chunks of 4, yielding between them so the page stays responsive.
  for (let id = take(); id >= 0; id = take()) {
    const chunk = [id];
    for (let k = 1; k < 4; k++) {
      const more = take();
      if (more < 0) break;
      chunk.push(more);
    }
    await Promise.all(chunk.map(onMain));
    await new Promise(r => setTimeout(r, 0));
  }
  return out;
}

const photosPath = (dataset: string, rest: string) => `/v1/datasets/${encodeURIComponent(dataset)}/photos/${rest}`;

/** Judges metrics measured here (or anywhere) with the server's rules; items: [] returns the formats it converts. */
export function assess(server: Server, body: PhotoAssess): Promise<PhotoReview> {
  return post<PhotoReview>(server, '/v1/photos/assess', body);
}

/**
 * What a failed live check (assess) shows: nothing on a 404, which is an older
 * server with no photo checks (uploading still works there), else the error.
 */
export function assessFailure(e: unknown): unknown {
  return e instanceof ApiError && e.status === 404 ? null : e;
}

/** The stored photos' review. pending > 0 means analysis is still running: call again. */
export function checkPhotos(
  server: Server,
  dataset: string,
  q: { kind?: PhotoKind; resolution?: number[]; pick?: number; wait_ms?: number },
): Promise<PhotoReview> {
  const p = new URLSearchParams();
  if (q.kind) p.set('kind', q.kind);
  if (q.resolution?.length) p.set('resolution', q.resolution.join(','));
  if (q.pick) p.set('pick', String(q.pick));
  if (q.wait_ms !== undefined) p.set('wait_ms', String(q.wait_ms));
  const qs = p.toString();
  return get<PhotoReview>(server, photosPath(dataset, 'check') + (qs ? `?${qs}` : ''));
}

/** Previews the fix ops (or applies them with apply: true) and returns the new review. */
export function fixPhotos(
  server: Server,
  dataset: string,
  body: { ops: PhotoFixOp[]; files?: string[]; apply?: boolean; resolution?: number[] },
): Promise<PhotoFix> {
  return post<PhotoFix>(server, photosPath(dataset, 'fix'), body);
}

/** Kind, the reference photo and per-photo decisions (null clears). */
export function photoSettings(server: Server, dataset: string, body: PhotoSettingsPatch): Promise<PhotoReview> {
  return call<PhotoReview>(server, 'PUT', photosPath(dataset, 'settings'), body);
}

/** Forgets the dataset's photo analysis; the photos are untouched. */
export function forgetPhotos(server: Server, dataset: string): Promise<void> {
  return del<void>(server, photosPath(dataset, 'analysis'));
}

/** Starts the face check, a photo_faces job on the server's CPU queue. */
export function checkFaces(server: Server, dataset: string, body?: { text?: boolean }): Promise<Job & { warnings: string[] }> {
  return post<Job & { warnings: string[] }>(server, photosPath(dataset, 'faces'), body ?? {});
}

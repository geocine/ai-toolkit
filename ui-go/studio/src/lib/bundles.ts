// Run bundles in the browser: chunked, resumable uploads of a bundle zip,
// reading its manifest locally, and following import/push tasks live.
// docs/headless.md (Run bundles) and docs/bundle-format.md.
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import {
  ApiError,
  get,
  post,
  putBytes,
  streamEvents,
  type BundleSummary,
  type BundleTask,
  type BundleUpload,
  type Items,
  type Server,
} from './api';

// ---- the manifest, read from the zip's first entry ----

export interface BundleManifest {
  format: number;
  name: string;
  step?: number;
  total_steps?: number | null;
  datasets?: { name: string; dir: string; images?: number }[];
  checkpoints?: string[];
  optimizer?: boolean;
  samples?: number;
  caches?: boolean;
  base_model?: string;
  arch?: string;
}

/**
 * Writers put aitk_bundle.json first and stored (docs/bundle-format.md), so
 * its bytes sit right after the first local header. Hand-made zips have no
 * manifest; null then.
 */
async function readManifest(file: File): Promise<BundleManifest | null> {
  try {
    const head = new DataView(await file.slice(0, 30).arrayBuffer());
    if (head.byteLength < 30 || head.getUint32(0, true) !== 0x04034b50) return null;
    const method = head.getUint16(8, true);
    let size = head.getUint32(18, true);
    const nameLen = head.getUint16(26, true);
    const extraLen = head.getUint16(28, true);
    const name = await file.slice(30, 30 + nameLen).text();
    if (method !== 0 || !/(^|\/)aitk_bundle\.json$/.test(name)) return null;
    const start = 30 + nameLen + extraLen;
    if (!size || size === 0xffffffff) size = Math.min(file.size - start, 4 << 20); // data descriptor or zip64
    let text = await file.slice(start, start + Math.min(size, 4 << 20)).text();
    const end = text.lastIndexOf('}');
    if (end < 0) return null;
    text = text.slice(0, end + 1);
    const m = JSON.parse(text);
    return m && typeof m.name === 'string' ? (m as BundleManifest) : null;
  } catch {
    return null;
  }
}

export type Resumable = 'resumable' | 'weights' | 'dataset';

export function resumability(s: { checkpoints?: string[] | null; optimizer?: boolean } | null | undefined): Resumable | null {
  if (!s) return null;
  if (!s.checkpoints?.length) return 'dataset';
  return s.optimizer ? 'resumable' : 'weights';
}

export const resumableLabel: Record<Resumable, string> = {
  resumable: 'Resumable',
  weights: 'Weights only',
  dataset: 'Dataset only',
};

export const resumableHint: Record<Resumable, string> = {
  resumable: 'Has checkpoints and the optimizer state: training continues exactly where it stopped.',
  weights: 'Has checkpoints but no optimizer state: training continues from the weights with a fresh optimizer.',
  dataset: 'The dataset and config only: training starts from step 0.',
};

/**
 * What the load dialog shows about a bundle, from a library summary, the
 * zip's manifest, or (hand-made zips) its file list. `name` is set only when
 * it is the bundle's real job name; `guess` is for display.
 */
export interface BundleInfo {
  name: string;
  guess: string;
  step: number | null;
  total: number | null;
  datasets: string[];
  images: number | null;
  baseModel: string;
  resumable: Resumable | null;
}

export const unknownInfo = (guess = ''): BundleInfo => ({ name: '', guess, step: null, total: null, datasets: [], images: null, baseModel: '', resumable: null });

export function infoFromSummary(s: BundleSummary | null, fallback: string): BundleInfo {
  if (!s) return unknownInfo(fallback);
  return {
    name: s.name,
    guess: s.name,
    step: s.step,
    total: s.total_steps ?? null,
    datasets: s.datasets ?? [],
    images: s.images,
    baseModel: s.base_model ?? '',
    resumable: resumability(s),
  };
}

function infoFromManifest(m: BundleManifest): BundleInfo {
  return {
    name: m.name,
    guess: m.name,
    step: m.step ?? 0,
    total: m.total_steps ?? null,
    datasets: (m.datasets ?? []).map(d => d.name),
    images: m.datasets ? m.datasets.reduce((n, d) => n + (d.images ?? 0), 0) : null,
    baseModel: m.base_model ?? '',
    resumable: resumability(m),
  };
}

/** Entry names from the zip's central directory (zip64 aware). */
async function zipNames(file: File): Promise<string[] | null> {
  const tailLen = Math.min(file.size, 65557);
  const tail = new DataView(await file.slice(file.size - tailLen).arrayBuffer());
  let eocd = -1;
  for (let i = tailLen - 22; i >= 0; i--) {
    if (tail.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) return null;
  let cdSize = tail.getUint32(eocd + 12, true);
  let cdOffset = tail.getUint32(eocd + 16, true);
  if (cdOffset === 0xffffffff || cdSize === 0xffffffff) {
    const loc = eocd - 20;
    if (loc < 0 || tail.getUint32(loc, true) !== 0x07064b50) return null;
    const recOff = Number(tail.getBigUint64(loc + 8, true));
    const rec = new DataView(await file.slice(recOff, recOff + 56).arrayBuffer());
    if (rec.getUint32(0, true) !== 0x06064b50) return null;
    cdSize = Number(rec.getBigUint64(40, true));
    cdOffset = Number(rec.getBigUint64(48, true));
  }
  if (cdSize > 64 << 20) return null;
  const cd = new DataView(await file.slice(cdOffset, cdOffset + cdSize).arrayBuffer());
  const dec = new TextDecoder();
  const names: string[] = [];
  for (let p = 0; p + 46 <= cd.byteLength && cd.getUint32(p, true) === 0x02014b50; ) {
    const n = cd.getUint16(p + 28, true);
    const x = cd.getUint16(p + 30, true);
    const c = cd.getUint16(p + 32, true);
    names.push(dec.decode(new Uint8Array(cd.buffer, cd.byteOffset + p + 46, n)));
    p += 46 + n + x + c;
  }
  return names;
}

const media = /\.(jpe?g|png|webp|bmp|gif|tiff?|mp4|webm|mov|mkv|wav|mp3|flac)$/i;

/** A hand-made zip (data/ + output/<name>/): what its file list tells. */
function infoFromNames(all: string[]): BundleInfo | null {
  let names = all.filter(n => !n.startsWith('__MACOSX/'));
  const tops = new Set(names.map(n => n.split('/')[0]));
  if (tops.size === 1) {
    const top = [...tops][0];
    if (!/^(data(_\d+)?|output)$/.test(top)) names = names.map(n => n.slice(top.length + 1)).filter(Boolean);
  }
  const hidden = (n: string) => n.split('/').some(seg => seg.startsWith('.'));
  names = names.filter(n => !hidden(n));
  const cfg = names.find(n => /^output\/[^/]+\/config\.(ya?ml|json)$/.test(n));
  const images = names.filter(n => /^data(_\d+)?\//.test(n) && media.test(n) && !/\/_(latent|t_e|clip_vision)_cache\//.test(n)).length;
  if (!cfg) return images ? { ...unknownInfo(), images, resumable: 'dataset' } : null;
  const folder = cfg.split('/')[1];
  const files = names.filter(n => n.startsWith(`output/${folder}/`)).map(n => n.slice(`output/${folder}/`.length));
  const saves = files.filter(f => !f.includes('/') && /\.(safetensors|pt|ckpt)$/.test(f) && f !== 'optimizer.pt');
  const steps = saves.map(f => /_(\d+)\.[a-z]+$/.exec(f)?.[1]).filter(Boolean).map(Number);
  return {
    ...unknownInfo(folder),
    step: steps.length ? Math.max(...steps) : saves.length ? null : 0,
    images,
    resumable: resumability({ checkpoints: saves, optimizer: files.includes('optimizer.pt') }),
  };
}

/** Reads what a local bundle zip holds, without uploading it. */
export async function readBundleInfo(file: File): Promise<BundleInfo | null> {
  const m = await readManifest(file);
  if (m) return infoFromManifest(m);
  try {
    const names = await zipNames(file);
    return names ? infoFromNames(names) : null;
  } catch {
    return null;
  }
}

export const phaseLabel = (t: BundleTask): string => {
  if (t.state === 'queued') return 'Queued';
  if (t.state === 'done') return t.kind === 'push' ? 'Saved' : 'Loaded';
  if (t.state === 'failed') return 'Failed';
  switch (t.phase) {
    case 'downloading':
      return 'Downloading';
    case 'reading':
      return 'Checking the zip';
    case 'restoring':
      return 'Restoring files';
    case 'building':
      return 'Building the zip';
    case 'uploading':
      return 'Uploading';
  }
  return 'Working';
};

// ---- chunked uploads (module state, so they survive page changes) ----

export interface Transfer {
  key: string; // server id + file name + size
  serverId: string;
  file: File;
  info: BundleInfo | null;
  upload: BundleUpload | null;
  sent: number;
  rate: number; // bytes/s, smoothed
  state: 'starting' | 'uploading' | 'paused' | 'complete' | 'failed';
  error: string | null;
}

const transfers = new Map<string, Transfer>();
const aborts = new Map<string, AbortController>();
const listeners = new Set<() => void>();
let snapshot: Transfer[] = [];

const emit = () => {
  snapshot = [...transfers.values()];
  listeners.forEach(l => l());
};

const patch = (key: string, p: Partial<Transfer>) => {
  const t = transfers.get(key);
  if (!t) return;
  transfers.set(key, { ...t, ...p });
  emit();
};

export function useTransfers(serverId: string | undefined): Transfer[] {
  const all = useSyncExternalStore(
    l => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => snapshot,
  );
  return all.filter(t => t.serverId === serverId);
}

export const transferKey = (s: Server, f: { name: string; size: number }) => `${s.id}|${f.name}|${f.size}`;

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

/**
 * Uploads (or resumes) a bundle zip in chunks. Matches an existing session by
 * file name and size, so choosing the same file after a reload continues
 * where it stopped. Resolves with the complete session.
 */
export async function startUpload(s: Server, file: File, onComplete?: (t: Transfer) => void): Promise<void> {
  const key = transferKey(s, file);
  const prev = transfers.get(key);
  if (prev && (prev.state === 'uploading' || prev.state === 'starting')) return;
  const ctl = new AbortController();
  aborts.set(key, ctl);
  transfers.set(key, {
    key,
    serverId: s.id,
    file,
    info: prev?.info ?? null,
    upload: prev?.upload ?? null,
    sent: prev?.sent ?? 0,
    rate: 0,
    state: 'starting',
    error: null,
  });
  emit();
  const reading = prev?.info
    ? Promise.resolve()
    : readBundleInfo(file).then(info => patch(key, { info }));

  try {
    const list = await get<Items<BundleUpload>>(s, '/v1/bundles/uploads');
    let up = list.items.find(u => u.filename === file.name && u.size === file.size) ?? null;
    if (!up) up = await post<BundleUpload>(s, '/v1/bundles/uploads', { size: file.size, filename: file.name });
    patch(key, { upload: up, sent: up.received, state: up.complete ? 'complete' : 'uploading' });

    const samples: { t: number; b: number }[] = [{ t: performance.now(), b: up.received }];
    let failures = 0;
    while (!up.complete) {
      const start: number = up.received;
      const end: number = Math.min(file.size, start + (up.chunk_size || 32 << 20));
      try {
        up = await putBytes<BundleUpload>(
          s,
          `/v1/bundles/uploads/${encodeURIComponent(up.id)}`,
          file.slice(start, end),
          { 'Content-Range': `bytes ${start}-${end - 1}/${file.size}` },
          loaded => {
            const now = performance.now();
            const b = start + loaded;
            samples.push({ t: now, b });
            while (samples.length > 2 && now - samples[0].t > 6000) samples.shift();
            const span = (now - samples[0].t) / 1000;
            patch(key, { sent: b, rate: span > 0.5 ? (b - samples[0].b) / span : 0 });
          },
          ctl.signal,
        );
        failures = 0;
        patch(key, { upload: up, sent: up.received });
      } catch (e) {
        if (ctl.signal.aborted) throw e;
        const status = e instanceof ApiError ? e.status : 0;
        // a gap (409) or a dropped connection: ask where the server is and go on
        if ((status === 409 || status === 0 || status >= 500) && failures < 6) {
          failures++;
          await sleep(status === 409 ? 0 : 1000 * 2 ** failures);
          up = await get<BundleUpload>(s, `/v1/bundles/uploads/${encodeURIComponent(up.id)}`);
          continue;
        }
        throw e;
      }
    }
    await reading;
    patch(key, { state: 'complete', sent: file.size, rate: 0 });
    const t = transfers.get(key);
    if (t) onComplete?.(t);
  } catch (e) {
    if (ctl.signal.aborted) patch(key, { state: 'paused', rate: 0 });
    else patch(key, { state: 'failed', rate: 0, error: e instanceof Error ? e.message : String(e) });
  } finally {
    aborts.delete(key);
  }
}

export function pauseUpload(key: string) {
  aborts.get(key)?.abort();
}

/** Forgets a transfer locally (after an import, or a discard). */
export function dropTransfer(key: string) {
  aborts.get(key)?.abort();
  transfers.delete(key);
  emit();
}

export function dropTransfersFor(uploadId: string) {
  for (const t of transfers.values()) if (t.upload?.id === uploadId) transfers.delete(t.key);
  emit();
}

// ---- import and push tasks, live ----

const TERMINAL = ['done', 'failed'];
export const isActive = (t: BundleTask) => !TERMINAL.includes(t.state);

/**
 * Tasks of one kind (optionally for one job), kept fresh by the event stream
 * with polling while it is down or while a task is active without events.
 */
export function useTasks(server: Server | null, kind: 'import' | 'push', jobId?: string) {
  const [tasks, setTasks] = useState<BundleTask[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [live, setLive] = useState(false);
  const lastEvent = useRef(0);
  const path = `/v1/bundles/${kind === 'import' ? 'imports' : 'pushes'}${jobId ? `?job_id=${encodeURIComponent(jobId)}` : ''}`;

  const reload = useCallback(async () => {
    if (!server) return;
    try {
      const r = await get<Items<BundleTask>>(server, path);
      setTasks(sortTasks(r.items));
      setError(null);
    } catch (e) {
      setError(e);
    }
  }, [server, path]);

  const upsert = useCallback((t: BundleTask) => {
    setTasks(list => sortTasks([t, ...(list ?? []).filter(x => x.id !== t.id)]));
  }, []);

  useEffect(() => {
    reload();
    if (!server) return;
    return streamEvents(
      server,
      `types=bundle.${kind}`,
      e => {
        const t = e.data as BundleTask;
        if (!t?.id || (jobId && t.job_id !== jobId)) return;
        lastEvent.current = Date.now();
        upsert(t);
      },
      on => {
        setLive(on);
        if (on) reload();
      },
    );
  }, [server, kind, jobId, reload, upsert]);

  // polling fallback: always while the stream is down, and while something is
  // active but no event came for a few seconds
  const anyActive = (tasks ?? []).some(isActive);
  useEffect(() => {
    if (live && !anyActive) return;
    const t = setInterval(() => {
      if (!live || Date.now() - lastEvent.current > 4000) reload();
    }, anyActive ? 2000 : 8000);
    return () => clearInterval(t);
  }, [live, anyActive, reload]);

  return { tasks, error, live, reload, upsert };
}

const sortTasks = (l: BundleTask[]) => [...l].sort((a, b) => b.created_at.localeCompare(a.created_at));

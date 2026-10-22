// Client for one headless ai-toolkit server (/v1, docs/headless.md).

export interface Server {
  id: string;
  name: string;
  base: string; // "https://pod-8675.proxy.runpod.net", or "/aitk" through the dev proxy
  token: string;
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

const trimBase = (b: string) => b.replace(/\/+$/, '');

export function url(s: Server, path: string): string {
  return trimBase(s.base) + path;
}

function headers(s: Server, extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  if (s.token) h.set('Authorization', `Bearer ${s.token}`);
  return h;
}

async function fail(res: Response): Promise<never> {
  let code = 'http_' + res.status;
  let message = res.statusText || `HTTP ${res.status}`;
  try {
    const body = await res.json();
    if (body?.error?.code) code = body.error.code;
    if (body?.error?.message) message = body.error.message;
    else if (typeof body?.error === 'string') message = body.error;
  } catch {
    /* not JSON */
  }
  throw new ApiError(res.status, code, message);
}

export async function call<T>(s: Server, method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url(s, path), {
      method,
      headers: headers(s, body === undefined ? undefined : { 'Content-Type': 'application/json' }),
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    throw new ApiError(0, 'network', `cannot reach ${s.base} (${e instanceof Error ? e.message : e}). ` +
      'Check the URL, and that the server allows this origin (AITK_CORS_ORIGINS).');
  }
  if (!res.ok) return fail(res);
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const get = <T,>(s: Server, path: string) => call<T>(s, 'GET', path);
export const post = <T,>(s: Server, path: string, body?: unknown) => call<T>(s, 'POST', path, body ?? {});

/** Multipart upload with progress (fetch has no upload progress). */
export function upload<T>(s: Server, path: string, files: File[], onProgress: (done: number, total: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const f of files) form.append('files[]', f, f.name);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url(s, path));
    if (s.token) xhr.setRequestHeader('Authorization', `Bearer ${s.token}`);
    xhr.upload.onprogress = e => e.lengthComputable && onProgress(e.loaded, e.total);
    xhr.onerror = () => reject(new ApiError(0, 'network', 'upload failed: network error'));
    xhr.onload = () => {
      let body: any = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* empty */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else reject(new ApiError(xhr.status, body?.error?.code ?? 'http_' + xhr.status, body?.error?.message ?? xhr.statusText));
    };
    xhr.send(form);
  });
}

/** Downloads with the bearer token (plain links cannot send it). */
export async function download(s: Server, path: string, filename: string): Promise<void> {
  const res = await fetch(url(s, path), { headers: headers(s) });
  if (!res.ok) return fail(res);
  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30_000);
}

/** Public media URL (the server serves /api/img without a token, for <img>). */
export function mediaUrl(s: Server, absPath: string, thumb = false): string {
  const idx = Math.max(absPath.lastIndexOf('/'), absPath.lastIndexOf('\\'));
  const folder = idx === 0 ? absPath[0] : absPath.slice(0, idx);
  const name = absPath.slice(idx + 1);
  return url(s, `/api/img/${encodeURIComponent(folder)}/${encodeURIComponent(name)}`) + (thumb ? '?thumb=1' : '');
}

export interface AitkEvent {
  id: number;
  type: string;
  job_id: string | null;
  at: string;
  data: any;
}

/**
 * Follows /v1/events/stream. EventSource cannot send Authorization, so this
 * reads the stream with fetch and reconnects with Last-Event-ID.
 * Returns a stop function.
 */
export function streamEvents(s: Server, query: string, onEvent: (e: AitkEvent) => void, onState?: (live: boolean) => void): () => void {
  const ctl = new AbortController();
  let last = 0;
  let afterNow = true;
  let stopped = false;
  const run = async () => {
    while (!stopped) {
      try {
        // the first connect skips history (after=now); reconnects resume from
        // the last event seen
        const h = headers(s, last ? { 'Last-Event-ID': String(last) } : undefined);
        const from = last || !afterNow ? '' : '&after=now';
        const res = await fetch(url(s, `/v1/events/stream?${query}${from}`), { headers: h, signal: ctl.signal });
        if (res.status === 400 && from) {
          afterNow = false; // older server: no after=now, take the replay
          continue;
        }
        if (!res.ok || !res.body) throw new Error(`stream ${res.status}`);
        onState?.(true);
        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
        let buf = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += value;
          let i: number;
          while ((i = buf.indexOf('\n\n')) >= 0) {
            const frame = buf.slice(0, i);
            buf = buf.slice(i + 2);
            const data = frame.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trim()).join('\n');
            if (!data) continue;
            try {
              const e = JSON.parse(data) as AitkEvent;
              if (e.id) last = e.id;
              onEvent(e);
            } catch {
              /* partial or non-JSON frame */
            }
          }
        }
      } catch {
        if (stopped) return;
      }
      onState?.(false);
      if (!stopped) await new Promise(r => setTimeout(r, 2000));
    }
  };
  run();
  return () => {
    stopped = true;
    ctl.abort();
  };
}

// ---- types (docs/headless.md) ----

export type JobState = 'draft' | 'queued' | 'starting' | 'running' | 'stopping' | 'stopped' | 'completed' | 'failed' | 'interrupted';
export interface JobConfig {
  job: string;
  config: { name: string; process: Record<string, any>[] };
  meta?: Record<string, any>;
}
export interface Checkpoint { name: string; path: string; step: number | null; size: number; modified: string; sha256?: string }
export interface Sample { name: string; path: string; step: number | null; index: number | null; modified: string }
export interface Attempt { n: number; started_at: string; ended_at: string | null; end_state: string | null; start_step: number; end_step: number | null; auto: boolean }
export interface Job {
  id: string; name: string; type: string; gpu_ids: string; state: JobState; status: string; info: string;
  step: number; total_steps: number | null; speed: string; created_at: string; updated_at: string;
  config: JobConfig; policy: { auto_resume: boolean; sync_remote: string | null; auto_upload_remote: string | null };
  attempts: number; last_attempt: Attempt | null; latest_checkpoint: Checkpoint | null;
}
export interface Dataset { name: string; path: string; images: number; captions: number; size: number; modified: string }
export interface DatasetItem { file: string; caption: string | null; size: number; width?: number; height?: number }
export interface DatasetDetail extends Dataset { items: DatasetItem[]; next_cursor: string | null }
export interface Remote { id: string; name: string; kind: 's3' | 'hf' | 'fs'; config: Record<string, any> }
export interface Upload { id: string; remote_id: string; job_id: string | null; path: string; url: string | null; state: 'queued' | 'running' | 'done' | 'failed'; bytes: number; bytes_done: number; error: string | null }
export interface SyncStatus { remote: string | null; last_sync_at: string | null; last_error: string | null; step?: number | null; lease: { node_label: string; expires_at: string } | null }
export interface Settings { training_folder: string; datasets_folder: string }
export interface Health { ok: boolean; version: string; upstream_version: string }
export interface SystemInfo { gpus: { index: number; name: string; memory?: { total: number; used: number } }[]; toolkit_root: string }
export interface LossSeries { keys: string[]; key: string; points: { step: number; value: number }[] }
export interface Items<T> { items: T[] }

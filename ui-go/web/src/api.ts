// Client for the server's /v1 API (docs/headless.md) plus the public media
// routes upstream exposes (/api/img, /api/files), which need no token.

const TOKEN_KEY = 'aitk.console.token';

export const getToken = (): string => {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
};

export const setToken = (token: string) => {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // private mode: the token lives for this tab only
  }
  memoryToken = token;
};

let memoryToken = getToken();
const token = () => memoryToken || getToken();

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

type Listener = () => void;
const unauthorizedListeners = new Set<Listener>();

/** Called whenever the server answers 401 (expired or revoked token). */
export const onUnauthorized = (fn: Listener) => {
  unauthorizedListeners.add(fn);
  return () => {
    unauthorizedListeners.delete(fn);
  };
};

const authHeaders = (): Record<string, string> => {
  const t = token();
  return t ? { Authorization: `Bearer ${t}` } : {};
};

async function parseError(res: Response): Promise<ApiError> {
  let code = 'http_' + res.status;
  let message = res.statusText || 'request failed';
  try {
    const body = await res.json();
    if (body?.error?.code) {
      code = body.error.code;
      message = body.error.message || message;
    } else if (typeof body?.error === 'string') {
      message = body.error;
    }
  } catch {
    // not JSON
  }
  return new ApiError(res.status, code, message);
}

export async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders() };
  let payload: BodyInit | undefined;
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }
  const res = await fetch(path, { method, headers, body: payload, signal });
  if (res.status === 401) {
    unauthorizedListeners.forEach(fn => fn());
    throw await parseError(res);
  }
  if (!res.ok) throw await parseError(res);
  if (res.status === 204) return null as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : null) as T;
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => request<T>('GET', path, undefined, signal),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body ?? {}),
  patch: <T>(path: string, body: unknown) => request<T>('PATCH', path, body),
  put: <T>(path: string, body: unknown) => request<T>('PUT', path, body),
  del: <T>(path: string) => request<T>('DELETE', path),
};

/** Query string from an object, skipping empty values. */
export const qs = (params: Record<string, string | number | boolean | undefined | null>) => {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') p.set(k, String(v));
  }
  const s = p.toString();
  return s ? '?' + s : '';
};

/** Multipart upload with progress (fetch cannot report upload progress). */
export function upload<T>(
  path: string,
  files: File[],
  field = 'files[]',
  onProgress?: (fraction: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach(f => form.append(field, f, f.name));
    const xhr = new XMLHttpRequest();
    xhr.open('POST', path);
    const t = token();
    if (t) xhr.setRequestHeader('Authorization', `Bearer ${t}`);
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: any = null;
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        // keep null
      }
      if (xhr.status === 401) unauthorizedListeners.forEach(fn => fn());
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else reject(new ApiError(xhr.status, body?.error?.code ?? 'http_' + xhr.status, body?.error?.message ?? xhr.statusText));
    };
    xhr.onerror = () => reject(new ApiError(0, 'network', 'network error'));
    xhr.send(form);
  });
}

// ---- signed links and raw uploads ----

/**
 * Downloads through a signed link (POST /v1/links): the browser streams the
 * file straight to disk, so multi-GB checkpoints and bundles never sit in
 * memory, and no token is needed on the GET.
 */
export async function signedDownload(path: string): Promise<void> {
  const link = await request<{ url: string; path: string; expires_at: string }>('POST', '/v1/links', { path });
  const a = document.createElement('a');
  a.href = link.path; // relative: same origin, also through the dev proxy
  a.rel = 'noopener';
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** A file under the training folder as an attachment (checkpoints). */
export const trainingFilePath = (rel: string) => `/v1/files/content${qs({ root: 'training', path: rel, download: 1 })}`;

/** PUT of raw bytes with upload progress, for chunked uploads. */
export function putBytes<T>(path: string, body: Blob, headers: Record<string, string>, onProgress: (loaded: number) => void, signal?: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', path);
    const t = token();
    if (t) xhr.setRequestHeader('Authorization', `Bearer ${t}`);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = e => onProgress(e.loaded);
    const onAbort = () => xhr.abort();
    signal?.addEventListener('abort', onAbort);
    const done = () => signal?.removeEventListener('abort', onAbort);
    xhr.onabort = () => {
      done();
      reject(new DOMException('aborted', 'AbortError'));
    };
    xhr.onerror = () => {
      done();
      reject(new ApiError(0, 'network', 'upload interrupted: network error'));
    };
    xhr.onload = () => {
      done();
      let b: any = null;
      try {
        b = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        // keep null
      }
      if (xhr.status === 401) unauthorizedListeners.forEach(fn => fn());
      if (xhr.status >= 200 && xhr.status < 300) resolve(b as T);
      else reject(new ApiError(xhr.status, b?.error?.code ?? 'http_' + xhr.status, b?.error?.message ?? xhr.statusText));
    };
    xhr.send(body);
  });
}

/** Content-Length of a GET, via HEAD (bundles are stored, so their size is known up front). */
export async function headSize(path: string, signal?: AbortSignal): Promise<number | null> {
  const res = await fetch(path, { method: 'HEAD', headers: authHeaders(), signal });
  const n = Number(res.headers.get('Content-Length'));
  return res.ok && n > 0 ? n : null;
}

// ---- media URLs (public upstream routes, like the upstream UI uses) ----

const splitPath = (p: string) => {
  const idx = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
  if (idx === -1) return ['', p];
  return [idx === 0 ? p[0] : p.slice(0, idx), p.slice(idx + 1)];
};

/** `<encoded folder>/<encoded name>`, as upstream's encodeFilePathForUrl. */
const encodePath = (p: string) => {
  const [folder, name] = splitPath(p);
  return folder ? `${encodeURIComponent(folder)}/${encodeURIComponent(name)}` : encodeURIComponent(name);
};

export const imgUrl = (absPath: string, thumb = false) => `/api/img/${encodePath(absPath)}${thumb ? '?thumb=1' : ''}`;
export const fileUrl = (absPath: string) => `/api/files/${encodePath(absPath)}`;

/** Joins a folder from the server with a relative name using its separator. */
export const joinServerPath = (folder: string, rel: string) => {
  const sep = folder.includes('\\') && !folder.includes('/') ? '\\' : '/';
  return folder.replace(/[\\/]+$/, '') + sep + rel.replace(/\//g, sep);
};

// ---- events (Server-Sent Events over fetch, so the bearer header works) ----

export type AitkEvent = { id: number; type: string; job_id: string | null; at: string; data: any };

/**
 * Follows GET /v1/events/stream until the signal aborts, reconnecting with
 * Last-Event-ID after drops. Live-only events (job.log) carry id 0.
 */
export async function streamEvents(
  params: { job_id?: string; types?: string[]; after?: number | 'now' },
  onEvent: (e: AitkEvent) => void,
  signal: AbortSignal,
  onStatus?: (connected: boolean) => void,
) {
  let lastId = typeof params.after === 'number' ? params.after : 0;
  // 'now' skips the stored history on the first connection; reconnects resume by id
  let fromNow = params.after === 'now';
  let backoff = 1000;
  while (!signal.aborted) {
    try {
      const headers: Record<string, string> = { ...authHeaders(), Accept: 'text/event-stream' };
      if (lastId) headers['Last-Event-ID'] = String(lastId);
      const url = '/v1/events/stream' + qs({ job_id: params.job_id, types: params.types?.join(','), after: lastId || (fromNow ? 'now' : undefined) });
      const res = await fetch(url, { headers, signal });
      if (res.status === 401) {
        unauthorizedListeners.forEach(fn => fn());
        return;
      }
      if (!res.ok || !res.body) throw new Error('stream ' + res.status);
      onStatus?.(true);
      backoff = 1000;
      fromNow = false;
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let ev: { id?: number; type?: string; data: string[] } = { data: [] };
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).replace(/\r$/, '');
          buf = buf.slice(nl + 1);
          if (line === '') {
            if (ev.data.length) {
              try {
                const parsed = JSON.parse(ev.data.join('\n')) as AitkEvent;
                if (parsed.id) lastId = parsed.id;
                onEvent(parsed);
              } catch {
                // malformed frame
              }
            }
            ev = { data: [] };
          } else if (line.startsWith(':')) {
            // comment / ping
          } else if (line.startsWith('id:')) {
            ev.id = Number(line.slice(3).trim());
          } else if (line.startsWith('event:')) {
            ev.type = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            ev.data.push(line.slice(5).replace(/^ /, ''));
          }
        }
      }
    } catch {
      if (signal.aborted) return;
    }
    onStatus?.(false);
    await new Promise(r => setTimeout(r, backoff));
    backoff = Math.min(backoff * 2, 15000);
  }
}

// ---- formatting ----

export const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

export const fmtBytes = (n: number | null | undefined) => {
  if (n == null || !isFinite(n)) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
};

export const fmtAgo = (iso: string | null | undefined) => {
  if (!iso) return '—';
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  // a few seconds ahead is clock skew between server and browser
  if (s < -5) return 'in ' + fmtDuration(-s);
  if (s < 5) return 'just now';
  return fmtDuration(s) + ' ago';
};

export const fmtDuration = (s: number) => {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86400).toFixed(1)}d`;
};

export const fmtTime = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString() : '—');

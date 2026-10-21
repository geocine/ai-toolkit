import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import type { JobState, Server } from '../lib/api';
import { href } from '../lib/route';

export function StatePill({ state }: { state: JobState | string }) {
  return <span className={`pill pill-${state}`}>{state}</span>;
}

export function NeedServer() {
  return (
    <div className="empty">
      <p className="empty-title">No server yet.</p>
      <p>
        Add the training server you want to work with — a RunPod pod, a box under your desk, or the dev proxy.
      </p>
      <a className="btn btn-ink" href={href({ page: 'servers' })}>
        Add a server
      </a>
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <p className="error-note">{msg}</p>;
}

export function Section({ title, kicker, children, actions }: { title: string; kicker?: string; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="section">
      <header className="section-head">
        <div>
          {kicker && <p className="kicker">{kicker}</p>}
          <h2>{title}</h2>
        </div>
        {actions && <div className="section-actions">{actions}</div>}
      </header>
      {children}
    </section>
  );
}

/** Loads data for the active server, refreshing on demand or on an interval. */
export function useLoad<T>(server: Server | null, fn: (s: Server) => Promise<T>, deps: unknown[], everyMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const reload = useCallback(async () => {
    if (!server) return;
    setLoading(true);
    try {
      setData(await fnRef.current(server));
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [server, ...deps]);
  useEffect(() => {
    reload();
    if (!everyMs) return;
    const t = setInterval(reload, everyMs);
    return () => clearInterval(t);
  }, [reload, everyMs]);
  return { data, error, loading, reload, setData };
}

export function Progress({ value, max }: { value: number; max: number | null }) {
  const pct = max ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="progress" role="progressbar" aria-valuenow={value} aria-valuemax={max ?? undefined}>
      <div className="progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

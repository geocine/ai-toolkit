import { CircleAlert, CircleCheck, Info, Server as ServerIcon, TriangleAlert } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import type { JobState, Server } from '../lib/api';
import { href } from '../lib/route';

/** Status colour per job or upload state (DESIGN.md, Status). */
export function tone(state: string): string {
  switch (state) {
    case 'running':
    case 'starting':
      return 'accent';
    case 'completed':
    case 'done':
      return 'ok';
    case 'failed':
      return 'danger';
    case 'interrupted':
      return 'warn';
    case 'stopped':
    case 'stopping':
      return 'muted';
    default:
      return 'quiet';
  }
}

const sentence = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);

export function StatePill({ state, label }: { state: JobState | string; label?: string }) {
  return (
    <span className={`badge tone-${tone(state)}${state === 'running' ? ' is-running' : ''}`}>
      <span className="dot" aria-hidden />
      {label ?? sentence(state)}
    </span>
  );
}

export function Empty({ icon, children, action }: { icon: ReactNode; children: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <span className="empty-icon">{icon}</span>
      <p>{children}</p>
      {action}
    </div>
  );
}

export function NeedServer({ title }: { title?: string }) {
  return (
    <>
      {title && <PageHeader title={title} />}
      <Empty
      icon={<ServerIcon />}
      action={
        <a className="btn btn-primary" href={href({ page: 'servers' })}>
          Add a server
        </a>
      }
    >
        Add the training server you want to work with: a RunPod pod, a box under your desk, or the dev proxy.
      </Empty>
    </>
  );
}

export function Callout({ kind = 'info', children }: { kind?: 'info' | 'warn' | 'danger' | 'ok'; children: ReactNode }) {
  const Icon = kind === 'danger' ? CircleAlert : kind === 'warn' ? TriangleAlert : kind === 'ok' ? CircleCheck : Info;
  return (
    <div className={`callout callout-${kind}`} role={kind === 'danger' ? 'alert' : undefined}>
      <Icon className="callout-icon" aria-hidden />
      <div className="callout-body">{children}</div>
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <Callout kind="danger">{msg}</Callout>;
}

export function PageHeader({ title, description, actions, children }: { title: ReactNode; description?: ReactNode; actions?: ReactNode; children?: ReactNode }) {
  return (
    <header className="page-head">
      <div className="page-head-main">
        <h1>{title}</h1>
        {description && <p className="page-desc">{description}</p>}
        {children}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Section({ title, description, children, actions }: { title: ReactNode; description?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="section">
      <header className="section-head">
        <div>
          <h2>{title}</h2>
          {description && <p className="section-desc">{description}</p>}
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

export function Progress({ value, max, indeterminate }: { value: number; max: number | null; indeterminate?: boolean }) {
  const pct = max ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className={indeterminate ? 'progress is-indeterminate' : 'progress'} role="progressbar" aria-valuenow={indeterminate ? undefined : value} aria-valuemax={max ?? undefined}>
      <div className="progress-fill" style={indeterminate ? undefined : { width: `${pct}%` }} />
    </div>
  );
}

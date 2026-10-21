// Small shared components, styled by styles.css (zinc design system).
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { AlertTriangle, CheckCircle2, Copy, Info, Loader2, X } from 'lucide-react';
import { ApiError } from './api';
import type { JobState } from './types';

// ---- toasts ----

type Toast = { id: number; kind: 'error' | 'success' | 'info'; text: string };
type ToastApi = { error: (e: unknown) => void; success: (t: string) => void; info: (t: string) => void };
const ToastCtx = createContext<ToastApi>({ error: () => {}, success: () => {}, info: () => {} });
export const useToast = () => useContext(ToastCtx);

export const errorText = (e: unknown) =>
  e instanceof ApiError ? `${e.message}${e.code && !e.code.startsWith('http_') ? ` (${e.code})` : ''}` : e instanceof Error ? e.message : String(e);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);
  const push = useCallback((kind: Toast['kind'], text: string) => {
    const id = next.current++;
    setToasts(t => [...t.slice(-3), { id, kind, text }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), kind === 'error' ? 8000 : 4000);
  }, []);
  const api = useRef<ToastApi>({
    error: e => push('error', errorText(e)),
    success: t => push('success', t),
    info: t => push('info', t),
  });
  return (
    <ToastCtx.Provider value={api.current}>
      {children}
      <div className="toasts">
        {toasts.map(t => (
          <div key={t.id} className={`toast toast-${t.kind}`}>
            {t.kind === 'error' ? <AlertTriangle size={14} /> : t.kind === 'success' ? <CheckCircle2 size={14} /> : <Info size={14} />}
            <span>{t.text}</span>
            <button className="icon-btn ghost" onClick={() => setToasts(x => x.filter(y => y.id !== t.id))} aria-label="Dismiss">
              <X size={12} />
            </button>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

// ---- confirm dialog ----

type ConfirmOpts = { title: string; body?: ReactNode; confirm?: string; danger?: boolean };
const ConfirmCtx = createContext<(o: ConfirmOpts) => Promise<boolean>>(async () => false);
export const useConfirm = () => useContext(ConfirmCtx);

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<(ConfirmOpts & { resolve: (v: boolean) => void }) | null>(null);
  const ask = useCallback((o: ConfirmOpts) => new Promise<boolean>(resolve => setState({ ...o, resolve })), []);
  const close = (v: boolean) => {
    state?.resolve(v);
    setState(null);
  };
  return (
    <ConfirmCtx.Provider value={ask}>
      {children}
      {state && (
        <Modal title={state.title} onClose={() => close(false)}>
          {state.body && <div className="modal-body">{state.body}</div>}
          <div className="modal-actions">
            <button className="btn" onClick={() => close(false)}>
              Cancel
            </button>
            <button className={state.danger ? 'btn btn-danger' : 'btn btn-primary'} onClick={() => close(true)} autoFocus>
              {state.confirm ?? 'Confirm'}
            </button>
          </div>
        </Modal>
      )}
    </ConfirmCtx.Provider>
  );
}

export function Modal({ title, onClose, children, wide }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const on = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={e => e.target === e.currentTarget && onClose()}>
      <div className={wide ? 'modal modal-wide' : 'modal'} role="dialog" aria-label={title}>
        <div className="modal-head">
          <h3>{title}</h3>
          <button className="icon-btn ghost" onClick={onClose} aria-label="Close">
            <X size={14} />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

// ---- bits ----

const stateClass: Record<string, string> = {
  running: 'ok',
  starting: 'ok',
  completed: 'done',
  queued: 'warn',
  stopping: 'warn',
  interrupted: 'warn',
  failed: 'bad',
  stopped: 'muted',
  draft: 'muted',
  done: 'done',
  error: 'bad',
};

export const StateBadge = ({ state }: { state: JobState | string }) => (
  <span className={`badge badge-${stateClass[state] ?? 'muted'}`}>
    {(state === 'running' || state === 'starting') && <span className="pulse" />}
    {state}
  </span>
);

export const Bar = ({ value, max, label }: { value: number; max: number; label?: string }) => {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="bar" title={label ?? `${pct.toFixed(0)}%`}>
      <div className="bar-fill" style={{ width: `${pct}%` }} />
    </div>
  );
};

export const Spinner = ({ size = 14 }: { size?: number }) => <Loader2 size={size} className="spin" />;

export const Loading = ({ what }: { what?: string }) => (
  <div className="loading">
    <Spinner /> <span>Loading{what ? ` ${what}` : ''}…</span>
  </div>
);

export const Empty = ({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) => (
  <div className="empty">
    {icon}
    <div className="empty-title">{title}</div>
    {children && <div className="empty-body">{children}</div>}
  </div>
);

export const Section = ({ title, actions, children }: { title: string; actions?: ReactNode; children: ReactNode }) => (
  <section className="section">
    <div className="section-head">
      <h2>{title}</h2>
      {actions && <div className="section-actions">{actions}</div>}
    </div>
    {children}
  </section>
);

export const Field = ({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) => (
  <label className="field">
    <span className="field-label">{label}</span>
    {children}
    {hint && <span className="field-hint">{hint}</span>}
  </label>
);

export const Mono = ({ children, title }: { children: ReactNode; title?: string }) => (
  <span className="mono" title={title}>
    {children}
  </span>
);

export function CopyButton({ text }: { text: string }) {
  const toast = useToast();
  return (
    <button
      className="icon-btn"
      onClick={() => navigator.clipboard?.writeText(text).then(() => toast.success('Copied'), () => toast.info(text))}
      aria-label="Copy"
      title="Copy"
    >
      <Copy size={12} />
    </button>
  );
}

/** Runs fn every ms while mounted (and right away); returns a manual refresh. */
export function usePoll(fn: (signal: AbortSignal) => Promise<void> | void, ms: number, deps: unknown[] = []) {
  const ref = useRef(fn);
  ref.current = fn;
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const ctrl = new AbortController();
    let stopped = false;
    const run = async () => {
      try {
        await ref.current(ctrl.signal);
      } catch {
        // callers surface their own errors
      }
    };
    run();
    const id = setInterval(() => !stopped && document.visibilityState !== 'hidden' && run(), ms);
    return () => {
      stopped = true;
      clearInterval(id);
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ms, tick, ...deps]);
  return () => setTick(t => t + 1);
}

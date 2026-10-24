// "When done" (docs/headless.md, Unattended runs): what the server does with
// itself once the work is finished, followed live for the banner and page.
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';
import { call, get, streamEvents, type Server } from './api';

export type LifeMode = 'keep' | 'stop' | 'destroy';
export type LifeState = 'off' | 'waiting' | 'countdown' | 'verifying' | 'acting' | 'held' | 'cancelled' | 'done' | 'failed';
export interface JobResult { job_id: string; job_name: string; safe: boolean; via: string; detail: string }
export interface Lifecycle {
  mode: LifeMode;
  on: 'completed' | 'any';
  delay_s: number;
  dry_run: boolean;
  state: LifeState;
  deadline: string | null;
  reason: string;
  provider: 'runpod' | 'vast' | 'quickpod' | 'command' | 'exit';
  job_id: string | null;
  job_name: string | null;
  results: JobResult[] | null;
  updated_at: string;
}

/** How it acts, finishing "Acts ...". */
export const providerHow: Record<Lifecycle['provider'], string> = {
  runpod: 'through RunPod',
  vast: 'through Vast.ai',
  quickpod: 'through the QuickPod API',
  command: 'by running AITK_WHEN_DONE_CMD',
  exit: 'by ending the server process',
};

/** The cause in the server's reason ("job x completed; destroying ... unless cancelled"). */
export const cause = (l: Lifecycle) => {
  const first = (l.reason || '').split(';')[0].trim();
  return first ? first[0].toUpperCase() + first.slice(1) + '.' : '';
};

export const modeVerb = (m: LifeMode) => (m === 'destroy' ? 'Destroying' : m === 'stop' ? 'Stopping' : 'Keeping');

/** States worth a banner on every page. */
export const needsAttention = (l: Lifecycle | null) => !!l && ['countdown', 'verifying', 'acting', 'held', 'failed'].includes(l.state);

interface Ctx {
  life: Lifecycle | null;
  error: unknown;
  set: (l: Lifecycle) => void;
  reload: () => void;
  act: (path: 'cancel' | 'now' | 'extend', body?: unknown) => Promise<Lifecycle>;
  patch: (p: Partial<Pick<Lifecycle, 'mode' | 'on' | 'delay_s' | 'dry_run'>>) => Promise<Lifecycle>;
}

const LifeCtx = createContext<Ctx | null>(null);

export function LifecycleProvider({ server, children }: { server: Server | null; children: ReactNode }) {
  const [life, setLife] = useState<Lifecycle | null>(null);
  const [error, setError] = useState<unknown>(null);

  const reload = useCallback(() => {
    if (!server) return;
    get<Lifecycle>(server, '/v1/lifecycle').then(
      l => {
        setLife(l);
        setError(null);
      },
      e => setError(e),
    );
  }, [server]);

  useEffect(() => {
    setLife(null);
    if (!server) return;
    reload();
    const poll = setInterval(reload, 30000);
    const stop = streamEvents(
      server,
      'types=lifecycle.state',
      e => e.data?.state && setLife(e.data as Lifecycle),
      on => on && reload(),
    );
    return () => {
      clearInterval(poll);
      stop();
    };
  }, [server, reload]);

  const act = useCallback(
    async (path: 'cancel' | 'now' | 'extend', body?: unknown) => {
      const l = await call<Lifecycle>(server!, 'POST', `/v1/lifecycle/${path}`, body ?? {});
      setLife(l);
      return l;
    },
    [server],
  );
  const patch = useCallback(
    async (p: Partial<Pick<Lifecycle, 'mode' | 'on' | 'delay_s' | 'dry_run'>>) => {
      const l = await call<Lifecycle>(server!, 'PATCH', '/v1/lifecycle', p);
      setLife(l);
      return l;
    },
    [server],
  );

  return <LifeCtx.Provider value={{ life, error, set: setLife, reload, act, patch }}>{children}</LifeCtx.Provider>;
}

export function useLifecycle(): Ctx {
  const v = useContext(LifeCtx);
  if (!v) throw new Error('useLifecycle outside LifecycleProvider');
  return v;
}

/** Re-renders every `ms` while mounted, for countdowns. */
export function useNow(ms = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(t);
  }, [ms]);
  return now;
}

/** 4:05, or 1:02:09 past an hour. */
export function clock(sec: number): string {
  const s = Math.max(0, Math.ceil(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${r}` : `${m}:${r}`;
}

// ---- notifications ----

export type NotifyKind = 'started' | 'resumed' | 'sample' | 'checkpoint' | 'completed' | 'failed' | 'interrupted' | 'lease_lost' | 'sync_failed' | 'lifecycle';
export interface NotifySink { id: string; kind: 'ntfy' | 'discord' | 'telegram' | 'pushover' | 'webhook'; target: string; source: 'env' | 'api'; created_at: string }
export interface NotifyConfig { sinks: NotifySink[]; events: NotifyKind[] | null; kinds: NotifyKind[]; public_url: string; studio_url: string; sample_every_s: number }
export interface TestResult { sink: string; kind: string; target: string; ok: boolean; detail: string }

export const kindLabel: Record<string, string> = {
  started: 'Run started',
  resumed: 'Run resumed',
  sample: 'New samples',
  checkpoint: 'Checkpoint saved off the server',
  completed: 'Run completed',
  failed: 'Run failed',
  interrupted: 'Run interrupted',
  lease_lost: 'Another server took the run',
  sync_failed: 'Sync failed',
  lifecycle: 'When done (countdown, held, destroyed)',
};

export const sinkLabel: Record<NotifySink['kind'], string> = {
  ntfy: 'ntfy',
  discord: 'Discord',
  telegram: 'Telegram',
  pushover: 'Pushover',
  webhook: 'Webhook',
};

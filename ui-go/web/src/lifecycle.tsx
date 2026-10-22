// "When done" and notifications (docs/headless.md, Unattended runs): the
// Settings sections, plus a countdown/held notice for the sidebar and overview.
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';
import { BellRing, CheckCircle2, ChevronRight, Clock, Plus, Send, Trash2, TriangleAlert, XCircle } from 'lucide-react';
import { ApiError, api, request, streamEvents } from './api';
import { href } from './router';
import type { Lifecycle, LifeMode, NotifyConfig, NotifyTest } from './types';
import { Callout, Field, Loading, Section, Spinner, useConfirm, useToast } from './ui';

// ---- shared state ----

type Ctx = { life: Lifecycle | null; set: (l: Lifecycle) => void; reload: () => void };
const LifeCtx = createContext<Ctx>({ life: null, set: () => {}, reload: () => {} });
export const useLifecycle = () => useContext(LifeCtx);

export function LifecycleProvider({ children }: { children: ReactNode }) {
  const [life, setLife] = useState<Lifecycle | null>(null);
  const reload = useCallback(() => {
    api.get<Lifecycle>('/v1/lifecycle').then(setLife, () => {});
  }, []);
  useEffect(() => {
    reload();
    const poll = setInterval(reload, 30000);
    const ctrl = new AbortController();
    streamEvents({ types: ['lifecycle.state'], after: 'now' }, e => e.data?.state && setLife(e.data as Lifecycle), ctrl.signal, on => on && reload());
    return () => {
      clearInterval(poll);
      ctrl.abort();
    };
  }, [reload]);
  return <LifeCtx.Provider value={{ life, set: setLife, reload }}>{children}</LifeCtx.Provider>;
}

function useNow(ms = 1000) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(t);
  }, [ms]);
  return now;
}

const clock = (sec: number) => {
  const s = Math.max(0, Math.ceil(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${r}` : `${m}:${r}`;
};

const verb = (m: LifeMode) => (m === 'destroy' ? 'Destroying' : m === 'stop' ? 'Stopping' : 'Keeping');
const how: Record<Lifecycle['provider'], string> = {
  runpod: 'through RunPod',
  vast: 'through Vast.ai',
  command: 'by running AITK_WHEN_DONE_CMD',
  exit: 'by ending the server process',
};
const cause = (l: Lifecycle) => {
  const first = (l.reason || '').split(';')[0].trim();
  return first ? first[0].toUpperCase() + first.slice(1) + '.' : '';
};
const attention = (l: Lifecycle | null) => !!l && ['countdown', 'verifying', 'acting', 'held', 'failed'].includes(l.state);

/** A countdown or held server: a compact notice that links to Settings. */
export function LifeNotice({ compact }: { compact?: boolean }) {
  const { life } = useLifecycle();
  const now = useNow();
  if (!life || !attention(life)) return null;
  const warn = life.state === 'held' || life.state === 'failed';
  const text =
    life.state === 'countdown' && life.deadline
      ? `${verb(life.mode)} in ${clock((new Date(life.deadline).getTime() - now) / 1000)}`
      : life.state === 'held'
        ? 'Held: results only here'
        : life.state === 'failed'
          ? 'When done failed'
          : life.state === 'verifying'
            ? 'Checking results'
            : `${verb(life.mode)} now`;
  return (
    <a className={`life-notice ${warn ? 'warn' : 'info'}${compact ? ' compact' : ''}`} href={href('settings') + '?when-done'} title={life.reason}>
      {warn ? <TriangleAlert size={14} /> : <Clock size={14} />}
      <span className="grow num">
        {text}
        {life.dry_run && <span className="muted"> · dry run</span>}
      </span>
      <ChevronRight size={14} className="muted" />
    </a>
  );
}

// ---- settings: when done ----

const MODES: { id: LifeMode; label: string; hint: string }[] = [
  { id: 'keep', label: 'Keep', hint: 'The server stays up.' },
  { id: 'stop', label: 'Stop', hint: 'Stops the machine once results are off it; the disk is kept.' },
  { id: 'destroy', label: 'Destroy', hint: 'Deletes the machine once results are off it.' },
];

export function WhenDoneSection() {
  const toast = useToast();
  const confirm = useConfirm();
  const { life, set } = useLifecycle();
  const now = useNow();
  const [busy, setBusy] = useState('');
  const [delay, setDelay] = useState('');
  useEffect(() => {
    if (life) setDelay(String(Math.round(life.delay_s / 60)));
  }, [life?.delay_s]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!life)
    return (
      <Section title="When done">
        <Loading />
      </Section>
    );

  const run = async (what: string, fn: () => Promise<Lifecycle>, ok?: string) => {
    setBusy(what);
    try {
      set(await fn());
      if (ok) toast.success(ok);
    } catch (e) {
      toast.error(e);
    } finally {
      setBusy('');
    }
  };
  const patch = (p: Partial<Lifecycle>) => run('patch', () => request<Lifecycle>('PATCH', '/v1/lifecycle', p), 'Saved');
  const now_ = async () => {
    if (!(await confirm({ title: life.mode === 'destroy' ? 'Destroy the server now' : 'Stop the server now', body: <>Skips the countdown. Finished jobs are checked first; any with results only on this server hold it up.{life.dry_run ? ' Dry run: only logged.' : ''}</>, confirm: life.mode === 'destroy' ? 'Destroy now' : 'Stop now', danger: true }))) return;
    run('now', () => api.post<Lifecycle>('/v1/lifecycle/now'));
  };
  const off = life.mode === 'keep';
  const counting = life.state === 'countdown' && life.deadline;
  const held = life.state === 'held' || life.state === 'failed';

  return (
    <Section title="When done">
      <div className="card" id="when-done">
        {counting && (
          <div className="countdown">
            <div>
              <div className="small muted">
                {verb(life.mode)} this server in{life.dry_run ? ' · dry run' : ''}
              </div>
              <div className="countdown-time num">{clock((new Date(life.deadline!).getTime() - now) / 1000)}</div>
              <div className="small muted">
                {cause(life)} Acts {how[life.provider]}.
              </div>
            </div>
            <div className="row-wrap">
              <button className="btn btn-primary" disabled={!!busy} onClick={() => run('cancel', () => api.post<Lifecycle>('/v1/lifecycle/cancel'), 'Kept')}>
                Keep server
              </button>
              <button className="btn" disabled={!!busy} onClick={() => run('extend', () => api.post<Lifecycle>('/v1/lifecycle/extend', { minutes: 30 }))}>
                <Plus size={14} /> 30 min
              </button>
              <button className="btn btn-destructive" disabled={!!busy} onClick={now_}>
                {life.mode === 'destroy' ? 'Destroy now' : 'Stop now'}
              </button>
            </div>
          </div>
        )}
        {held && (
          <div style={{ marginBottom: 16 }}>
            <Callout tone={life.state === 'failed' ? 'danger' : 'warn'}>
              <b>{life.state === 'held' ? 'Held: the server stays up.' : 'When done failed.'}</b>{' '}
              {life.state === 'held' ? 'Some finished jobs have results only on this server (below). Push them to a library or set a sync remote, then check again.' : life.reason}
              <div className="row-wrap" style={{ marginTop: 10 }}>
                <button className="btn btn-sm" disabled={!!busy} onClick={now_}>
                  Check again
                </button>
                <button className="btn btn-sm btn-ghost" disabled={!!busy} onClick={() => run('cancel', () => api.post<Lifecycle>('/v1/lifecycle/cancel'), 'Kept')}>
                  Keep server
                </button>
              </div>
            </Callout>
          </div>
        )}
        {!counting && !held && (
          <div className="life-line">
            <span className="status">
              <span className="dot" style={{ ['--c' as string]: life.state === 'done' ? 'var(--ok)' : 'var(--text-3)' }} />
              {stateText[life.state] ?? life.state}
            </span>
            {life.reason && <span className="small muted">{life.reason}</span>}
          </div>
        )}
        <div className="form-grid life-grid">
          <Field label="Then" hint={MODES.find(m => m.id === life.mode)?.hint}>
            <div className="seg seg-text" role="radiogroup" aria-label="When done">
              {MODES.map(m => (
                <button key={m.id} type="button" role="radio" aria-checked={life.mode === m.id} className={life.mode === m.id ? 'on' : ''} onClick={() => life.mode !== m.id && patch({ mode: m.id })}>
                  {m.label}
                </button>
              ))}
            </div>
          </Field>
          <Field label="When">
            <select value={life.on} disabled={off} onChange={e => patch({ on: e.target.value as Lifecycle['on'] })}>
              <option value="completed">A job completes</option>
              <option value="any">A job completes or fails</option>
            </select>
          </Field>
          <Field label="Wait (minutes)" hint="Any new job cancels the countdown.">
            <input type="number" min={0} value={delay} disabled={off} onChange={e => setDelay(e.target.value)} onBlur={() => delay !== '' && Number(delay) * 60 !== life.delay_s && patch({ delay_s: Math.max(0, Number(delay)) * 60 })} />
          </Field>
          <label className="check">
            <input type="checkbox" checked={life.dry_run} onChange={e => patch({ dry_run: e.target.checked })} /> Dry run (only log and notify)
          </label>
        </div>
        <div className="small muted" style={{ marginTop: 14 }}>
          Acts {how[life.provider]}. AITK_WHEN_DONE* in the environment apply again at the next start.
          {busy === 'patch' && (
            <>
              {' '}
              <Spinner size={12} />
            </>
          )}
        </div>
        {!!life.results?.length && (
          <div className="table-wrap" style={{ marginTop: 16 }}>
            <table>
              <thead>
                <tr>
                  <th>Job</th>
                  <th>Results</th>
                </tr>
              </thead>
              <tbody>
                {life.results.map(r => (
                  <tr key={r.job_id}>
                    <td className="nowrap">
                      <a className="cell-title" href={href('jobs', r.job_id)}>
                        {r.job_name}
                      </a>
                    </td>
                    <td>
                      <span className={r.safe ? 'ok-text row' : 'danger-text row'} style={{ gap: 6, alignItems: 'flex-start' }}>
                        {r.safe ? <CheckCircle2 size={14} style={{ marginTop: 2, flex: 'none' }} /> : <XCircle size={14} style={{ marginTop: 2, flex: 'none' }} />}
                        <span className="small" style={{ color: 'var(--text-2)' }}>
                          {r.safe ? `Kept via ${r.via}${r.detail ? ` · ${r.detail}` : ''}` : r.detail || 'Only on this server'}
                        </span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Section>
  );
}

const stateText: Record<string, string> = {
  off: 'Off: the server is kept',
  waiting: 'Waiting for the work to finish',
  verifying: 'Checking that results are off the server',
  acting: 'Acting now',
  cancelled: 'Cancelled until the next job ends',
  done: 'Done',
};

// ---- settings: notifications ----

const KIND_LABEL: Record<string, string> = {
  started: 'Started',
  resumed: 'Resumed',
  sample: 'Samples',
  checkpoint: 'Checkpoint off the server',
  completed: 'Completed',
  failed: 'Failed',
  interrupted: 'Interrupted',
  lease_lost: 'Lease lost',
  sync_failed: 'Sync failed',
  lifecycle: 'When done',
};
const SINK_LABEL: Record<string, string> = { ntfy: 'ntfy', discord: 'Discord', telegram: 'Telegram', pushover: 'Pushover', webhook: 'Webhook' };

// How to get each kind of destination URL; picking one also sets the
// input's placeholder.
const ext = (url: string, label: string) => (
  <a className="link" href={url} target="_blank" rel="noreferrer">
    {label}
  </a>
);
const SINK_GUIDES: { id: string; label: string; placeholder: string; steps: ReactNode[] }[] = [
  {
    id: 'ntfy',
    label: 'ntfy',
    placeholder: 'https://ntfy.sh/your-topic',
    steps: [
      <>Install the ntfy app on your phone, or open {ext('https://ntfy.sh/app', 'ntfy.sh/app')} in a browser.</>,
      <>Subscribe to a topic with a name nobody could guess, like <span className="mono">aitk-7f3k9q</span>. Anyone who knows the name can read it.</>,
      <>
        Add <span className="mono">https://ntfy.sh/&lt;topic&gt;</span>. A server of your own: <span className="mono">ntfy+https://&lt;host&gt;/&lt;topic&gt;</span>, with an access token{' '}
        <span className="mono">https://:&lt;token&gt;@&lt;host&gt;/&lt;topic&gt;</span>.
      </>,
    ],
  },
  {
    id: 'discord',
    label: 'Discord',
    placeholder: 'https://discord.com/api/webhooks/…',
    steps: [
      <>In Discord, open the channel's settings, then Integrations › Webhooks › New Webhook.</>,
      <>Click Copy Webhook URL and add it here.</>,
    ],
  },
  {
    id: 'telegram',
    label: 'Telegram',
    placeholder: 'telegram://<bot token>@<chat id>',
    steps: [
      <>
        Message {ext('https://t.me/BotFather', '@BotFather')}, send <span className="mono">/newbot</span> and copy the token it gives you (<span className="mono">123456:ABC…</span>).
      </>,
      <>Send your new bot any message, or add it to a group.</>,
      <>
        Open <span className="mono">https://api.telegram.org/bot&lt;token&gt;/getUpdates</span> and copy the number after <span className="mono">"chat":{'{'}"id":</span>. A group's id starts with -.
      </>,
      <>
        Add <span className="mono">telegram://&lt;token&gt;@&lt;chat id&gt;</span>.
      </>,
    ],
  },
  {
    id: 'pushover',
    label: 'Pushover',
    placeholder: 'pushover://<api token>@<user key>',
    steps: [
      <>Install the Pushover app and sign in. Your User Key is at the top of {ext('https://pushover.net', 'pushover.net')}.</>,
      <>
        {ext('https://pushover.net/apps/build', 'Create an application')} (any name, e.g. AI Toolkit) and copy its API Token.
      </>,
      <>
        Add <span className="mono">pushover://&lt;api token&gt;@&lt;user key&gt;</span>. Add <span className="mono">/&lt;device&gt;</span> to the end to reach one device only. Failed runs arrive at high priority.
      </>,
    ],
  },
  {
    id: 'webhook',
    label: 'Webhook',
    placeholder: 'https://example.com/hook',
    steps: [
      <>
        Any other <span className="mono">https://</span> URL gets a POST with JSON: <span className="mono">kind, level, title, body, url, job_id, job_name, at, server</span>, and{' '}
        <span className="mono">image_jpeg_base64</span> when there is a sample grid.
      </>,
      <>
        Start the URL with <span className="mono">webhook+</span> to get JSON from a URL that looks like ntfy or Discord.
      </>,
    ],
  },
];


export function NotificationsSection() {
  const toast = useToast();
  const confirm = useConfirm();
  const [cfg, setCfg] = useState<NotifyConfig | null>(null);
  const [url, setUrl] = useState('');
  const [guide, setGuide] = useState(SINK_GUIDES[0]);
  const [tests, setTests] = useState<Record<string, NotifyTest | 'busy'>>({});
  const [links, setLinks] = useState({ studio: '', pub: '', every: '15' });
  const load = useCallback(() => api.get<NotifyConfig>('/v1/notify').then(setCfg, toast.error), [toast]);
  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    if (cfg) setLinks({ studio: cfg.studio_url, pub: cfg.public_url, every: String(Math.round(cfg.sample_every_s / 60)) });
  }, [cfg?.studio_url, cfg?.public_url, cfg?.sample_every_s]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!cfg)
    return (
      <Section title="Notifications">
        <Loading />
      </Section>
    );

  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.post('/v1/notify/sinks', { url: url.trim() });
      setUrl('');
      load();
      toast.success('Added');
    } catch (err) {
      toast.error(err instanceof ApiError && err.status === 409 ? new Error('That destination is already set up') : err);
    }
  };
  const test = async (sink?: string) => {
    const ids = sink ? [sink] : cfg.sinks.map(s => s.id);
    setTests(t => ({ ...t, ...Object.fromEntries(ids.map(i => [i, 'busy' as const])) }));
    try {
      const r = await api.post<{ items: NotifyTest[] }>('/v1/notify/test', sink ? { sink } : {});
      setTests(t => ({ ...t, ...Object.fromEntries(r.items.map(i => [i.sink, i])) }));
    } catch (err) {
      toast.error(err);
      setTests(t => Object.fromEntries(Object.entries(t).filter(([k]) => !ids.includes(k))));
    }
  };
  const remove = async (id: string) => {
    if (!(await confirm({ title: 'Remove destination', body: 'Stop sending notifications there?', confirm: 'Remove', danger: true }))) return;
    try {
      await api.del(`/v1/notify/sinks/${encodeURIComponent(id)}`);
      load();
    } catch (err) {
      toast.error(err);
    }
  };
  const patch = async (p: Partial<NotifyConfig>, ok?: string) => {
    try {
      setCfg(await request<NotifyConfig>('PATCH', '/v1/notify', p));
      if (ok) toast.success(ok);
    } catch (err) {
      toast.error(err);
    }
  };
  const enabled = new Set(cfg.events ?? cfg.kinds);
  const toggle = (k: string, on: boolean) => {
    const next = cfg.kinds.filter(x => (x === k ? on : enabled.has(x)));
    patch({ events: next.length === cfg.kinds.length ? null : next });
  };

  return (
    <Section
      title="Notifications"
      actions={
        cfg.sinks.length > 0 && (
          <button className="btn btn-sm" onClick={() => test()}>
            <Send size={14} /> Test all
          </button>
        )
      }
    >
      <div className="grid grid-2">
        <div className="card">
          <div className="card-head">
            <h3 className="card-title">Destinations</h3>
          </div>
          {cfg.sinks.length === 0 ? (
            <div className="small muted row" style={{ gap: 8, marginBottom: 12 }}>
              <BellRing size={14} /> None yet.
            </div>
          ) : (
            <div className="sink-list">
              {cfg.sinks.map(s => {
                const t = tests[s.id];
                return (
                  <div key={s.id} className="sink-row">
                    <div className="grow" style={{ minWidth: 0 }}>
                      <div className="row" style={{ gap: 8 }}>
                        <span className="cell-title">{SINK_LABEL[s.kind] ?? s.kind}</span>
                        {s.source === 'env' && <span className="tag">From env</span>}
                      </div>
                      <div className="mono small muted ellipsis">{s.target}</div>
                      {t && t !== 'busy' && <div className={t.ok ? 'ok-text small' : 'danger-text small'}>{t.ok ? 'Sent' : t.detail}</div>}
                    </div>
                    <div className="row" style={{ gap: 4, flex: 'none' }}>
                      <button className="btn btn-sm" onClick={() => test(s.id)} disabled={t === 'busy'}>
                        {t === 'busy' ? <Spinner /> : <Send size={14} />} Test
                      </button>
                      {s.source === 'api' && (
                        <button className="icon-btn danger" onClick={() => remove(s.id)} aria-label="Remove" title="Remove">
                          <Trash2 size={14} />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <form onSubmit={add} className="stack" style={{ gap: 8, marginTop: 12 }}>
            <div className="row" style={{ gap: 8, flexWrap: 'nowrap' }}>
              <input type="text" value={url} onChange={e => setUrl(e.target.value)} placeholder={guide.placeholder} aria-label="Destination URL" spellCheck={false} />
              <button className="btn btn-primary" type="submit" disabled={!url.trim()}>
                <Plus size={14} /> Add
              </button>
            </div>
          </form>
          <div className="sink-guide">
            <div className="field-label">How to set one up</div>
            <div className="seg seg-text" role="radiogroup" aria-label="Destination type">
              {SINK_GUIDES.map(g => (
                <button key={g.id} type="button" role="radio" aria-checked={guide.id === g.id} className={guide.id === g.id ? 'on' : ''} onClick={() => setGuide(g)}>
                  {g.label}
                </button>
              ))}
            </div>
            <ol className="sink-steps">
              {guide.steps.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ol>
            <div className="field-hint">The URL holds its secret, so the server keeps it and shows it shortened.</div>
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <h3 className="card-title">Events</h3>
            <span className="small muted">{cfg.events === null ? 'All' : `${enabled.size} of ${cfg.kinds.length}`}</span>
          </div>
          <div className="kinds">
            {cfg.kinds.map(k => (
              <label key={k} className="check">
                <input type="checkbox" checked={enabled.has(k)} onChange={e => toggle(k, e.target.checked)} /> {KIND_LABEL[k] ?? k}
              </label>
            ))}
          </div>
          <form
            className="form-grid"
            style={{ marginTop: 16 }}
            onSubmit={e => {
              e.preventDefault();
              patch({ studio_url: links.studio.trim(), public_url: links.pub.trim(), sample_every_s: Math.max(1, Number(links.every) || 15) * 60 }, 'Saved');
            }}
          >
            <Field label="Studio URL" hint="Links open runs here">
              <input type="text" value={links.studio} onChange={e => setLinks({ ...links, studio: e.target.value })} placeholder="https://…/studio/" />
            </Field>
            <Field label="Public URL" hint="This server, as seen from outside">
              <input type="text" value={links.pub} onChange={e => setLinks({ ...links, pub: e.target.value })} placeholder="Detected on RunPod / Vast.ai" />
            </Field>
            <Field label="Samples every (min)">
              <input type="number" min={1} value={links.every} onChange={e => setLinks({ ...links, every: e.target.value })} />
            </Field>
            <div className="form-actions" style={{ gridColumn: '1 / -1', marginTop: 0 }}>
              <button className="btn btn-primary" type="submit">
                Save
              </button>
            </div>
          </form>
        </div>
      </div>
    </Section>
  );
}

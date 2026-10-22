import { BellRing, CircleCheck, CircleX, Clock, Plus, Send, Trash2 } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';
import { Confirm } from '../components/Dialog';
import { Callout, ErrorNote, NeedServer, PageHeader, Section, useLoad } from '../components/ui';
import { ApiError, call, del, get, post, type Server } from '../lib/api';
import { useServer } from '../lib/context';
import {
  clock,
  kindLabel,
  modeVerb,
  cause,
  providerHow,
  sinkLabel,
  useLifecycle,
  useNow,
  type LifeMode,
  type Lifecycle as Life,
  type NotifyConfig,
  type NotifyKind,
  type TestResult,
} from '../lib/lifecycle';
import { href } from '../lib/route';

const modes: { value: LifeMode; label: string; hint: string }[] = [
  { value: 'keep', label: 'Keep', hint: 'The server stays up. Nothing happens on its own.' },
  { value: 'stop', label: 'Stop', hint: 'Stops the machine once the results are safely off it. Disk is kept and billed.' },
  { value: 'destroy', label: 'Destroy', hint: 'Deletes the machine once the results are safely off it. Nothing more is billed.' },
];

const delays = [5, 15, 30, 60];

const stateText: Record<string, string> = {
  off: 'Off: the server is kept',
  waiting: 'Waiting for the work to finish',
  countdown: 'Counting down',
  verifying: 'Checking that every result is off the server',
  acting: 'Acting now',
  held: 'Held: results only on this server',
  cancelled: 'Cancelled until the next run ends',
  done: 'Done',
  failed: 'Failed',
};

/** The on-the-go screen: what happens when training is done, and where the news goes. */
export function Lifecycle() {
  const server = useServer();
  if (!server) return <NeedServer title="When done" />;
  return <LifecyclePage server={server} />;
}

function LifecyclePage({ server }: { server: Server }) {
  const { life, error, act, patch } = useLifecycle();
  const [busy, setBusy] = useState('');
  const [actError, setActError] = useState<unknown>(null);
  const [confirmNow, setConfirmNow] = useState(false);

  const run = async (what: string, fn: () => Promise<unknown>) => {
    setBusy(what);
    setActError(null);
    try {
      await fn();
    } catch (e) {
      setActError(e);
    } finally {
      setBusy('');
    }
  };

  return (
    <>
      <PageHeader title="When done" description="What this server does with itself once training ends, and where it tells you about it." />
      <ErrorNote error={error} />
      {life && (
        <>
          {life.state === 'countdown' && life.deadline && (
            <Countdown
              life={life}
              busy={busy}
              onKeep={() => run('cancel', () => act('cancel'))}
              onExtend={() => run('extend', () => act('extend', { minutes: 30 }))}
              onNow={() => setConfirmNow(true)}
            />
          )}
          {life.state !== 'countdown' && <StatusCard life={life} busy={busy} onKeep={() => run('cancel', () => act('cancel'))} onNow={() => setConfirmNow(true)} />}
          <ErrorNote error={actError} />
          {!!life.results?.length && <Results life={life} />}
          <Settings life={life} patch={patch} />
        </>
      )}
      <Notifications server={server} />
      {confirmNow && life && (
        <Confirm
          title={life.mode === 'destroy' ? 'Destroy the server now' : 'Stop the server now'}
          confirm={life.mode === 'destroy' ? 'Destroy now' : 'Stop now'}
          busy={busy === 'now'}
          onClose={() => setConfirmNow(false)}
          onConfirm={() =>
            run('now', async () => {
              await act('now');
              setConfirmNow(false);
            })
          }
        >
          Skips the rest of the countdown. Every finished run is checked first; if one has results only on this server, nothing happens and the server is held.
          {life.dry_run ? ' This is a dry run: it is only logged.' : ''}
        </Confirm>
      )}
    </>
  );
}

function Countdown({ life, busy, onKeep, onExtend, onNow }: { life: Life; busy: string; onKeep: () => void; onExtend: () => void; onNow: () => void }) {
  const now = useNow();
  const left = (new Date(life.deadline!).getTime() - now) / 1000;
  return (
    <section className="card countdown" aria-live="polite">
      <div className="countdown-top">
        <span className="countdown-icon">
          <Clock aria-hidden />
        </span>
        <div className="countdown-text">
          <p className="countdown-label">
            {modeVerb(life.mode)} this server in
            {life.dry_run && <span className="badge tone-quiet">Dry run</span>}
          </p>
          <p className="countdown-time num">{clock(left)}</p>
          <p className="countdown-reason">
            {cause(life) || 'The work is finished.'} Acts {providerHow[life.provider]}.
          </p>
        </div>
      </div>
      <div className="countdown-actions">
        <button className="btn btn-primary" onClick={onKeep} disabled={!!busy}>
          Keep server
        </button>
        <button className="btn" onClick={onExtend} disabled={!!busy}>
          <Plus />
          30 min
        </button>
        <button className="btn btn-danger" onClick={onNow} disabled={!!busy}>
          {life.mode === 'destroy' ? 'Destroy now' : 'Stop now'}
        </button>
      </div>
    </section>
  );
}

function StatusCard({ life, busy, onKeep, onNow }: { life: Life; busy: string; onKeep: () => void; onNow: () => void }) {
  const tone = life.state === 'done' ? 'ok' : 'quiet';
  if (life.state === 'held' || life.state === 'failed') {
    const held = life.state === 'held';
    const unsafe = (life.results ?? []).filter(r => !r.safe).length;
    return (
      <div className="life-held">
        <Callout kind={held ? 'warn' : 'danger'}>
          <p>
            <strong>{held ? 'Held: the server stays up.' : 'When done failed.'}</strong>{' '}
            {held && unsafe
              ? `${unsafe === 1 ? 'One finished run is' : `${unsafe} finished runs are`} only on this server. Save ${unsafe === 1 ? 'it' : 'them'} to a library (Export on the run) or give ${unsafe === 1 ? 'it' : 'them'} a sync remote, then check again.`
              : life.reason}
          </p>
        </Callout>
        <div className="countdown-actions">
          <button className="btn" onClick={onNow} disabled={!!busy}>
            Check again
          </button>
          <button className="btn btn-ghost" onClick={onKeep} disabled={!!busy}>
            Keep server
          </button>
        </div>
      </div>
    );
  }
  return (
    <div className={`life-status tone-${tone}`}>
      <span className="dot" aria-hidden />
      <div>
        <p className="life-status-title">{stateText[life.state] ?? life.state}</p>
        {life.reason && <p className="life-status-reason">{life.reason}</p>}
      </div>
      {(life.state === 'verifying' || life.state === 'acting') && (
        <button className="btn btn-sm" onClick={onKeep} disabled={!!busy}>
          Keep server
        </button>
      )}
    </div>
  );
}

function Results({ life }: { life: Life }) {
  return (
    <Section title="Finished runs" description="Each run's newest model file must be on a remote before the server goes away.">
      <ul className="results">
        {life.results!.map(r => (
          <li key={r.job_id} className="result">
            {r.safe ? <CircleCheck className="result-icon ok" aria-label="Safe" /> : <CircleX className="result-icon danger" aria-label="Not safe" />}
            <div className="result-main">
              <a className="result-name" href={href({ page: 'run', id: r.job_id })}>
                {r.job_name}
              </a>
              <p className="result-detail">
                {r.safe ? `Kept via ${r.via}${r.detail ? ` · ${r.detail}` : ''}` : r.detail || 'Only on this server'}
              </p>
            </div>
          </li>
        ))}
      </ul>
    </Section>
  );
}

function Settings({ life, patch }: { life: Life; patch: (p: Partial<Life>) => Promise<Life> }) {
  const [error, setError] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const [custom, setCustom] = useState(String(Math.round(life.delay_s / 60)));
  useEffect(() => setCustom(String(Math.round(life.delay_s / 60))), [life.delay_s]);
  const save = async (p: Partial<Life>) => {
    setSaving(true);
    setError(null);
    try {
      await patch(p);
    } catch (e) {
      setError(e);
    } finally {
      setSaving(false);
    }
  };
  const minutes = Math.round(life.delay_s / 60);
  const off = life.mode === 'keep';
  return (
    <Section title="After training" actions={saving ? <span className="save-state">Saving…</span> : undefined}>
      <div className="card life-settings">
        <Row label="Then" hint={modes.find(m => m.value === life.mode)?.hint}>
          <Seg value={life.mode} options={modes} onChange={v => save({ mode: v })} label="When done" />
        </Row>
        <Row label="When" hint={life.on === 'any' ? 'Also after a run fails on its own.' : 'Only after a run completes.'}>
          <Seg
            value={life.on}
            options={[
              { value: 'completed', label: 'Completed' },
              { value: 'any', label: 'Completed or failed' },
            ]}
            onChange={v => save({ on: v })}
            label="Trigger"
            disabled={off}
          />
        </Row>
        <Row label="Wait" hint="Time to step in before it acts; any new run cancels it.">
          <div className="delay-row">
            <Seg
              value={delays.includes(minutes) ? String(minutes) : ''}
              options={delays.map(d => ({ value: String(d), label: `${d} min` }))}
              onChange={v => save({ delay_s: Number(v) * 60 })}
              label="Delay"
              disabled={off}
            />
            <label className="delay-custom">
              <input
                type="number"
                min={0}
                inputMode="numeric"
                value={custom}
                disabled={off}
                aria-label="Delay in minutes"
                onChange={e => setCustom(e.target.value)}
                onBlur={() => Number(custom) !== minutes && custom !== '' && save({ delay_s: Math.max(0, Number(custom)) * 60 })}
              />
              <span>min</span>
            </label>
          </div>
        </Row>
        <label className="check check-hint">
          <input type="checkbox" checked={life.dry_run} onChange={e => save({ dry_run: e.target.checked })} />
          <span>
            Dry run
            <small>Only log and notify what would happen; the server is never stopped.</small>
          </span>
        </label>
        <p className="quiet-note">
          Acts {providerHow[life.provider]}. Settings from the server's environment (AITK_WHEN_DONE) apply again at its next start.
        </p>
        <ErrorNote error={error} />
      </div>
    </Section>
  );
}

function Row({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="life-row">
      <span className="life-row-label">{label}</span>
      <div className="life-row-body">
        {children}
        {hint && <small>{hint}</small>}
      </div>
    </div>
  );
}

function Seg<T extends string>({ value, options, onChange, label, disabled }: { value: T | ''; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string; disabled?: boolean }) {
  return (
    <div className={disabled ? 'seg is-disabled' : 'seg'} role="radiogroup" aria-label={label}>
      {options.map(o => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} className={value === o.value ? 'is-on' : ''} disabled={disabled} onClick={() => value !== o.value && onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ---- notifications ----

// How to get each kind of destination URL; picking one also sets the
// input's placeholder.
const ext = (url: string, label: string) => (
  <a className="link" href={url} target="_blank" rel="noreferrer">
    {label}
  </a>
);
const sinkGuides: { id: string; label: string; placeholder: string; steps: ReactNode[] }[] = [
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

function Notifications({ server }: { server: Server }) {
  const cfg = useLoad(server, s => get<NotifyConfig>(s, '/v1/notify'), []);
  const [url, setUrl] = useState('');
  const [guide, setGuide] = useState(sinkGuides[0]);
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [tests, setTests] = useState<Record<string, TestResult | 'busy'>>({});
  const [remove, setRemove] = useState<string | null>(null);
  const [links, setLinks] = useState<{ studio: string; pub: string; every: string } | null>(null);
  const c = cfg.data;
  useEffect(() => {
    if (c) setLinks({ studio: c.studio_url, pub: c.public_url, every: String(Math.round(c.sample_every_s / 60)) });
  }, [c?.studio_url, c?.public_url, c?.sample_every_s]); // eslint-disable-line react-hooks/exhaustive-deps

  const add = async () => {
    if (!url.trim()) return;
    setAdding(true);
    setAddError(null);
    try {
      await post(server, '/v1/notify/sinks', { url: url.trim() });
      setUrl('');
      cfg.reload();
    } catch (e) {
      setAddError(e instanceof ApiError && e.status === 409 ? 'That one is already set up.' : e instanceof Error ? e.message : String(e));
    } finally {
      setAdding(false);
    }
  };

  const test = async (sink?: string) => {
    const keys = sink ? [sink] : (c?.sinks ?? []).map(s => s.id);
    setTests(t => ({ ...t, ...Object.fromEntries(keys.map(k => [k, 'busy' as const])) }));
    try {
      const r = await post<{ items: TestResult[] }>(server, '/v1/notify/test', sink ? { sink } : {});
      setTests(t => ({ ...t, ...Object.fromEntries(r.items.map(i => [i.sink, i])) }));
    } catch (e) {
      setError(e);
      setTests(t => Object.fromEntries(Object.entries(t).filter(([k]) => !keys.includes(k))));
    }
  };

  const update = async (p: Partial<NotifyConfig>) => {
    setError(null);
    try {
      cfg.setData(await call<NotifyConfig>(server, 'PATCH', '/v1/notify', p));
    } catch (e) {
      setError(e);
    }
  };

  const enabled = new Set<NotifyKind>(c?.events ?? c?.kinds ?? []);
  const toggle = (k: NotifyKind, on: boolean) => {
    if (!c) return;
    const next = c.kinds.filter(x => (x === k ? on : enabled.has(x)));
    update({ events: next.length === c.kinds.length ? null : next });
  };

  return (
    <Section title="Notifications" description="Messages to your phone when runs start, sample, finish or fail, and before the server goes away.">
      <ErrorNote error={cfg.error || error} />
      {c && (
        <div className="notify">
          <div className="card notify-card">
            {c.sinks.length === 0 ? (
              <p className="quiet-note notify-empty">
                <BellRing aria-hidden />
                Nowhere to send yet. Add a URL below.
              </p>
            ) : (
              <ul className="sinks">
                {c.sinks.map(s => {
                  const t = tests[s.id];
                  return (
                    <li key={s.id} className="sink">
                      <div className="sink-main">
                        <p className="sink-title">
                          {sinkLabel[s.kind]}
                          {s.source === 'env' && (
                            <span className="badge tone-quiet" title="Set in the server's environment (AITK_NOTIFY)">
                              From env
                            </span>
                          )}
                        </p>
                        <p className="sink-target mono">{s.target}</p>
                        {t && t !== 'busy' && (
                          <p className={t.ok ? 'sink-test ok' : 'sink-test danger'}>
                            {t.ok ? <CircleCheck aria-hidden /> : <CircleX aria-hidden />}
                            {t.ok ? 'Sent' : t.detail}
                          </p>
                        )}
                      </div>
                      <div className="sink-actions">
                        <button className="btn btn-sm" onClick={() => test(s.id)} disabled={t === 'busy'}>
                          <Send />
                          {t === 'busy' ? 'Sending…' : 'Test'}
                        </button>
                        {s.source === 'api' && (
                          <button className="btn btn-sm btn-ghost btn-danger btn-icon" onClick={() => setRemove(s.id)} aria-label={`Remove ${sinkLabel[s.kind]}`} title="Remove">
                            <Trash2 />
                          </button>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
            <form
              className="sink-add"
              onSubmit={e => {
                e.preventDefault();
                add();
              }}
            >
              <label className="field">
                <span>Add a destination</span>
                <div className="sink-add-row">
                  <input
                    value={url}
                    onChange={e => setUrl(e.target.value)}
                    placeholder={guide.placeholder}
                    inputMode="url"
                    autoCapitalize="off"
                    autoCorrect="off"
                    spellCheck={false}
                    aria-invalid={!!addError}
                  />
                  <button className="btn btn-primary" type="submit" disabled={adding || !url.trim()}>
                    <Plus />
                    Add
                  </button>
                </div>
                {addError && <small className="field-error">{addError}</small>}
              </label>
              <div className="sink-guide">
                <span className="sink-guide-label">How to set one up</span>
                <Seg value={guide.id} options={sinkGuides.map(g => ({ value: g.id, label: g.label }))} onChange={id => setGuide(sinkGuides.find(g => g.id === id)!)} label="Destination type" />
                <ol className="sink-steps">
                  {guide.steps.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ol>
              </div>
              <p className="quiet-note">The URL holds its secret, so it is stored on the server and shown shortened.</p>
            </form>
            {c.sinks.length > 1 && (
              <button className="btn btn-sm notify-test-all" onClick={() => test()}>
                <Send />
                Test all
              </button>
            )}
          </div>

          <div className="card notify-card">
            <h3 className="notify-h">Tell me about</h3>
            <div className="kinds">
              {c.kinds.map(k => (
                <label key={k} className="check">
                  <input type="checkbox" checked={enabled.has(k)} onChange={e => toggle(k, e.target.checked)} />
                  {kindLabel[k] ?? k}
                </label>
              ))}
            </div>
          </div>

          {links && (
            <form
              className="card notify-card notify-links"
              onSubmit={e => {
                e.preventDefault();
                update({ studio_url: links.studio.trim(), public_url: links.pub.trim(), sample_every_s: Math.max(1, Number(links.every) || 15) * 60 });
              }}
            >
              <h3 className="notify-h">Links and pacing</h3>
              <label className="field">
                <span>Studio URL</span>
                <input value={links.studio} onChange={e => setLinks({ ...links, studio: e.target.value })} placeholder="https://studio.example.com/" inputMode="url" autoCapitalize="off" spellCheck={false} />
                <small>Messages open the run here.</small>
              </label>
              <label className="field">
                <span>Server URL</span>
                <input value={links.pub} onChange={e => setLinks({ ...links, pub: e.target.value })} placeholder="Detected from RunPod or Vast.ai" inputMode="url" autoCapitalize="off" spellCheck={false} />
                <small>Used when there is no studio URL, and for links to files.</small>
              </label>
              <label className="field">
                <span>Samples at most every</span>
                <div className="delay-custom">
                  <input type="number" min={1} inputMode="numeric" value={links.every} onChange={e => setLinks({ ...links, every: e.target.value })} />
                  <span>min</span>
                </div>
              </label>
              <div className="form-actions">
                <button className="btn btn-primary" type="submit">
                  Save
                </button>
              </div>
            </form>
          )}
        </div>
      )}
      {remove && (
        <Confirm
          title="Remove destination"
          confirm="Remove"
          onClose={() => setRemove(null)}
          onConfirm={async () => {
            try {
              await del(server, `/v1/notify/sinks/${encodeURIComponent(remove)}`);
              cfg.reload();
            } catch (e) {
              setError(e);
            }
            setRemove(null);
          }}
        >
          Stop sending notifications there?
        </Confirm>
      )}
    </Section>
  );
}

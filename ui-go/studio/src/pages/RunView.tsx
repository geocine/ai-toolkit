import { useCallback, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { LossChart } from '../components/LossChart';
import { ErrorNote, NeedServer, Progress, Section, StatePill, useLoad } from '../components/ui';
import {
  download,
  get,
  mediaUrl,
  post,
  streamEvents,
  type Checkpoint,
  type Items,
  type Job,
  type LossSeries,
  type Remote,
  type Sample,
  type SyncStatus,
  type Upload,
} from '../lib/api';
import { useServer } from '../lib/context';
import { ago, bytes, duration, pad, stepsPerSecond } from '../lib/format';
import { href } from '../lib/route';
import { totalSteps } from './Runs';

const ACTIVE = ['queued', 'starting', 'running', 'stopping'];

export function RunView({ id }: { id: string }) {
  const server = useServer();
  const job = useLoad(server, s => get<Job>(s, `/v1/jobs/${id}`), [id], 15000);
  const samples = useLoad(server, s => get<Items<Sample>>(s, `/v1/jobs/${id}/samples`), [id]);
  const checkpoints = useLoad(server, s => get<Items<Checkpoint>>(s, `/v1/jobs/${id}/checkpoints`), [id]);
  const loss = useLoad(server, s => get<LossSeries>(s, `/v1/jobs/${id}/loss?limit=20000`), [id]);
  const sync = useLoad(server, s => get<SyncStatus>(s, `/v1/jobs/${id}/sync`).catch(() => null), [id], 30000);
  const uploads = useLoad(server, s => get<Items<Upload>>(s, `/v1/uploads?job_id=${id}`), [id]);
  const remotes = useLoad(server, s => get<Items<Remote>>(s, '/v1/remotes').catch(() => ({ items: [] as Remote[] })), []);
  const [live, setLive] = useState<boolean | null>(null);
  const [log, setLog] = useState('');
  const [error, setError] = useState<unknown>(null);
  const [view, setView] = useState<Sample | null>(null);

  // live updates: progress and state into the job, files and loss as they land
  useEffect(() => {
    if (!server) return;
    return streamEvents(
      server,
      `job_id=${encodeURIComponent(id)}`,
      e => {
        switch (e.type) {
          case 'job.progress':
            job.setData(j => (j ? { ...j, step: e.data.step, total_steps: e.data.total_steps ?? j.total_steps, speed: e.data.speed, info: e.data.info } : j));
            break;
          case 'job.state':
          case 'job.attempt':
            job.reload();
            break;
          case 'job.checkpoint':
            checkpoints.reload();
            break;
          case 'job.sample':
            samples.reload();
            break;
          case 'job.loss':
            loss.setData(l => {
              if (!l) return l;
              const key = l.key;
              const add = (e.data.points as { step: number; key: string; value: number }[]).filter(p => p.key === key);
              return add.length ? { ...l, points: [...l.points, ...add] } : l;
            });
            break;
          case 'job.sync':
            sync.reload();
            break;
          case 'upload.state':
            uploads.reload();
            break;
          case 'job.log':
            setLog(prev => (prev + e.data.text).slice(-20000));
            break;
        }
      },
      on => {
        // the stream starts at "now": catch up on whatever landed before it
        // connected (or while it was down)
        if (on) {
          job.reload();
          samples.reload();
          checkpoints.reload();
          loss.reload();
        }
        setLive(on);
      },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [server, id]);

  useEffect(() => {
    if (!view) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setView(null);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [view]);

  // seed the log tail once
  useEffect(() => {
    if (!server) return;
    get<{ text: string }>(server, `/v1/jobs/${id}/log?offset=-8000`).then(r => setLog(r.text), () => undefined);
  }, [server, id]);

  const act = useCallback(
    async (action: 'start' | 'stop' | 'kill' | 'sample_now' | 'save_now') => {
      if (!server) return;
      setError(null);
      try {
        job.setData(await post<Job>(server, `/v1/jobs/${id}/${action}`));
      } catch (e) {
        setError(e);
      }
    },
    [server, id, job],
  );

  const byStep = useMemo(() => {
    const m = new Map<number, Sample[]>();
    for (const s of samples.data?.items ?? []) {
      const k = s.step ?? -1;
      m.set(k, [...(m.get(k) ?? []), s]);
    }
    return [...m.entries()].sort((a, b) => b[0] - a[0]);
  }, [samples.data]);

  if (!server) return <NeedServer />;
  const j = job.data;
  if (!j) return <ErrorNote error={job.error} />;

  const total = totalSteps(j);
  const sps = stepsPerSecond(j.speed);
  const eta = sps && total && j.state === 'running' ? (total - j.step) / sps : null;
  const p = j.config?.config?.process?.[0] ?? {};
  const ckpts = checkpoints.data?.items ?? [];

  return (
    <>
      <header className="run-head">
        <a className="kicker back" href={href({ page: 'runs' })}>
          ← Runs
        </a>
        <div className="run-title">
          <h1>{j.name}</h1>
          <StatePill state={j.state} />
          <span className={live ? 'live live-on' : 'live'} title={live ? 'live updates' : 'waiting for the event stream'}>
            {live === null ? 'connecting' : live ? 'live' : 'reconnecting'}
          </span>
        </div>
        <div className="run-counter">
          <span className="counter mono">{pad(j.step)}</span>
          <span className="dim mono">/ {total ?? '?'}</span>
          <div className="counter-side">
            <p>{j.info || '—'}</p>
            <p className="dim">
              {j.speed || 'no speed yet'}
              {eta !== null && <> · about {duration(eta)} left</>}
            </p>
          </div>
        </div>
        <Progress value={j.step} max={total} />
        <div className="run-actions">
          {ACTIVE.includes(j.state) ? (
            <>
              {j.state === 'running' && (
                <>
                  <button className="btn" onClick={() => act('sample_now')}>
                    Sample now
                  </button>
                  <button className="btn" onClick={() => act('save_now')}>
                    Save now
                  </button>
                </>
              )}
              {j.state === 'stopping' ? (
                <button className="btn btn-grease" onClick={() => confirm('Force-kill the trainer? Unsaved steps are lost.') && act('kill')}>
                  Force stop
                </button>
              ) : (
                <button className="btn btn-grease" onClick={() => act('stop')}>
                  Stop
                </button>
              )}
            </>
          ) : (
            j.state !== 'completed' && (
              <button className="btn btn-ink" onClick={() => act('start')}>
                {j.step > 0 ? `Resume from ${j.latest_checkpoint?.step ?? 'last checkpoint'}` : 'Start'}
              </button>
            )
          )}
        </div>
        <ErrorNote error={error} />
        <p className="dim run-facts">
          trigger <b>{p.trigger_word ?? '—'}</b> · {p.model?.name_or_path} · {p.network?.type} {p.network?.linear}
          {p.network?.lokr_factor !== undefined && ` / f${p.network.lokr_factor}`} · {p.train?.optimizer} · gpu {j.gpu_ids} ·{' '}
          {j.attempts} attempt{j.attempts === 1 ? '' : 's'}
          {sync.data?.remote && (
            <>
              {' '}
              · portable on <b>{sync.data.remote}</b>
              {sync.data.last_sync_at ? ` (synced ${ago(sync.data.last_sync_at)}${sync.data.step != null ? ` at step ${sync.data.step}` : ''})` : ''}
              {sync.data.last_error && <span className="warn"> · sync error: {sync.data.last_error}</span>}
            </>
          )}
        </p>
      </header>

      <Section kicker="Contact sheets" title="Samples">
        {!byStep.length && <p className="dim">Samples appear here as the trainer writes them.</p>}
        {byStep.map(([step, list]) => (
          <div key={step} className="strip">
            <span className="strip-label mono">{step >= 0 ? `STEP ${pad(step)}` : 'OTHER'}</span>
            <div className="strip-frames">
              {list.map(s => (
                <button key={s.name} className="strip-frame" onClick={() => setView(s)}>
                  <img src={mediaUrl(server, s.path, true)} alt={s.name} loading="lazy" />
                  <span className="mono">{s.index !== null ? `▸${s.index}` : ''}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </Section>

      <Section kicker="Loss" title={loss.data?.key ?? 'Loss'}>
        <LossChart points={loss.data?.points ?? []} checkpoints={ckpts.map(c => c.step).filter((s): s is number => s !== null)} totalSteps={total} />
      </Section>

      <Section kicker="Weights" title="Checkpoints">
        <Checkpoints job={j} items={ckpts} remotes={remotes.data?.items ?? []} uploads={uploads.data?.items ?? []} onUpload={uploads.reload} />
      </Section>

      <Section kicker="Trainer" title="Log">
        <pre className="log">{log || 'no output yet'}</pre>
      </Section>

      {/* portalled: the page's rise animation makes it a containing block for fixed children */}
      {view &&
        createPortal(
          <div className="lightbox" onClick={() => setView(null)} role="dialog">
            <img src={mediaUrl(server, view.path)} alt={view.name} />
            <p className="mono">
              step {view.step ?? '?'} · prompt {view.index ?? '?'} — {String(p.sample?.samples?.[view.index ?? -1]?.prompt ?? '').replaceAll('[trigger]', p.trigger_word ?? '[trigger]')}
            </p>
          </div>,
          document.body,
        )}
    </>
  );
}

function Checkpoints({ job, items, remotes, uploads, onUpload }: { job: Job; items: Checkpoint[]; remotes: Remote[]; uploads: Upload[]; onUpload: () => void }) {
  const server = useServer()!;
  const [remote, setRemote] = useState('');
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState('');
  if (!items.length) return <p className="dim">Checkpoints appear every {job.config?.config?.process?.[0]?.save?.save_every ?? '…'} steps.</p>;
  const uploadFor = (c: Checkpoint) => uploads.find(u => u.path.replace(/\\/g, '/').endsWith('/' + c.name));
  const dl = async (c: Checkpoint) => {
    setBusy(c.name);
    setError(null);
    try {
      await download(server, `/v1/files/content?root=training&path=${encodeURIComponent(job.name + '/' + c.name)}`, c.name);
    } catch (e) {
      setError(e);
    } finally {
      setBusy('');
    }
  };
  const push = async (c: Checkpoint) => {
    if (!remote) return;
    setError(null);
    try {
      await post(server, '/v1/uploads', { remote, job_id: job.id, checkpoint: c.name });
      onUpload();
    } catch (e) {
      setError(e);
    }
  };
  return (
    <>
      {remotes.length > 0 && (
        <label className="field field-inline">
          <span>Push to</span>
          <select value={remote} onChange={e => setRemote(e.target.value)}>
            <option value="">choose a remote</option>
            {remotes.map(r => (
              <option key={r.id} value={r.name}>
                {r.name} ({r.kind})
              </option>
            ))}
          </select>
        </label>
      )}
      <ErrorNote error={error} />
      <table className="ckpts">
        <thead>
          <tr>
            <th>Step</th>
            <th>File</th>
            <th>Size</th>
            <th>Saved</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {[...items].reverse().map(c => {
            const u = uploadFor(c);
            return (
              <tr key={c.name}>
                <td className="mono step-cell">{c.step ?? 'final'}</td>
                <td className="mono">{c.name}</td>
                <td className="mono">{bytes(c.size)}</td>
                <td className="dim">{ago(c.modified)}</td>
                <td className="ckpt-actions">
                  <button className="btn" disabled={busy === c.name} onClick={() => dl(c)}>
                    {busy === c.name ? 'Downloading…' : 'Download'}
                  </button>
                  {remotes.length > 0 && (
                    <button className="btn" disabled={!remote} onClick={() => push(c)}>
                      Push
                    </button>
                  )}
                  {u && (
                    <span className={`upload-state up-${u.state}`}>
                      {u.state === 'running' && u.bytes ? `${Math.round((u.bytes_done / u.bytes) * 100)}%` : u.state}
                      {u.error && ` · ${u.error}`}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

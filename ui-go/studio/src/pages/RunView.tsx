import { ArrowLeft, Camera, Download, FileArchive, OctagonX, Play, Save, Square, Upload as UploadIcon, X } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { ExportDialog } from '../components/ExportDialog';
import { LossChart } from '../components/LossChart';
import { ErrorNote, NeedServer, Progress, Section, StatePill, useLoad } from '../components/ui';
import {
  get,
  mediaUrl,
  post,
  signedDownload,
  streamEvents,
  trainingFilePath,
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
import { ago, bytes, duration, stepsPerSecond } from '../lib/format';
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
  const [exporting, setExporting] = useState(false);

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
              const pts = e.data.points as { step: number; key: string; value: number }[];
              // a run with no loss yet has no key: take it from the first points
              const keys = l?.keys?.length ? l.keys : [...new Set(pts.map(p => p.key))];
              const key = l?.key || (keys.includes('loss/loss') ? 'loss/loss' : (keys.find(k => k.startsWith('loss')) ?? keys[0]));
              const add = pts.filter(p => p.key === key);
              return add.length ? { keys, key, points: [...(l?.points ?? []), ...add] } : l;
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

  if (!server) return <NeedServer title="Run" />;
  const j = job.data;
  if (!j) return <ErrorNote error={job.error} />;

  const total = totalSteps(j);
  const sps = stepsPerSecond(j.speed);
  const eta = sps && total && j.state === 'running' ? (total - j.step) / sps : null;
  const p = j.config?.config?.process?.[0] ?? {};
  const ckpts = checkpoints.data?.items ?? [];

  const facts: { label: string; value: ReactNode; mono?: boolean; wide?: boolean }[] = [
    { label: 'Trigger', value: p.trigger_word ?? '–' },
    { label: 'Model', value: p.model?.name_or_path ?? '–', mono: true, wide: true },
    {
      label: 'Network',
      value: (
        <>
          {p.network?.type ?? '–'} {p.network?.linear}
          {p.network?.lokr_factor !== undefined && ` / f${p.network.lokr_factor}`}
        </>
      ),
    },
    { label: 'Optimizer', value: p.train?.optimizer ?? '–', mono: true },
    { label: 'GPU', value: j.gpu_ids || '–' },
    { label: 'Attempts', value: j.attempts },
  ];
  if (sync.data?.remote)
    facts.push({
      label: 'Portable on',
      wide: true,
      value: (
        <>
          {sync.data.remote}
          {sync.data.last_sync_at && (
            <span className="fact-note">
              {' '}
              · synced {ago(sync.data.last_sync_at)}
              {sync.data.step != null ? ` at step ${sync.data.step}` : ''}
            </span>
          )}
          {sync.data.last_error && <span className="warn"> · sync error: {sync.data.last_error}</span>}
        </>
      ),
    });

  const liveLabel = live === null ? 'Connecting' : live ? 'Live' : 'Reconnecting';
  const prompt = view ? String(p.sample?.samples?.[view.index ?? -1]?.prompt ?? '').replaceAll('[trigger]', p.trigger_word ?? '[trigger]') : '';

  return (
    <>
      <a className="back" href={href({ page: 'runs' })}>
        <ArrowLeft aria-hidden />
        Runs
      </a>
      <header className="run-head">
        <div className="run-title">
          <h1>{j.name}</h1>
          <StatePill state={j.state} />
          <span className={`live live-${live === null ? 'wait' : live ? 'on' : 'off'}`} title={live ? 'Live updates' : 'Waiting for the event stream'}>
            <span className="dot" aria-hidden />
            {liveLabel}
          </span>
          <div className="run-actions">
            <button className="btn" onClick={() => setExporting(true)}>
              <FileArchive />
              Export
            </button>
            {ACTIVE.includes(j.state) ? (
              <>
                {j.state === 'running' && (
                  <>
                    <button className="btn" onClick={() => act('sample_now')}>
                      <Camera />
                      Sample now
                    </button>
                    <button className="btn" onClick={() => act('save_now')}>
                      <Save />
                      Save now
                    </button>
                  </>
                )}
                {j.state === 'stopping' ? (
                  <button className="btn btn-danger" onClick={() => confirm('Force-kill the trainer? Unsaved steps are lost.') && act('kill')}>
                    <OctagonX />
                    Force stop
                  </button>
                ) : (
                  <button className="btn btn-danger" onClick={() => act('stop')}>
                    <Square />
                    Stop
                  </button>
                )}
              </>
            ) : (
              j.state !== 'completed' && (
                <button className="btn btn-primary" onClick={() => act('start')}>
                  <Play />
                  {j.step > 0 ? `Resume from ${j.latest_checkpoint?.step ?? 'last checkpoint'}` : 'Start'}
                </button>
              )
            )}
          </div>
        </div>
        <div className="run-progress">
          <Progress value={j.step} max={total} />
          <dl className="run-stats">
            <div>
              <dt className="visually-hidden">Step</dt>
              <dd>
                {j.step.toLocaleString()}
                <span className="of"> / {total?.toLocaleString() ?? '?'}</span>
              </dd>
            </div>
            <div>
              <dt className="visually-hidden">Speed</dt>
              <dd>{sps ? `${sps.toFixed(2)} it/s` : j.speed || '– it/s'}</dd>
            </div>
            <div>
              <dt>ETA</dt>
              <dd>{eta !== null ? duration(eta) : '–'}</dd>
            </div>
          </dl>
        </div>
        {j.info && <p className="run-info">{j.info}</p>}
        <ErrorNote error={error} />
        <dl className="card facts run-facts">
          {facts.map(f => (
            <div key={f.label} className={f.wide ? 'fact-wide' : undefined}>
              <dt>{f.label}</dt>
              <dd className={f.mono ? 'mono' : undefined} title={typeof f.value === 'string' ? f.value : undefined}>
                {f.value}
              </dd>
            </div>
          ))}
        </dl>
      </header>

      <Section title="Samples" description={byStep.length ? 'Grouped by step, newest first.' : undefined}>
        {!byStep.length && <p className="quiet-note">Samples appear here as the trainer writes them.</p>}
        {byStep.map(([step, list]) => (
          <div key={step} className="sample-set">
            <p className="sample-step">{step >= 0 ? `Step ${step.toLocaleString()}` : 'Other'}</p>
            <div className="sample-grid">
              {list.map(s => (
                <button
                  key={s.name}
                  className="sample"
                  onClick={() => setView(s)}
                  title={s.index !== null ? `Prompt ${s.index + 1}` : s.name}
                  aria-label={`Open ${s.name}`}
                >
                  <img src={mediaUrl(server, s.path, true)} alt={s.name} loading="lazy" />
                </button>
              ))}
            </div>
          </div>
        ))}
      </Section>

      <Section title="Loss" actions={loss.data?.key && <span className="mono chart-key">{loss.data.key}</span>}>
        <div className="card chart-card">
          <LossChart points={loss.data?.points ?? []} checkpoints={ckpts.map(c => c.step).filter((s): s is number => s !== null)} totalSteps={total} />
        </div>
      </Section>

      <Section title="Checkpoints">
        <Checkpoints job={j} items={ckpts} remotes={remotes.data?.items ?? []} uploads={uploads.data?.items ?? []} onUpload={uploads.reload} />
      </Section>

      <Section title="Log">
        <LogTail text={log} />
      </Section>

      {exporting && <ExportDialog server={server} job={j} onClose={() => setExporting(false)} />}

      {/* portalled so it sits above the shell's stacking contexts */}
      {view &&
        createPortal(
          <div className="lightbox" onClick={() => setView(null)} role="dialog" aria-modal="true" aria-label={view.name}>
            <figure className="lightbox-panel">
              <img src={mediaUrl(server, view.path)} alt={view.name} />
              <figcaption>
                <div>
                  <p className="lightbox-title">
                    Step {view.step?.toLocaleString() ?? '?'} · prompt {view.index !== null ? view.index + 1 : '?'}
                  </p>
                  {prompt && <p className="lightbox-prompt">{prompt}</p>}
                </div>
                <button className="btn btn-ghost btn-icon" onClick={() => setView(null)} aria-label="Close">
                  <X />
                </button>
              </figcaption>
            </figure>
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
  if (!items.length) return <p className="quiet-note">Checkpoints appear every {job.config?.config?.process?.[0]?.save?.save_every ?? '…'} steps.</p>;
  const uploadFor = (c: Checkpoint) => uploads.find(u => u.path.replace(/\\/g, '/').endsWith('/' + c.name));
  const dl = async (c: Checkpoint) => {
    setBusy(c.name);
    setError(null);
    try {
      // a signed link: the browser streams the file to disk, nothing in memory
      await signedDownload(server, trainingFilePath(job.name + '/' + c.name));
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
        <div className="ckpt-toolbar">
          <label className="field field-inline">
            <span>Push to</span>
            <select value={remote} onChange={e => setRemote(e.target.value)}>
              <option value="">Choose a remote</option>
              {remotes.map(r => (
                <option key={r.id} value={r.name}>
                  {r.name} ({r.kind})
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
      <ErrorNote error={error} />
      <div className="table-wrap">
        <table className="table ckpts">
          <thead>
            <tr>
              <th className="num">Step</th>
              <th>File</th>
              <th className="num">Size</th>
              <th>Saved</th>
              <th>
                <span className="visually-hidden">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {[...items].reverse().map(c => {
              const u = uploadFor(c);
              return (
                <tr key={c.name}>
                  <td className="num step-cell">{c.step?.toLocaleString() ?? 'Final'}</td>
                  <td className="mono file-cell" title={c.name}>
                    {c.name}
                  </td>
                  <td className="num">{bytes(c.size)}</td>
                  <td className="muted-cell">{ago(c.modified)}</td>
                  <td>
                    <div className="ckpt-actions">
                      {u && (
                        <StatePill
                          state={u.state}
                          label={(u.state === 'running' && u.bytes ? `${Math.round((u.bytes_done / u.bytes) * 100)}%` : u.state[0].toUpperCase() + u.state.slice(1)) + (u.error ? ` · ${u.error}` : '')}
                        />
                      )}
                      <button className="btn btn-sm" disabled={busy === c.name} onClick={() => dl(c)}>
                        <Download />
                        {busy === c.name ? 'Preparing…' : 'Download'}
                      </button>
                      {remotes.length > 0 && (
                        <button className="btn btn-sm" disabled={!remote} onClick={() => push(c)}>
                          <UploadIcon />
                          Push
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

// LogTail follows the end of the log unless the reader has scrolled up.
function LogTail({ text }: { text: string }) {
  const ref = useRef<HTMLPreElement>(null);
  const stick = useRef(true);
  useLayoutEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [text]);
  return (
    <pre
      ref={ref}
      className="log"
      onScroll={e => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
    >
      {text ? collapseCR(text) : 'No output yet.'}
    </pre>
  );
}

// collapseCR keeps what a terminal would show: a progress bar rewrites its
// line with \r (and ANSI cursor moves), so only the text after the last \r of
// each line remains.
function collapseCR(text: string): string {
  return text
    .replace(/\x1b\[[0-9;?]*[A-Za-z]/g, '')
    .split('\n')
    .map(line => {
      const parts = line.split('\r').filter(Boolean);
      return parts.length ? parts[parts.length - 1] : '';
    })
    .join('\n');
}

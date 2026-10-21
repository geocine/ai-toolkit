import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Camera, Download, Hash, Octagon, Play, RefreshCw, Save, Square, Trash2, UploadCloud } from 'lucide-react';
import { api, fileUrl, fmtAgo, fmtBytes, fmtTime, imgUrl, qs, streamEvents, type AitkEvent } from '../api';
import { LossChart } from '../LossChart';
import { go, href } from '../router';
import type { Attempt, Checkpoint, Items, Job, LogChunk, LossSeries, Remote, Sample, SyncStatus } from '../types';
import { Bar, CopyButton, Empty, Field, Loading, Modal, Mono, Section, StateBadge, Spinner, usePoll, useConfirm, useToast } from '../ui';
import { stepsOf } from './Jobs';

const TABS = ['overview', 'log', 'loss', 'samples', 'checkpoints', 'attempts', 'config'] as const;
type Tab = (typeof TABS)[number];
const LIVE_TYPES = ['job.state', 'job.progress', 'job.checkpoint', 'job.sample', 'job.attempt', 'job.sync', 'job.loss', 'job.log', 'job.deleted'];
const LOG_KEEP = 512 * 1024;

export function JobDetail({ id }: { id: string }) {
  const toast = useToast();
  const confirm = useConfirm();
  const [job, setJob] = useState<Job | null>(null);
  const [missing, setMissing] = useState(false);
  const [tab, setTab] = useState<Tab>('overview');
  const [live, setLive] = useState(false);
  const [busy, setBusy] = useState('');
  // bumped by events so tabs refetch only what changed
  const [ver, setVer] = useState({ checkpoints: 0, samples: 0, attempts: 0, sync: 0 });
  const [log, setLog] = useState({ text: '', end: 0, loaded: false });
  const [lossLive, setLossLive] = useState<{ step: number; key: string; value: number }[]>([]);

  const load = useCallback(async () => {
    try {
      setJob(await api.get<Job>(`/v1/jobs/${encodeURIComponent(id)}`));
    } catch (e: any) {
      if (e?.status === 404) setMissing(true);
      else throw e;
    }
  }, [id]);

  usePoll(load, 10000, [id]);

  // live updates
  const reloadTimer = useRef<number | undefined>(undefined);
  useEffect(() => {
    const ctrl = new AbortController();
    const reload = () => {
      window.clearTimeout(reloadTimer.current);
      reloadTimer.current = window.setTimeout(() => load().catch(() => {}), 250);
    };
    // the tabs load history themselves, so only events from the moment the
    // page opened matter (after=now skips the stored ones)
    const openedAt = Date.now() - 2000;
    streamEvents(
      { job_id: id, types: LIVE_TYPES, after: 'now' },
      (e: AitkEvent) => {
        if (e.id && new Date(e.at).getTime() < openedAt) return;
        switch (e.type) {
          case 'job.log':
            setLog(l => {
              if (!l.loaded) return l;
              const start = e.data.offset as number;
              const text = e.data.text as string;
              if (start + text.length <= l.end) return l; // already have it
              const fresh = start < l.end ? text.slice(l.end - start) : text;
              const all = l.text + fresh;
              return { ...l, text: all.length > LOG_KEEP ? all.slice(-LOG_KEEP) : all, end: start + text.length };
            });
            return;
          case 'job.loss':
            setLossLive(p => [...p, ...(e.data.points ?? [])]);
            return;
          case 'job.progress':
            setJob(j => (j ? { ...j, step: e.data.step, total_steps: e.data.total_steps, speed: e.data.speed, info: e.data.info } : j));
            return;
          case 'job.checkpoint':
            setVer(v => ({ ...v, checkpoints: v.checkpoints + 1 }));
            break;
          case 'job.sample':
            setVer(v => ({ ...v, samples: v.samples + 1 }));
            return;
          case 'job.attempt':
            setVer(v => ({ ...v, attempts: v.attempts + 1 }));
            break;
          case 'job.sync':
            setVer(v => ({ ...v, sync: v.sync + 1 }));
            return;
          case 'job.deleted':
            setMissing(true);
            return;
        }
        reload();
      },
      ctrl.signal,
      setLive,
    );
    return () => {
      ctrl.abort();
      window.clearTimeout(reloadTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  if (missing) {
    return (
      <Empty title="Job not found">
        It may have been deleted. <a href={href('jobs')}>Back to jobs</a>
      </Empty>
    );
  }
  if (!job) return <Loading what="job" />;

  const total = job.total_steps ?? stepsOf(job);
  const active = ['queued', 'starting', 'running', 'stopping'].includes(job.state);
  const running = ['starting', 'running'].includes(job.state);

  const act = async (what: string, path: string, label: string) => {
    setBusy(what);
    try {
      setJob(await api.post<Job>(`/v1/jobs/${encodeURIComponent(job.id)}/${path}`));
      toast.success(label);
    } catch (e) {
      toast.error(e);
    } finally {
      setBusy('');
    }
  };

  const kill = async () => {
    if (await confirm({ title: 'Force stop', body: 'Kill the trainer process now? Progress since the last checkpoint is lost.', confirm: 'Kill', danger: true })) {
      act('kill', 'kill', 'Job killed');
    }
  };

  const del = async () => {
    if (!(await confirm({ title: 'Delete job', body: <>Delete <b>{job.name}</b> and its output folder (checkpoints, samples, logs)?</>, confirm: 'Delete', danger: true }))) return;
    try {
      await api.del(`/v1/jobs/${encodeURIComponent(job.id)}?files=true`);
      toast.success('Job deleted');
      go('jobs');
    } catch (e) {
      toast.error(e);
    }
  };

  return (
    <>
      <div className="crumbs">
        <a href={href('jobs')}>Jobs</a> / {job.type}
      </div>
      <div className="page-head">
        <div className="grow">
          <div className="row-wrap">
            <h1 className="page-title">{job.name}</h1>
            <StateBadge state={job.state} />
            <span className={live ? 'live on' : 'live'} title={live ? 'Receiving live events' : 'Reconnecting; polling meanwhile'}>
              <span className="live-dot" />
              {live ? 'live' : 'offline'}
            </span>
          </div>
          <div className="page-sub ellipsis" title={job.info}>
            {job.info || '—'}
          </div>
        </div>
        <div className="row-wrap">
          {!active && (
            <button className="btn btn-primary" disabled={!!busy} onClick={() => act('start', 'start', 'Job queued')}>
              {busy === 'start' ? <Spinner size={12} /> : <Play size={12} />} {job.step > 0 ? 'Resume' : 'Start'}
            </button>
          )}
          {running && (
            <>
              <button className="btn" disabled={!!busy} onClick={() => act('save', 'save_now', 'Will save at the next step')}>
                <Save size={12} /> Save now
              </button>
              <button className="btn" disabled={!!busy} onClick={() => act('sample', 'sample_now', 'Will sample at the next step')}>
                <Camera size={12} /> Sample now
              </button>
            </>
          )}
          {active && (
            <button className="btn" disabled={!!busy || job.state === 'stopping'} onClick={() => act('stop', 'stop', 'Stopping')}>
              <Square size={12} /> Stop
            </button>
          )}
          {(running || job.state === 'stopping') && (
            <button className="btn btn-ghost-danger" disabled={!!busy} onClick={kill}>
              <Octagon size={12} /> Kill
            </button>
          )}
          {!active && (
            <button className="btn btn-ghost-danger" onClick={del}>
              <Trash2 size={12} /> Delete
            </button>
          )}
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
          <span className="mono">
            step {job.step}
            {total ? ` / ${total}` : ''}
            {total ? <span className="muted"> · {((job.step / total) * 100).toFixed(1)}%</span> : null}
          </span>
          <span className="mono muted small">{job.speed}</span>
        </div>
        <Bar value={job.step} max={total ?? 0} />
      </div>

      <div className="tabs" role="tablist">
        {TABS.map(t => (
          <button key={t} role="tab" className={tab === t ? 'tab active' : 'tab'} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab job={job} onChange={setJob} syncVer={ver.sync} />}
      {tab === 'log' && <LogTab job={job} log={log} setLog={setLog} />}
      {tab === 'loss' && <LossTab job={job} live={lossLive} />}
      {tab === 'samples' && <SamplesTab job={job} ver={ver.samples} />}
      {tab === 'checkpoints' && <CheckpointsTab job={job} ver={ver.checkpoints} />}
      {tab === 'attempts' && <AttemptsTab job={job} ver={ver.attempts} />}
      {tab === 'config' && <ConfigTab job={job} />}
    </>
  );
}

// ---------------------------------------------------------------- overview

function OverviewTab({ job, onChange, syncVer }: { job: Job; onChange: (j: Job) => void; syncVer: number }) {
  const toast = useToast();
  const [remotes, setRemotes] = useState<Remote[]>([]);
  const [sync, setSync] = useState<SyncStatus | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [policy, setPolicy] = useState(() => policyForm(job));
  const [saving, setSaving] = useState(false);

  useEffect(() => setPolicy(policyForm(job)), [job.id, JSON.stringify(job.policy)]);
  useEffect(() => {
    api.get<Items<Remote>>('/v1/remotes').then(r => setRemotes(r.items), () => {});
  }, []);
  const refreshSync = usePoll(async () => setSync(await api.get<SyncStatus>(`/v1/jobs/${encodeURIComponent(job.id)}/sync`)), 15000, [job.id, syncVer]);

  const remoteName = (id: string | null | undefined) => remotes.find(r => r.id === id)?.name ?? id ?? '—';
  const busy = ['queued', 'starting', 'running', 'stopping'].includes(job.state);

  const savePolicy = async () => {
    setSaving(true);
    try {
      const updated = await api.patch<Job>(`/v1/jobs/${encodeURIComponent(job.id)}`, {
        policy: {
          auto_resume: policy.auto_resume,
          max_auto_resumes: Number(policy.max_auto_resumes),
          sync_remote: policy.sync_remote, // "" = explicitly none
          auto_upload_remote: policy.auto_upload_remote,
        },
      });
      onChange(updated);
      refreshSync();
      toast.success('Policy saved');
    } catch (e) {
      toast.error(e);
    } finally {
      setSaving(false);
    }
  };

  const forceSync = async () => {
    setSyncing(true);
    try {
      setSync(await api.post<SyncStatus>(`/v1/jobs/${encodeURIComponent(job.id)}/sync`));
      toast.success('Synced');
    } catch (e) {
      toast.error(e);
    } finally {
      setSyncing(false);
    }
  };

  const a = job.last_attempt;
  return (
    <div className="grid grid-2">
      <div className="card">
        <dl className="kv">
          <dt>ID</dt>
          <dd className="row">
            <Mono>{job.id}</Mono> <CopyButton text={job.id} />
          </dd>
          <dt>Type</dt>
          <dd>{job.type}</dd>
          <dt>GPUs</dt>
          <dd className="mono">{job.gpu_ids}</dd>
          <dt>Status</dt>
          <dd className="mono">
            {job.status} {job.pid ? <span className="muted">· pid {job.pid}</span> : null}
          </dd>
          <dt>Created</dt>
          <dd title={fmtTime(job.created_at)}>{fmtAgo(job.created_at)}</dd>
          <dt>Updated</dt>
          <dd title={fmtTime(job.updated_at)}>{fmtAgo(job.updated_at)}</dd>
          <dt>Attempts</dt>
          <dd>
            {job.attempts}
            {a ? (
              <span className="muted">
                {' '}
                · last {a.end_state ?? 'running'} from step {a.start_step}
                {a.auto ? ' (auto)' : ''}
              </span>
            ) : null}
          </dd>
          <dt>Checkpoint</dt>
          <dd>
            {job.latest_checkpoint ? (
              <a className="mono" href={fileUrl(job.latest_checkpoint.path)} download>
                {job.latest_checkpoint.name}
              </a>
            ) : (
              <span className="dim">none yet</span>
            )}
          </dd>
          {job.ref && (
            <>
              <dt>Ref</dt>
              <dd className="mono">{job.ref}</dd>
            </>
          )}
        </dl>
      </div>

      <div className="card">
        <div className="section-head">
          <h2>Policy</h2>
        </div>
        <div className="form-grid">
          <label className="check">
            <input type="checkbox" checked={policy.auto_resume} onChange={e => setPolicy({ ...policy, auto_resume: e.target.checked })} />
            Auto-resume when interrupted
          </label>
          <Field label="Max auto-resumes">
            <input type="number" min={0} value={policy.max_auto_resumes} onChange={e => setPolicy({ ...policy, max_auto_resumes: e.target.value })} />
          </Field>
          <Field label="Sync remote" hint="Keeps the job portable to another server">
            <select value={policy.sync_remote} onChange={e => setPolicy({ ...policy, sync_remote: e.target.value })}>
              <option value="">none</option>
              {remotes.map(r => (
                <option key={r.id} value={r.id}>
                  {r.name} ({r.kind})
                </option>
              ))}
            </select>
          </Field>
          <Field label="Auto-upload remote" hint="Every new checkpoint is uploaded">
            <select value={policy.auto_upload_remote} onChange={e => setPolicy({ ...policy, auto_upload_remote: e.target.value })}>
              <option value="">none</option>
              {remotes.map(r => (
                <option key={r.id} value={r.id}>
                  {r.name} ({r.kind})
                </option>
              ))}
            </select>
          </Field>
        </div>
        <div className="form-actions">
          {busy && <span className="muted small">Policy can be changed while the job is not queued or running.</span>}
          <button className="btn btn-primary" onClick={savePolicy} disabled={saving || busy}>
            {saving ? <Spinner size={12} /> : <Save size={12} />} Save policy
          </button>
        </div>
      </div>

      <div className="card">
        <div className="section-head">
          <h2>Portable sync</h2>
          {sync?.remote && (
            <button className="btn btn-sm" onClick={forceSync} disabled={syncing}>
              {syncing ? <Spinner size={12} /> : <UploadCloud size={12} />} Sync now
            </button>
          )}
        </div>
        {!sync ? (
          <Loading />
        ) : !sync.remote ? (
          <div className="muted small">Not synced. Set a sync remote in the policy to make this job resumable on another server.</div>
        ) : (
          <dl className="kv">
            <dt>Remote</dt>
            <dd>{remoteName(sync.remote)}</dd>
            <dt>Synced step</dt>
            <dd className="mono">{sync.step ?? '—'}</dd>
            <dt>Last sync</dt>
            <dd title={fmtTime(sync.last_sync_at)}>{fmtAgo(sync.last_sync_at)}</dd>
            <dt>Lease</dt>
            <dd>
              {sync.lease ? (
                <>
                  <Mono>{sync.lease.node_label}</Mono> <span className="muted">until {fmtTime(sync.lease.expires_at)}</span>
                </>
              ) : (
                <span className="muted">released</span>
              )}
            </dd>
            {sync.last_error && (
              <>
                <dt>Error</dt>
                <dd className="danger-text">{sync.last_error}</dd>
              </>
            )}
          </dl>
        )}
      </div>
    </div>
  );
}

const policyForm = (job: Job) => ({
  auto_resume: !!job.policy?.auto_resume,
  max_auto_resumes: String(job.policy?.max_auto_resumes ?? 3),
  sync_remote: job.policy?.sync_remote ?? '',
  auto_upload_remote: job.policy?.auto_upload_remote ?? '',
});

// ---------------------------------------------------------------- log

type LogState = { text: string; end: number; loaded: boolean };

function LogTab({ job, log, setLog }: { job: Job; log: LogState; setLog: (fn: (l: LogState) => LogState) => void }) {
  const ref = useRef<HTMLPreElement>(null);
  const [follow, setFollow] = useState(true);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    try {
      const chunk = await api.get<LogChunk>(`/v1/jobs/${encodeURIComponent(job.id)}/log${qs({ offset: -262144, limit: 262144 })}`);
      setLog(() => ({ text: chunk.text, end: chunk.next_offset, loaded: true }));
      setError('');
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  }, [job.id, setLog]);

  useEffect(() => {
    if (!log.loaded) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id]);

  useEffect(() => {
    if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [log.text, follow]);

  return (
    <>
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
        <label className="check small">
          <input type="checkbox" checked={follow} onChange={e => setFollow(e.target.checked)} /> Follow
        </label>
        <button className="btn btn-sm" onClick={load}>
          <RefreshCw size={12} /> Reload
        </button>
      </div>
      {error && <div className="danger-text small">{error}</div>}
      {!log.loaded ? (
        <Loading what="log" />
      ) : (
        <pre className="log" ref={ref} onScroll={e => {
          const el = e.currentTarget;
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          if (atBottom !== follow) setFollow(atBottom);
        }}>
          {log.text ? cleanLog(log.text) : <span className="dim">The log is empty.</span>}
        </pre>
      )}
    </>
  );
}

/** tqdm redraws lines with \r; show only the latest state of each line. */
const cleanLog = (t: string) =>
  t
    .split('\n')
    .map(line => {
      const parts = line.split('\r').filter(p => p !== '');
      return parts.length ? parts[parts.length - 1] : '';
    })
    .join('\n');

// ---------------------------------------------------------------- loss

function LossTab({ job, live }: { job: Job; live: { step: number; key: string; value: number }[] }) {
  const [series, setSeries] = useState<LossSeries | null>(null);
  const [key, setKey] = useState('');
  const [smoothing, setSmoothing] = useState(0.9);
  const [log, setLogScale] = useState(false);

  usePoll(
    async () => {
      setSeries(await api.get<LossSeries>(`/v1/jobs/${encodeURIComponent(job.id)}/loss${qs({ key, limit: 20000 })}`));
    },
    30000,
    [job.id, key],
  );

  const points = useMemo(() => {
    if (!series) return [];
    const k = series.key;
    const base = series.points.map(p => ({ step: p.step, value: p.value }));
    const lastStep = base.length ? base[base.length - 1].step : -1;
    const extra = live.filter(p => p.key === k && p.step > lastStep).map(p => ({ step: p.step, value: p.value }));
    return [...base, ...extra];
  }, [series, live]);

  if (!series) return <Loading what="loss" />;
  if (series.keys.length === 0) return <Empty title="No metrics yet">The trainer writes loss_log.db once training steps start.</Empty>;
  const last = points[points.length - 1];
  return (
    <div className="card">
      <div className="row-wrap" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="row-wrap">
          <select value={series.key} onChange={e => setKey(e.target.value)} style={{ width: 200 }}>
            {series.keys.map(k => (
              <option key={k}>{k}</option>
            ))}
          </select>
          <label className="check small">
            <input type="checkbox" checked={log} onChange={e => setLogScale(e.target.checked)} /> Log scale
          </label>
          <label className="check small">
            Smoothing
            <input type="range" min={0} max={0.99} step={0.01} value={smoothing} onChange={e => setSmoothing(Number(e.target.value))} />
            <span className="mono">{smoothing.toFixed(2)}</span>
          </label>
        </div>
        {last && (
          <span className="mono small muted">
            step {last.step} · {last.value.toPrecision(4)} · {points.length} points
          </span>
        )}
      </div>
      <LossChart points={points} smoothing={smoothing} log={log} />
    </div>
  );
}

// ---------------------------------------------------------------- samples

function SamplesTab({ job, ver }: { job: Job; ver: number }) {
  const [items, setItems] = useState<Sample[] | null>(null);
  const [open, setOpen] = useState<Sample | null>(null);
  usePoll(async () => setItems((await api.get<Items<Sample>>(`/v1/jobs/${encodeURIComponent(job.id)}/samples`)).items), 30000, [job.id, ver]);

  if (!items) return <Loading what="samples" />;
  if (items.length === 0) return <Empty title="No samples yet">Samples appear at every sample_every steps.</Empty>;
  const byStep = [...items].reverse();
  const isVideo = (n: string) => /\.(mp4|webm)$/i.test(n);
  return (
    <>
      <div className="grid grid-media">
        {byStep.map(s => (
          <button key={s.name} className="thumb" style={{ padding: 0, cursor: 'zoom-in', font: 'inherit', color: 'inherit', textAlign: 'left' }} onClick={() => setOpen(s)}>
            {isVideo(s.name) ? <video src={imgUrl(s.path)} muted /> : <img src={imgUrl(s.path, true)} loading="lazy" alt={s.name} />}
            <div className="thumb-meta">
              <span className="mono small">
                step {s.step ?? '—'} · #{s.index ?? '—'}
              </span>
            </div>
          </button>
        ))}
      </div>
      {open && (
        <Modal title={`Step ${open.step ?? '—'} · sample ${open.index ?? '—'}`} onClose={() => setOpen(null)} wide>
          <div className="modal-body">
            {isVideo(open.name) ? <video className="lightbox" src={imgUrl(open.path)} controls autoPlay /> : <img className="lightbox" src={imgUrl(open.path)} alt={open.name} />}
            <div className="mono small muted" style={{ marginTop: 8 }}>
              {open.name}
            </div>
          </div>
        </Modal>
      )}
    </>
  );
}

// ---------------------------------------------------------------- checkpoints

function CheckpointsTab({ job, ver }: { job: Job; ver: number }) {
  const toast = useToast();
  const [items, setItems] = useState<Checkpoint[] | null>(null);
  const [hashing, setHashing] = useState(false);
  const [remotes, setRemotes] = useState<Remote[]>([]);
  const load = async (hash = false) => {
    setItems((await api.get<Items<Checkpoint>>(`/v1/jobs/${encodeURIComponent(job.id)}/checkpoints${hash ? '?hash=true' : ''}`)).items);
  };
  usePoll(() => load(), 30000, [job.id, ver]);
  useEffect(() => {
    api.get<Items<Remote>>('/v1/remotes').then(r => setRemotes(r.items), () => {});
  }, []);

  const hash = async () => {
    setHashing(true);
    try {
      await load(true);
    } catch (e) {
      toast.error(e);
    } finally {
      setHashing(false);
    }
  };
  const uploadTo = async (c: Checkpoint, remote: string) => {
    try {
      await api.post('/v1/uploads', { remote, job_id: job.id, checkpoint: c.name });
      toast.success(`Uploading ${c.name}`);
    } catch (e) {
      toast.error(e);
    }
  };

  if (!items) return <Loading what="checkpoints" />;
  if (items.length === 0) return <Empty title="No checkpoints yet">They are saved every save_every steps and at the end.</Empty>;
  return (
    <Section
      title={`${items.length} checkpoints`}
      actions={
        <button className="btn btn-sm" onClick={hash} disabled={hashing}>
          {hashing ? <Spinner size={12} /> : <Hash size={12} />} Compute sha256
        </button>
      }
    >
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>File</th>
              <th>Step</th>
              <th>Size</th>
              <th>Saved</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {[...items].reverse().map(c => (
              <tr key={c.name}>
                <td>
                  <div className="mono">{c.name}</div>
                  {c.sha256 && <div className="mono dim small ellipsis" style={{ maxWidth: 420 }}>{c.sha256}</div>}
                </td>
                <td className="mono">{c.step ?? 'final'}</td>
                <td className="mono nowrap">{fmtBytes(c.size)}</td>
                <td className="small muted nowrap" title={fmtTime(c.modified)}>
                  {fmtAgo(c.modified)}
                </td>
                <td className="actions">
                  <div className="row" style={{ justifyContent: 'flex-end' }}>
                    {remotes.length > 0 && (
                      <select
                        value=""
                        onChange={e => e.target.value && uploadTo(c, e.target.value)}
                        style={{ width: 130, padding: '4px 6px', fontSize: 11 }}
                        aria-label="Upload to remote"
                      >
                        <option value="">Upload to…</option>
                        {remotes.map(r => (
                          <option key={r.id} value={r.id}>
                            {r.name}
                          </option>
                        ))}
                      </select>
                    )}
                    <a className="btn btn-sm" href={fileUrl(c.path)} download>
                      <Download size={12} /> Download
                    </a>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------- attempts

function AttemptsTab({ job, ver }: { job: Job; ver: number }) {
  const [items, setItems] = useState<Attempt[] | null>(null);
  usePoll(async () => setItems((await api.get<Items<Attempt>>(`/v1/jobs/${encodeURIComponent(job.id)}/attempts`)).items), 15000, [job.id, ver]);
  if (!items) return <Loading what="attempts" />;
  if (items.length === 0) return <Empty title="Never started">Each launch of the job is recorded here.</Empty>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Started</th>
            <th>Steps</th>
            <th>Ended</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {[...items].reverse().map(a => (
            <tr key={a.n}>
              <td className="mono">
                {a.n}
                {a.auto && <span className="badge" style={{ marginLeft: 6 }}>auto</span>}
              </td>
              <td className="small nowrap">{fmtTime(a.started_at)}</td>
              <td className="mono">
                {a.start_step} → {a.end_step ?? '…'}
              </td>
              <td>{a.end_state ? <StateBadge state={a.end_state} /> : <StateBadge state="running" />}</td>
              <td className="small muted" style={{ maxWidth: 420 }}>
                {a.end_reason ?? ''}
                {a.pid ? <span className="dim"> · pid {a.pid}</span> : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- config

function ConfigTab({ job }: { job: Job }) {
  const text = useMemo(() => JSON.stringify(job.config, null, 2), [job.config]);
  return (
    <>
      <div className="row" style={{ justifyContent: 'flex-end', marginBottom: 8 }}>
        <CopyButton text={text} />
      </div>
      <pre className="json">{text}</pre>
    </>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Camera, ChevronRight, Download, FileArchive, FileX, Hash, History, Image as ImageIcon, Octagon, Package, Play, RefreshCw, Save, Square, Trash2, UploadCloud } from 'lucide-react';
import { api, fmtAgo, fmtBytes, fmtTime, imgUrl, plural, qs, signedDownload, streamEvents, trainingFilePath, type AitkEvent } from '../api';
import { ExportDialog } from '../bundleUi';
import { LossChart } from '../LossChart';
import { go, href } from '../router';
import type { Attempt, Checkpoint, Items, Job, LogChunk, LossSeries, Remote, Sample, SyncStatus } from '../types';
import { Bar, Callout, CopyButton, Empty, Field, Loading, Modal, Mono, StateBadge, Spinner, sentence, usePoll, useConfirm, useToast } from '../ui';
import { stepsOf } from './Jobs';

const TABS = ['overview', 'log', 'loss', 'samples', 'checkpoints', 'attempts', 'config'] as const;
type Tab = (typeof TABS)[number];
const LIVE_TYPES = ['job.state', 'job.progress', 'job.checkpoint', 'job.sample', 'job.attempt', 'job.sync', 'job.loss', 'job.log', 'job.deleted'];
const LOG_KEEP = 512 * 1024;

/** Time left from the trainer's speed text ("2.2 iter/sec" or "1.4 sec/iter"). */
const etaOf = (speed: string | undefined, left: number): string | null => {
  const m = speed?.match(/([\d.]+)\s*(iter\/sec|it\/s|sec\/iter|s\/it)/i);
  if (!m || left <= 0) return null;
  const v = Number(m[1]);
  if (!v) return null;
  const secs = /^(iter|it)/i.test(m[2]) ? left / v : left * v;
  const h = Math.floor(secs / 3600);
  const min = Math.floor((secs % 3600) / 60);
  return h ? `${h}h ${min}m` : min ? `${min}m` : `${Math.max(1, Math.round(secs))}s`;
};

export function JobDetail({ id }: { id: string }) {
  const toast = useToast();
  const confirm = useConfirm();
  const [job, setJob] = useState<Job | null>(null);
  const [missing, setMissing] = useState(false);
  const [tab, setTab] = useState<Tab>('overview');
  const [live, setLive] = useState(false);
  const [busy, setBusy] = useState('');
  const [exporting, setExporting] = useState(false);
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
              if (start === 0 && l.end > 0) {
                // a new attempt rotated log.txt and writes a fresh one from 0
                const all = l.text + '\n' + text;
                return { ...l, text: all.length > LOG_KEEP ? all.slice(-LOG_KEEP) : all, end: text.length };
              }
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
      <Empty
        icon={<FileX size={18} />}
        title="Job not found"
        action={
          <a className="btn btn-sm" href={href('jobs')}>
            Back to jobs
          </a>
        }
      >
        It may have been deleted.
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

  const eta = running && total ? etaOf(job.speed, total - job.step) : null;

  return (
    <>
      <div className="crumbs">
        <a href={href('jobs')}>Jobs</a>
        <ChevronRight size={14} />
        <span>{sentence(job.type)}</span>
      </div>
      <div className="page-head">
        <div>
          <div className="title-row">
            <h1 className="page-title">{job.name}</h1>
            <StateBadge state={job.state} badge />
            <span className={live ? 'live on' : 'live'} title={live ? 'Receiving live events' : 'Reconnecting; polling meanwhile'}>
              <span className="dot" />
              {live ? 'Live' : 'Offline'}
            </span>
          </div>
          <div className="page-sub ellipsis" title={job.info}>
            {job.info || '—'}
          </div>
        </div>
        <div className="page-actions">
          {running && (
            <>
              <button className="btn" disabled={!!busy} onClick={() => act('save', 'save_now', 'Will save at the next step')}>
                <Save size={14} /> Save now
              </button>
              <button className="btn" disabled={!!busy} onClick={() => act('sample', 'sample_now', 'Will sample at the next step')}>
                <Camera size={14} /> Sample now
              </button>
            </>
          )}
          {active && (
            <button className="btn" disabled={!!busy || job.state === 'stopping'} onClick={() => act('stop', 'stop', 'Stopping')}>
              <Square size={14} /> Stop
            </button>
          )}
          {(running || job.state === 'stopping') && (
            <button className="btn btn-destructive" disabled={!!busy} onClick={kill}>
              <Octagon size={14} /> Kill
            </button>
          )}
          <button className="btn" onClick={() => setExporting(true)}>
            <FileArchive size={14} /> Export
          </button>
          {!active && (
            <button className="btn btn-destructive" onClick={del}>
              <Trash2 size={14} /> Delete
            </button>
          )}
          {!active && (
            <button className="btn btn-primary" disabled={!!busy} onClick={() => act('start', 'start', 'Job queued')}>
              {busy === 'start' ? <Spinner size={14} /> : <Play size={14} />} {job.step > 0 ? 'Resume' : 'Start'}
            </button>
          )}
        </div>
      </div>

      <div className="run-progress">
        <Bar value={job.step} max={total ?? 0} />
        <div className="run-stats">
          <span>
            Step <b>{job.step}</b>
            {total ? <> / {total}</> : null}
            {total ? <span className="dim"> · {((job.step / total) * 100).toFixed(1)}%</span> : null}
          </span>
          {job.speed && <span>{job.speed}</span>}
          {eta && <span>ETA {eta}</span>}
        </div>
      </div>

      <div className="tabs" role="tablist">
        {TABS.map(t => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'tab active' : 'tab'} onClick={() => setTab(t)}>
            {sentence(t)}
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
      {exporting && <ExportDialog job={job} onClose={() => setExporting(false)} />}
    </>
  );
}

/** A checkpoint through a signed link: streamed to disk by the browser, never held in memory. */
const downloadCheckpoint = (job: Job, name: string) => signedDownload(trainingFilePath(`${job.name}/${name}`));

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
        <div className="card-head">
          <h3 className="card-title">Details</h3>
        </div>
        <dl className="kv">
          <dt>ID</dt>
          <dd className="row" style={{ gap: 4 }}>
            <Mono>{job.id}</Mono> <CopyButton text={job.id} />
          </dd>
          <dt>Type</dt>
          <dd>{sentence(job.type)}</dd>
          <dt>GPUs</dt>
          <dd className="num">{job.gpu_ids}</dd>
          <dt>Status</dt>
          <dd>
            {job.status || '—'} {job.pid ? <span className="muted num">· pid {job.pid}</span> : null}
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
              <button className="link mono" onClick={() => downloadCheckpoint(job, job.latest_checkpoint!.name).catch(toast.error)}>
                {job.latest_checkpoint.name}
              </button>
            ) : (
              <span className="dim">None yet</span>
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
        <div className="card-head">
          <h3 className="card-title">Policy</h3>
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
              <option value="">None</option>
              {remotes.map(r => (
                <option key={r.id} value={r.id}>
                  {r.name} ({r.kind})
                </option>
              ))}
            </select>
          </Field>
          <Field label="Auto-upload remote" hint="Every new checkpoint is uploaded">
            <select value={policy.auto_upload_remote} onChange={e => setPolicy({ ...policy, auto_upload_remote: e.target.value })}>
              <option value="">None</option>
              {remotes.map(r => (
                <option key={r.id} value={r.id}>
                  {r.name} ({r.kind})
                </option>
              ))}
            </select>
          </Field>
        </div>
        <div className="form-actions">
          {busy && <span className="muted small grow">Policy can be changed while the job is not queued or running.</span>}
          <button className="btn btn-primary" onClick={savePolicy} disabled={saving || busy}>
            {saving ? <Spinner size={14} /> : <Save size={14} />} Save policy
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h3 className="card-title">Portable sync</h3>
          {sync?.remote && (
            <button className="btn btn-sm" onClick={forceSync} disabled={syncing}>
              {syncing ? <Spinner size={14} /> : <UploadCloud size={14} />} Sync now
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
            <dd className="num">{sync.step ?? '—'}</dd>
            <dt>Last sync</dt>
            <dd title={fmtTime(sync.last_sync_at)}>{fmtAgo(sync.last_sync_at)}</dd>
            <dt>Lease</dt>
            <dd>
              {sync.lease ? (
                <>
                  <Mono>{sync.lease.node_label}</Mono> <span className="muted">until {fmtTime(sync.lease.expires_at)}</span>
                </>
              ) : (
                <span className="muted">Released</span>
              )}
            </dd>
          </dl>
        )}
        {sync?.last_error && (
          <div style={{ marginTop: 14 }}>
            <Callout tone="danger">{sync.last_error}</Callout>
          </div>
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
      <div className="toolbar">
        <label className="check small">
          <input type="checkbox" checked={follow} onChange={e => setFollow(e.target.checked)} /> Follow output
        </label>
        <button className="btn btn-sm" onClick={load}>
          <RefreshCw size={14} /> Reload
        </button>
      </div>
      {error && (
        <div style={{ marginBottom: 12 }}>
          <Callout tone="danger">{error}</Callout>
        </div>
      )}
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

/** tqdm redraws lines with \r (and ANSI cursor moves); show only the latest state of each line. */
const cleanLog = (t: string) =>
  t
    .replace(/\x1b\[[0-9;?]*[A-Za-z]/g, '')
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
  if (series.keys.length === 0)
    return (
      <Empty icon={<Activity size={18} />} title="No metrics yet">
        The trainer writes <span className="mono">loss_log.db</span> once training steps start.
      </Empty>
    );
  const last = points[points.length - 1];
  return (
    <div className="card">
      <div className="toolbar">
        <div className="row-wrap" style={{ gap: 16 }}>
          <select className="sm" value={series.key} onChange={e => setKey(e.target.value)} style={{ width: 180 }} aria-label="Metric">
            {series.keys.map(k => (
              <option key={k}>{k}</option>
            ))}
          </select>
          <label className="check small">
            Smoothing
            <input type="range" min={0} max={0.99} step={0.01} value={smoothing} onChange={e => setSmoothing(Number(e.target.value))} />
            <span className="num" style={{ minWidth: 28 }}>
              {smoothing.toFixed(2)}
            </span>
          </label>
          <label className="check small">
            <input type="checkbox" checked={log} onChange={e => setLogScale(e.target.checked)} /> Log scale
          </label>
        </div>
        {last && (
          <div className="legend">
            <span>
              <i />
              Smoothed
            </span>
            <span>
              <i className="raw" />
              Raw
            </span>
            <span className="hide-sm">
              Step {last.step} · {last.value.toPrecision(4)} · {points.length} points
            </span>
          </div>
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

  // newest step first, one row per step
  const groups = useMemo(() => {
    const out: { step: number | null; items: Sample[] }[] = [];
    for (const s of [...(items ?? [])].reverse()) {
      const g = out[out.length - 1];
      if (g && g.step === (s.step ?? null)) g.items.push(s);
      else out.push({ step: s.step ?? null, items: [s] });
    }
    for (const g of out) g.items.sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    return out;
  }, [items]);

  if (!items) return <Loading what="samples" />;
  if (items.length === 0)
    return (
      <Empty icon={<ImageIcon size={18} />} title="No samples yet">
        Samples appear every <span className="mono">sample_every</span> steps.
      </Empty>
    );
  const isVideo = (n: string) => /\.(mp4|webm)$/i.test(n);
  return (
    <>
      {groups.map(g => (
        <div className="sample-group" key={String(g.step) + g.items[0].name}>
          <div className="sample-step">{g.step === null ? 'Unknown step' : `Step ${g.step}`}</div>
          <div className="grid grid-media">
            {g.items.map(s => (
              <button key={s.name} className="sample-thumb" onClick={() => setOpen(s)} title={`Sample ${s.index ?? '—'}`} aria-label={`Open step ${s.step ?? '—'} sample ${s.index ?? '—'}`}>
                {isVideo(s.name) ? <video src={imgUrl(s.path)} muted /> : <img src={imgUrl(s.path, true)} loading="lazy" alt={s.name} />}
              </button>
            ))}
          </div>
        </div>
      ))}
      {open && (
        <Modal title={`Step ${open.step ?? '—'} · sample ${open.index ?? '—'}`} onClose={() => setOpen(null)} wide>
          <div className="modal-body">
            {isVideo(open.name) ? <video className="lightbox" src={imgUrl(open.path)} controls autoPlay /> : <img className="lightbox" src={imgUrl(open.path)} alt={open.name} />}
            <div className="mono dim" style={{ marginTop: 10, fontSize: 12 }}>
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
  if (items.length === 0)
    return (
      <Empty icon={<Package size={18} />} title="No checkpoints yet">
        They are saved every <span className="mono">save_every</span> steps and at the end.
      </Empty>
    );
  return (
    <>
      <div className="toolbar">
        <span className="muted small">{plural(items.length, 'checkpoint')}</span>
        <button className="btn btn-sm" onClick={hash} disabled={hashing}>
          {hashing ? <Spinner size={14} /> : <Hash size={14} />} Compute sha256
        </button>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>File</th>
              <th>Step</th>
              <th>Size</th>
              <th className="hide-sm">Saved</th>
              <th className="actions" />
            </tr>
          </thead>
          <tbody>
            {[...items].reverse().map(c => (
              <tr key={c.name}>
                <td>
                  <div className="mono">{c.name}</div>
                  {c.sha256 && (
                    <div className="mono dim ellipsis" style={{ maxWidth: 420, fontSize: 12 }}>
                      {c.sha256}
                    </div>
                  )}
                </td>
                <td>{c.step ?? 'Final'}</td>
                <td className="nowrap">{fmtBytes(c.size)}</td>
                <td className="small muted nowrap hide-sm" title={fmtTime(c.modified)}>
                  {fmtAgo(c.modified)}
                </td>
                <td className="actions">
                  <div className="row end">
                    {remotes.length > 0 && (
                      <select className="sm" value="" onChange={e => e.target.value && uploadTo(c, e.target.value)} style={{ width: 132 }} aria-label="Upload to remote">
                        <option value="">Upload to…</option>
                        {remotes.map(r => (
                          <option key={r.id} value={r.id}>
                            {r.name}
                          </option>
                        ))}
                      </select>
                    )}
                    <button className="btn btn-sm" onClick={() => downloadCheckpoint(job, c.name).catch(toast.error)}>
                      <Download size={14} /> Download
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ---------------------------------------------------------------- attempts

function AttemptsTab({ job, ver }: { job: Job; ver: number }) {
  const [items, setItems] = useState<Attempt[] | null>(null);
  usePoll(async () => setItems((await api.get<Items<Attempt>>(`/v1/jobs/${encodeURIComponent(job.id)}/attempts`)).items), 15000, [job.id, ver]);
  if (!items) return <Loading what="attempts" />;
  if (items.length === 0)
    return (
      <Empty icon={<History size={18} />} title="Never started">
        Each launch of the job is recorded here.
      </Empty>
    );
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Attempt</th>
            <th>Started</th>
            <th>Steps</th>
            <th>Ended</th>
            <th className="hide-sm">Reason</th>
          </tr>
        </thead>
        <tbody>
          {[...items].reverse().map(a => (
            <tr key={a.n}>
              <td>
                <span className="row" style={{ gap: 8 }}>
                  <span className="cell-title">#{a.n}</span>
                  {a.auto && <span className="tag">Auto</span>}
                </span>
              </td>
              <td className="small muted nowrap">{fmtTime(a.started_at)}</td>
              <td className="nowrap">
                {a.start_step} → {a.end_step ?? '…'}
              </td>
              <td>{a.end_state ? <StateBadge state={a.end_state} /> : <StateBadge state="running" />}</td>
              <td className="small muted hide-sm" style={{ maxWidth: 420 }}>
                {a.end_reason ?? ''}
                {a.pid ? (
                  <span className="dim">
                    {a.end_reason ? ' · ' : ''}pid {a.pid}
                  </span>
                ) : null}
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
      <div className="toolbar">
        <span className="muted small">Job config as sent to the trainer</span>
        <CopyButton text={text} bordered />
      </div>
      <pre className="json">{text}</pre>
    </>
  );
}

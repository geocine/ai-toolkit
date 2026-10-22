// Run bundle UI for the console: import (upload, link or library), export
// (download or push to a library), task lists and a remote's library.
import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, Download, FileArchive, Library, Pause, Play, Trash2, Upload, UploadCloud, X } from 'lucide-react';
import { ApiError, api, fmtAgo, fmtBytes, fmtTime, headSize, qs, signedDownload } from './api';
import {
  dropTransfer,
  dropTransfersFor,
  infoFromSummary,
  isActive,
  pauseUpload,
  phaseLabel,
  resumability,
  resumableHint,
  resumableLabel,
  startUpload,
  transferKey,
  unknownInfo,
  useTasks,
  useTransfers,
  type BundleInfo,
  type Tasks,
  type Transfer,
} from './bundles';
import { go, href } from './router';
import type { BundleTask, BundleUpload, Items, Job, LibraryBundle, Remote, SystemInfo } from './types';
import { Bar, Callout, Empty, Field, Loading, Modal, Spinner, StateBadge, useConfirm, useToast } from './ui';

const taskState = (t: BundleTask) => (t.state === 'done' ? 'completed' : t.state);
const conflictCodes = ['conflict', 'dataset_exists'];

/** Phase, bar and bytes of a running task. */
export function TaskProgress({ task }: { task: BundleTask }) {
  const known = task.bytes > 0;
  return (
    <div className="task-progress">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <span className="small">{phaseLabel(task)}</span>
        <span className="small muted num">{known ? `${fmtBytes(task.bytes_done)} / ${fmtBytes(task.bytes)}` : task.bytes_done ? fmtBytes(task.bytes_done) : ''}</span>
      </div>
      <Bar value={known ? task.bytes_done : 0} max={known ? task.bytes : 1} />
    </div>
  );
}

/** host/…/file.zip of a link, without its query (signatures stay out of sight). */
const linkLabel = (u: string) => {
  try {
    const x = new URL(u);
    return `${x.host}/…/${x.pathname.split('/').filter(Boolean).pop() ?? ''}`;
  } catch {
    return u;
  }
};

const transferLine = (t: Transfer) => {
  const pct = t.file.size ? Math.floor((t.sent / t.file.size) * 100) : 0;
  const left = t.rate > 0 ? (t.file.size - t.sent) / t.rate : 0;
  const eta = left > 0 ? (left < 60 ? `${Math.ceil(left)}s` : `${Math.ceil(left / 60)}m`) : '';
  return `${fmtBytes(t.sent)} / ${fmtBytes(t.file.size)} · ${pct}%${t.state === 'uploading' && t.rate > 0 ? ` · ${fmtBytes(t.rate)}/s · ${eta} left` : ''}`;
};

// ---------------------------------------------------------------- import

type Source =
  | { kind: 'upload'; uploadId: string; label: string; info: BundleInfo; size?: number }
  | { kind: 'url'; url: string; label: string; info: BundleInfo }
  | { kind: 'remote'; remote: string; key: string; label: string; info: BundleInfo; size?: number };

export type ImportStart = { tab?: 'upload' | 'url' | 'library'; source?: Source };

export const librarySource = (remote: string, b: LibraryBundle): Source => ({ kind: 'remote', remote, key: b.key, label: b.summary?.name ?? b.id, info: infoFromSummary(b.summary, b.id), size: b.size });
export const uploadSource = (u: BundleUpload, info?: BundleInfo | null): Source => ({ kind: 'upload', uploadId: u.id, label: u.filename, info: info ?? unknownInfo(), size: u.size });

/** Import a bundle: pick a source, name the job and dataset, follow the task. */
export function ImportDialog({ start, tasks, onClose }: { start?: ImportStart; tasks: Tasks; onClose: () => void }) {
  const [tab, setTab] = useState(start?.tab ?? 'upload');
  const [source, setSource] = useState<Source | null>(start?.source ?? null);
  const [url, setUrl] = useState('');
  const [name, setName] = useState('');
  const [dataset, setDataset] = useState('');
  const [gpu, setGpu] = useState('0');
  const [startNow, setStartNow] = useState(true);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [nameError, setNameError] = useState('');
  const [datasetError, setDatasetError] = useState('');
  const [gpus, setGpus] = useState<SystemInfo['gpus']>([]);
  const transfers = useTransfers();
  const fileInput = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [pending, setPending] = useState<string | null>(null); // transfer key this dialog started

  useEffect(() => {
    api.get<SystemInfo>('/v1/system').then(s => setGpus(s.gpus ?? []), () => {});
  }, []);

  // defaults follow the chosen bundle
  useEffect(() => {
    setName(source?.info.name ?? '');
    setDataset(source?.info.datasets[0] ?? '');
    setStartNow(source?.info.resumable !== 'dataset');
    setUploadId(source?.kind === 'upload' ? source.uploadId : null);
    setNameError('');
    setDatasetError('');
    setError('');
  }, [source]);

  const transfer = pending ? transfers.find(t => t.key === pending) : undefined;
  const task = taskId ? tasks.tasks?.find(t => t.id === taskId) : undefined;
  const info = source?.info;

  const explain = (code: string, message: string) => {
    if (code === 'conflict') setNameError(`A job named ${name.trim() || info?.guess || 'like this one'} already exists. Pick another name.`);
    else if (code === 'dataset_exists') setDatasetError(`A dataset of that name holds other files. Pick another name.`);
    else if (code === 'too_large') setError(`Not enough disk space: ${message}`);
    else setError(message);
  };

  useEffect(() => {
    if (!task || task.state !== 'failed') return;
    if (task.upload_id) setUploadId(task.upload_id);
    if (task.error && conflictCodes.includes(task.error.code) && task.upload_id) {
      explain(task.error.code, task.error.message);
      setTaskId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task?.state, task?.id]);

  useEffect(() => {
    if (task?.state === 'done' && uploadId) dropTransfersFor(uploadId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task?.state]);

  const pick = (files: FileList | null) => {
    const f = Array.from(files ?? []).find(x => /\.zip$/i.test(x.name));
    if (!f) return;
    setPending(transferKey(f));
    startUpload(f, t => t.upload && setSource({ kind: 'upload', uploadId: t.upload.id, label: t.file.name, info: t.info ?? unknownInfo(), size: t.file.size }));
    if (fileInput.current) fileInput.current.value = '';
  };

  const submit = async () => {
    const src = source ?? (tab === 'url' && url.trim() ? ({ kind: 'url', url: url.trim(), label: linkLabel(url.trim()), info: unknownInfo() } as Source) : null);
    if (!src) return;
    setBusy(true);
    setError('');
    setNameError('');
    setDatasetError('');
    const from = uploadId ? { upload_id: uploadId } : src.kind === 'url' ? { url: src.url } : src.kind === 'remote' ? { remote: src.remote, key: src.key } : { upload_id: src.uploadId };
    try {
      const t = await api.post<BundleTask>('/v1/bundles/imports', { ...from, name: name.trim() || undefined, dataset: dataset.trim() || undefined, gpu_ids: gpu, start: startNow });
      tasks.upsert(t);
      setTaskId(t.id);
    } catch (e) {
      if (e instanceof ApiError) explain(e.code, e.message);
      else setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const running = task && isActive(task);
  const done = task?.state === 'done';
  const failed = task?.state === 'failed';
  const canSubmit = !!source || (tab === 'url' && /^https?:\/\//.test(url.trim()));

  return (
    <Modal title="Import bundle" onClose={onClose}>
      <div className="modal-body import">
        {!source && !start?.source && (
          <>
            <div className="tabs tabs-tight" role="tablist">
              {(['upload', 'url', 'library'] as const).map(t => (
                <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'tab active' : 'tab'} onClick={() => setTab(t)}>
                  {t === 'upload' ? 'Upload' : t === 'url' ? 'From a link' : 'From a library'}
                </button>
              ))}
            </div>
            {tab === 'upload' &&
              (transfer ? (
                <div className="transfer">
                  <div className="row" style={{ justifyContent: 'space-between' }}>
                    <span className="mono small ellipsis">{transfer.file.name}</span>
                    {transfer.state === 'uploading' && (
                      <button className="btn btn-sm" onClick={() => pauseUpload(transfer.key)}>
                        <Pause size={14} /> Pause
                      </button>
                    )}
                    {(transfer.state === 'paused' || transfer.state === 'failed') && (
                      <button className="btn btn-sm" onClick={() => startUpload(transfer.file, t => t.upload && setSource({ kind: 'upload', uploadId: t.upload.id, label: t.file.name, info: t.info ?? unknownInfo(), size: t.file.size }))}>
                        <Play size={14} /> Resume
                      </button>
                    )}
                  </div>
                  <Bar value={transfer.sent} max={transfer.file.size} />
                  <div className="small muted num">{transferLine(transfer)}</div>
                  {transfer.error && <div className="small danger-text">{transfer.error}</div>}
                </div>
              ) : (
                <label
                  className={over ? 'drop over' : 'drop'}
                  onDragOver={e => {
                    e.preventDefault();
                    setOver(true);
                  }}
                  onDragLeave={() => setOver(false)}
                  onDrop={e => {
                    e.preventDefault();
                    setOver(false);
                    pick(e.dataTransfer.files);
                  }}
                >
                  <Upload size={16} />
                  <span>Drop a bundle .zip or click to choose. It goes up in pieces and resumes after a reload.</span>
                  <input ref={fileInput} type="file" accept=".zip,application/zip" hidden onChange={e => pick(e.target.files)} />
                </label>
              ))}
            {tab === 'url' && (
              <Field label="Link to a .zip" hint="The server downloads it itself (http or https, e.g. a presigned S3 link).">
                <input type="text" value={url} onChange={e => setUrl(e.target.value)} placeholder="https://…/run.zip" autoFocus />
              </Field>
            )}
            {tab === 'library' && <LibraryPicker onPick={(remote, b) => setSource(librarySource(remote, b))} />}
          </>
        )}

        {source && info && (
          <div className="bundle-brief">
            <div className="row" style={{ justifyContent: 'space-between', gap: 8 }}>
              <span className="cell-title ellipsis">{info.guess || source.label}</span>
              {info.resumable && (
                <span className={info.resumable === 'resumable' ? 'badge s-completed' : 'tag'} title={resumableHint[info.resumable]}>
                  {resumableLabel[info.resumable]}
                </span>
              )}
            </div>
            <div className="small muted num">
              {info.step != null ? `Step ${info.step}${info.total ? ` / ${info.total}` : ''}` : 'Step unknown'}
              {info.images != null ? ` · ${info.images} images` : ''}
              {source.kind !== 'url' && source.size ? ` · ${fmtBytes(source.size)}` : ''}
              {source.kind === 'remote' ? ` · from ${source.remote}` : ''}
            </div>
          </div>
        )}

        {done ? (
          <Callout>
            Imported <b>{task!.job_name}</b>
            {task!.dataset ? (
              <>
                {' '}
                with the dataset <b>{task!.dataset}</b>
              </>
            ) : null}
            .{startNow ? ' It is queued and resumes from its newest checkpoint.' : ''}
            {task!.skipped?.length ? ` Skipped ${task!.skipped.length} junk files.` : ''}
          </Callout>
        ) : running ? (
          <TaskProgress task={task!} />
        ) : (
          (source || tab === 'url') && (
            <div className="form-grid import-form">
              <Field label="Job name" hint={nameError ? <span className="danger-text">{nameError}</span> : 'Saves are renamed to match.'}>
                <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder={info?.guess || 'From the bundle'} aria-invalid={!!nameError} />
              </Field>
              <Field label="Dataset name" hint={datasetError ? <span className="danger-text">{datasetError}</span> : 'One with the same files is reused.'}>
                <input type="text" value={dataset} onChange={e => setDataset(e.target.value)} placeholder={info?.datasets[0] || 'From the bundle'} aria-invalid={!!datasetError} />
              </Field>
              <Field label="GPU">
                <select value={gpu} onChange={e => setGpu(e.target.value)}>
                  {(gpus.length ? gpus : [{ index: 0, name: '' } as SystemInfo['gpus'][number]]).map(g => (
                    <option key={g.index} value={String(g.index)}>
                      {g.index}
                      {g.name ? ` · ${g.name}` : ''}
                    </option>
                  ))}
                </select>
              </Field>
              <label className="check">
                <input type="checkbox" checked={startNow} onChange={e => setStartNow(e.target.checked)} /> {info?.resumable === 'dataset' ? 'Start now' : 'Continue training now'}
              </label>
            </div>
          )
        )}
        {failed && task!.error && !conflictCodes.includes(task!.error.code) && (
          <div style={{ marginTop: 12 }}>
            <Callout tone="danger">{task!.error.message}</Callout>
          </div>
        )}
        {error && (
          <div style={{ marginTop: 12 }}>
            <Callout tone="danger">{error}</Callout>
          </div>
        )}
      </div>
      <div className="modal-actions">
        {done ? (
          <>
            <button className="btn" onClick={onClose}>
              Close
            </button>
            {task!.job_id && (
              <button className="btn btn-primary" onClick={() => go('jobs', task!.job_id!)}>
                Open job <ArrowRight size={14} />
              </button>
            )}
          </>
        ) : running ? (
          <button className="btn" onClick={onClose}>
            Continue in the background
          </button>
        ) : (
          <>
            <button className="btn" onClick={onClose}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={submit} disabled={!canSubmit || busy}>
              {busy ? <Spinner /> : null} {failed ? 'Try again' : 'Import'}
            </button>
          </>
        )}
      </div>
    </Modal>
  );
}

function LibraryPicker({ onPick }: { onPick: (remote: string, b: LibraryBundle) => void }) {
  const [remotes, setRemotes] = useState<Remote[] | null>(null);
  const [remote, setRemote] = useState('');
  const [items, setItems] = useState<LibraryBundle[] | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    api.get<Items<Remote>>('/v1/remotes').then(r => {
      setRemotes(r.items);
      if (r.items[0]) setRemote(r.items[0].name);
    }, e => setError(e.message));
  }, []);
  useEffect(() => {
    if (!remote) return;
    setItems(null);
    api.get<Items<LibraryBundle>>(`/v1/bundles${qs({ remote })}`).then(r => setItems(r.items), e => setError(e.message));
  }, [remote]);
  if (error) return <Callout tone="danger">{error}</Callout>;
  if (!remotes) return <Loading />;
  if (!remotes.length) return <div className="muted small">No remotes on this server. Add one under Remotes to keep a library.</div>;
  return (
    <div className="stack" style={{ gap: 10 }}>
      {remotes.length > 1 && (
        <select className="sm" value={remote} onChange={e => setRemote(e.target.value)} aria-label="Remote">
          {remotes.map(r => (
            <option key={r.id} value={r.name}>
              {r.name}
            </option>
          ))}
        </select>
      )}
      {!items ? (
        <Loading />
      ) : !items.length ? (
        <div className="muted small">No bundles in {remote} yet.</div>
      ) : (
        <div className="pick-list">
          {items.map(b => (
            <button key={b.id} className="pick" onClick={() => onPick(remote, b)}>
              <span className="cell-title ellipsis">{b.summary?.name ?? b.id}</span>
              <span className="small muted num">
                {b.summary ? `Step ${b.summary.step} · ` : ''}
                {fmtBytes(b.size)} · {fmtAgo(b.modified)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- imports list

/** Uploads in flight or waiting, and recent import tasks (Jobs page). */
export function ImportActivity({ tasks, onLoad }: { tasks: Tasks; onLoad: (s: ImportStart) => void }) {
  const toast = useToast();
  const transfers = useTransfers();
  const [sessions, setSessions] = useState<BundleUpload[]>([]);
  const reloadSessions = () => api.get<Items<BundleUpload>>('/v1/bundles/uploads').then(r => setSessions(r.items), () => {});
  useEffect(() => {
    reloadSessions();
    const t = setInterval(reloadSessions, 15000);
    return () => clearInterval(t);
  }, [tasks.tasks?.length]);

  const localIds = new Set(transfers.map(t => t.upload?.id));
  const localKeys = new Set(transfers.map(t => t.key));
  const orphans = sessions.filter(u => !localIds.has(u.id) && !localKeys.has(transferKey({ name: u.filename, size: u.size })));
  // running and retryable imports always; finished ones from the last hour, a few at most
  const recent = useMemo(() => {
    const all = tasks.tasks ?? [];
    const staged = (t: BundleTask) => t.state === 'failed' && !!t.upload_id && sessions.some(u => u.id === t.upload_id);
    const live = all.filter(t => isActive(t) || staged(t));
    const done = all.filter(t => !live.includes(t) && Date.now() - new Date(t.updated_at).getTime() < 3600e3).slice(0, 3);
    return [...live, ...done];
  }, [tasks.tasks, sessions]);
  const discard = async (id: string) => {
    try {
      await api.del(`/v1/bundles/uploads/${encodeURIComponent(id)}`);
      dropTransfersFor(id);
      reloadSessions();
    } catch (e) {
      toast.error(e);
    }
  };
  if (!transfers.length && !orphans.length && !recent.length) return null;
  return (
    <div className="table-wrap" style={{ marginBottom: 24 }}>
      <table>
        <thead>
          <tr>
            <th>Import</th>
            <th className="col-state">State</th>
            <th className="col-progress">Progress</th>
            <th className="actions" />
          </tr>
        </thead>
        <tbody>
          {transfers.map(t => (
            <tr key={t.key}>
              <td>
                <div className="cell-title ellipsis mono small">{t.file.name}</div>
                <div className="cell-sub">Upload from this browser</div>
              </td>
              <td>
                <StateBadge state={t.state === 'complete' ? 'completed' : t.state === 'uploading' || t.state === 'starting' ? 'running' : t.state === 'failed' ? 'failed' : 'paused'} />
              </td>
              <td>
                <div className="stack" style={{ gap: 4 }}>
                  <Bar value={t.sent} max={t.file.size} />
                  <span className="small muted num">{t.state === 'complete' ? 'Uploaded, ready to import' : transferLine(t)}</span>
                </div>
              </td>
              <td className="actions">
                <div className="row end" style={{ gap: 4 }}>
                  {t.state === 'complete' && t.upload && (
                    <button className="btn btn-sm btn-primary" onClick={() => onLoad({ source: uploadSource(t.upload!, t.info) })}>
                      Import
                    </button>
                  )}
                  {t.state === 'uploading' && (
                    <button className="btn btn-sm" onClick={() => pauseUpload(t.key)}>
                      <Pause size={14} /> Pause
                    </button>
                  )}
                  {(t.state === 'paused' || t.state === 'failed') && (
                    <button className="btn btn-sm" onClick={() => startUpload(t.file)}>
                      <Play size={14} /> Resume
                    </button>
                  )}
                  <button className="icon-btn danger" aria-label="Discard" title="Discard" onClick={() => (t.upload ? discard(t.upload.id) : dropTransfer(t.key))}>
                    <X size={14} />
                  </button>
                </div>
              </td>
            </tr>
          ))}
          {orphans.map(u => (
            <tr key={u.id}>
              <td>
                <div className="cell-title ellipsis mono small">{u.filename}</div>
                <div className="cell-sub">{u.complete ? 'Uploaded' : 'Paused; choose the same file again to continue'}</div>
              </td>
              <td>
                <StateBadge state={u.complete ? 'completed' : 'paused'} />
              </td>
              <td>
                <div className="stack" style={{ gap: 4 }}>
                  <Bar value={u.received} max={u.size} />
                  <span className="small muted num">
                    {fmtBytes(u.received)} / {fmtBytes(u.size)}
                  </span>
                </div>
              </td>
              <td className="actions">
                <div className="row end" style={{ gap: 4 }}>
                  {u.complete ? (
                    <button className="btn btn-sm btn-primary" onClick={() => onLoad({ source: uploadSource(u) })}>
                      Import
                    </button>
                  ) : (
                    <button className="btn btn-sm" onClick={() => onLoad({ tab: 'upload' })} title="Choose the same file again to continue">
                      <Play size={14} /> Resume
                    </button>
                  )}
                  <button className="icon-btn danger" aria-label="Discard" title="Discard" onClick={() => discard(u.id)}>
                    <X size={14} />
                  </button>
                </div>
              </td>
            </tr>
          ))}
          {recent.map(t => (
            <tr key={t.id}>
              <td>
                <div className="cell-title ellipsis">{t.job_name ?? t.key?.split('/').pop() ?? 'Bundle'}</div>
                <div className="cell-sub ellipsis">
                  {t.source === 'remote' ? `From ${t.remote}` : t.source === 'url' ? 'From a link' : t.source === 'upload' ? 'From an upload' : 'At boot'} · {fmtAgo(t.updated_at)}
                </div>
                {t.error && <div className="small danger-text">{t.error.message}</div>}
              </td>
              <td>
                <StateBadge state={taskState(t)} />
              </td>
              <td>{isActive(t) ? <TaskProgress task={t} /> : <span className="small muted">{t.state === 'done' ? `Dataset ${t.dataset ?? '—'}` : '—'}</span>}</td>
              <td className="actions">
                {t.state === 'done' && t.job_id && (
                  <a className="btn btn-sm" href={href('jobs', t.job_id)}>
                    Open <ArrowRight size={14} />
                  </a>
                )}
                {t.state === 'failed' && t.upload_id && sessions.some(u => u.id === t.upload_id) && (
                  <button className="btn btn-sm" onClick={() => onLoad({ source: { kind: 'upload', uploadId: t.upload_id!, label: t.key?.split('/').pop() ?? 'staged zip', info: { ...unknownInfo(t.job_name ?? ''), name: t.job_name ?? '' } } })}>
                    {t.error && conflictCodes.includes(t.error.code) ? 'Pick another name' : 'Try again'}
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- export

type Ckpts = 'latest' | 'all' | 'none';

/** Export a job as a bundle: a signed download, or a push to a remote's library. */
export function ExportDialog({ job, onClose }: { job: Job; onClose: () => void }) {
  const [checkpoints, setCheckpoints] = useState<Ckpts>('latest');
  const [optimizer, setOptimizer] = useState(true);
  const [samples, setSamples] = useState(true);
  const [caches, setCaches] = useState(true);
  const [size, setSize] = useState<number | null>(null);
  const [remotes, setRemotes] = useState<Remote[]>([]);
  const [remote, setRemote] = useState('');
  const [bundle, setBundle] = useState('');
  const [overwrite, setOverwrite] = useState(false);
  const [exists, setExists] = useState(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [pushId, setPushId] = useState<string | null>(null);
  const pushes = useTasks('push', job.id);

  useEffect(() => {
    api.get<Items<Remote>>('/v1/remotes').then(r => setRemotes(r.items), () => {});
  }, []);
  const target = remote || remotes[0]?.name || '';
  const path = `/v1/jobs/${encodeURIComponent(job.id)}/bundle${qs({ checkpoints, optimizer: checkpoints !== 'none' && optimizer ? 1 : 0, samples: samples ? 1 : 0, caches: caches ? 1 : 0 })}`;

  useEffect(() => {
    setSize(null);
    const ctrl = new AbortController();
    const t = setTimeout(() => headSize(path, ctrl.signal).then(setSize, () => {}), 200);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [path]);

  const current = pushId ? pushes.tasks?.find(t => t.id === pushId) : pushes.tasks?.find(isActive);
  const clash = current?.state === 'failed' && current.error?.code === 'conflict';
  useEffect(() => {
    if (clash) setExists(true);
  }, [clash]);
  const defaultName = `${job.name}_step${job.latest_checkpoint?.step ?? job.step}`;

  const download = async () => {
    setBusy('download');
    setError('');
    try {
      await signedDownload(path);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setBusy('');
    }
  };
  const push = async () => {
    setBusy('push');
    setError('');
    try {
      const t = await api.post<BundleTask>(`/v1/jobs/${encodeURIComponent(job.id)}/bundle/push`, {
        remote: target,
        bundle: bundle.trim() || undefined,
        overwrite: overwrite || undefined,
        checkpoints,
        optimizer: checkpoints !== 'none' && optimizer,
        samples,
        caches,
      });
      pushes.upsert(t);
      setPushId(t.id);
      setExists(false);
    } catch (e: any) {
      if (e instanceof ApiError && e.status === 409) setExists(true);
      else setError(e?.message ?? String(e));
    } finally {
      setBusy('');
    }
  };

  return (
    <Modal title="Export bundle" onClose={onClose}>
      <div className="modal-body export">
        <div className="small muted" style={{ marginBottom: 14 }}>
          One zip with the dataset and the job folder, in the layout ai-toolkit resumes from.
        </div>
        <div className="form-grid" style={{ gridTemplateColumns: '1fr' }}>
          <Field group label="Checkpoints" hint={checkpoints === 'none' ? 'Dataset and config only: trains from step 0.' : checkpoints === 'all' ? 'Every save.' : 'The newest save, enough to resume.'}>
            <div className="seg seg-text" role="radiogroup" aria-label="Checkpoints">
              {(['latest', 'all', 'none'] as const).map(c => (
                <button key={c} type="button" role="radio" aria-checked={checkpoints === c} className={checkpoints === c ? 'on' : ''} onClick={() => setCheckpoints(c)}>
                  {c === 'latest' ? 'Latest' : c === 'all' ? 'All' : 'None'}
                </button>
              ))}
            </div>
          </Field>
          <div className="stack" style={{ gap: 8 }}>
            <label className="check">
              <input type="checkbox" checked={checkpoints !== 'none' && optimizer} disabled={checkpoints === 'none'} onChange={e => setOptimizer(e.target.checked)} /> Optimizer state
              <span className="muted small">(needed to resume exactly)</span>
            </label>
            <label className="check">
              <input type="checkbox" checked={samples} onChange={e => setSamples(e.target.checked)} /> Samples
            </label>
            <label className="check">
              <input type="checkbox" checked={caches} onChange={e => setCaches(e.target.checked)} /> Dataset caches
              <span className="muted small">(latents, text embeddings)</span>
            </label>
          </div>
        </div>
        <div className="row" style={{ justifyContent: 'space-between', margin: '16px 0 0' }}>
          <span className="small muted">
            Zip size <b className="num">{size ? fmtBytes(size) : '…'}</b>
          </span>
          <button className="btn" onClick={download} disabled={busy === 'download'}>
            {busy === 'download' ? <Spinner /> : <Download size={14} />} Download zip
          </button>
        </div>

        <div className="export-push">
          <div className="card-title" style={{ marginBottom: 10 }}>
            Save to a library
          </div>
          {!remotes.length ? (
            <div className="small muted">No remotes yet. Add one under Remotes.</div>
          ) : (
            <div className="stack" style={{ gap: 12 }}>
              <div className="form-grid">
                {remotes.length > 1 && (
                  <Field label="Remote">
                    <select value={target} onChange={e => setRemote(e.target.value)}>
                      {remotes.map(r => (
                        <option key={r.id} value={r.name}>
                          {r.name} ({r.kind})
                        </option>
                      ))}
                    </select>
                  </Field>
                )}
                <Field label="Bundle name">
                  <input
                    type="text"
                    value={bundle}
                    placeholder={defaultName}
                    onChange={e => {
                      setBundle(e.target.value);
                      setExists(false);
                    }}
                  />
                </Field>
              </div>
              {exists && (
                <Callout tone="warn">
                  <div>{bundle.trim() || defaultName} is already in the library.</div>
                  <label className="check small" style={{ marginTop: 6 }}>
                    <input type="checkbox" checked={overwrite} onChange={e => setOverwrite(e.target.checked)} /> Replace it
                  </label>
                </Callout>
              )}
              {current && isActive(current) ? (
                <TaskProgress task={current} />
              ) : (
                <div>
                  <button className="btn btn-primary" onClick={push} disabled={busy === 'push' || (exists && !overwrite)}>
                    {busy === 'push' ? <Spinner /> : <UploadCloud size={14} />} Push to {target}
                  </button>
                </div>
              )}
              {current?.state === 'failed' && current.error && !clash && <Callout tone="danger">{current.error.message}</Callout>}
              {current?.state === 'done' && (
                <Callout>
                  Saved as <span className="mono">{current.key?.split('/').pop()}</span> in {current.remote}.
                </Callout>
              )}
            </div>
          )}
        </div>
        {error && (
          <div style={{ marginTop: 12 }}>
            <Callout tone="danger">{error}</Callout>
          </div>
        )}
      </div>
    </Modal>
  );
}

// ---------------------------------------------------------------- library

/** A remote's bundles/ folder (Remotes page). */
export function LibraryTable({ remote, onImport }: { remote: string; onImport: (s: ImportStart) => void }) {
  const toast = useToast();
  const confirm = useConfirm();
  const [items, setItems] = useState<LibraryBundle[] | null>(null);
  const [error, setError] = useState('');
  const load = () =>
    api.get<Items<LibraryBundle>>(`/v1/bundles${qs({ remote })}`).then(
      r => {
        setItems(r.items);
        setError('');
      },
      e => {
        setError(e.message);
        setItems([]);
      },
    );
  useEffect(() => {
    setItems(null);
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [remote]);

  const del = async (b: LibraryBundle) => {
    if (!(await confirm({ title: 'Delete bundle', body: <>Delete <b>{b.summary?.name ?? b.id}</b> ({fmtBytes(b.size)}) from {remote}?</>, confirm: 'Delete', danger: true }))) return;
    try {
      await api.del(`/v1/bundles/${encodeURIComponent(b.id)}${qs({ remote })}`);
      toast.success('Bundle deleted');
      load();
    } catch (e) {
      toast.error(e);
    }
  };

  if (error) return <Callout tone="danger">{error}</Callout>;
  if (!items) return <Loading />;
  if (!items.length)
    return (
      <Empty icon={<Library size={18} />} title="No bundles in this remote">
        Push one from a job's Export menu; it lands in <span className="mono">bundles/</span>.
      </Empty>
    );
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Bundle</th>
            <th>Step</th>
            <th className="hide-sm">Images</th>
            <th>Size</th>
            <th className="hide-sm">Saved</th>
            <th className="actions" />
          </tr>
        </thead>
        <tbody>
          {items.map(b => {
            const s = b.summary;
            const kind = resumability(s);
            return (
              <tr key={b.id}>
                <td>
                  <div className="row" style={{ gap: 8 }}>
                    <span className="cell-title">{s?.name ?? b.id}</span>
                    {kind && (
                      <span className={kind === 'resumable' ? 'badge s-completed' : 'tag'} title={resumableHint[kind]}>
                        {resumableLabel[kind]}
                      </span>
                    )}
                  </div>
                  <div className="cell-sub mono ellipsis" style={{ fontSize: 12, maxWidth: 360 }} title={s?.base_model ? `${b.key} · ${s.base_model}` : b.key}>
                    {b.id}
                    {s?.base_model ? ` · ${s.base_model}` : ''}
                  </div>
                </td>
                <td className="nowrap">{s ? `${s.step}${s.total_steps ? ` / ${s.total_steps}` : ''}` : '—'}</td>
                <td className="hide-sm">{s?.images ?? '—'}</td>
                <td className="nowrap">{fmtBytes(b.size)}</td>
                <td className="small muted nowrap hide-sm" title={fmtTime(s?.created_at ?? b.modified)}>
                  {fmtAgo(s?.created_at ?? b.modified)}
                </td>
                <td className="actions">
                  <div className="row end" style={{ gap: 4 }}>
                    <button className="btn btn-sm" onClick={() => onImport({ source: librarySource(remote, b) })}>
                      <FileArchive size={14} /> Import
                    </button>
                    <button
                      className="icon-btn"
                      onClick={() => signedDownload(`/v1/bundles/content${qs({ remote, id: b.id })}`).catch(e => toast.error(e))}
                      aria-label="Download"
                      title="Download"
                    >
                      <Download size={14} />
                    </button>
                    <button className="icon-btn danger" onClick={() => del(b)} aria-label="Delete" title="Delete">
                      <Trash2 size={14} />
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

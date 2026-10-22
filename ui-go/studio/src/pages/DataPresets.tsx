import { ArrowRight, Archive, FileArchive, Pause, Play, Trash2, Upload, X } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import { Confirm } from '../components/Dialog';
import { LoadDialog, TaskProgress, type LoadSource } from '../components/LoadDialog';
import { Empty, ErrorNote, Progress, Section, StatePill, useLoad } from '../components/ui';
import { del, get, type BundleTask, type BundleUpload, type Items, type LibraryBundle, type Remote, type Server } from '../lib/api';
import {
  dropTransfer,
  infoFromSummary,
  isActive,
  pauseUpload,
  resumability,
  resumableHint,
  resumableLabel,
  startUpload,
  unknownInfo,
  transferKey,
  useTasks,
  useTransfers,
  type Transfer,
} from '../lib/bundles';
import { ago, bytes, duration } from '../lib/format';
import { href } from '../lib/route';
import { useStored } from '../lib/store';

const RECENT_MS = 3600 * 1000;

/**
 * Data presets: run bundles kept in a remote's library (bundles/), loaded
 * onto this server to continue training; plus uploads from this computer.
 */
export function DataPresets({ server }: { server: Server }) {
  const remotes = useLoad(server, s => get<Items<Remote>>(s, '/v1/remotes'), []);
  const [picked, setPicked] = useStored<string>(`aitk.studio.library.${server.id}`, '');
  const list = remotes.data?.items ?? [];
  const remote = list.find(r => r.name === picked)?.name ?? list[0]?.name ?? '';
  const library = useLoad(server, s => (remote ? get<Items<LibraryBundle>>(s, `/v1/bundles?remote=${encodeURIComponent(remote)}`) : Promise.resolve({ items: [] })), [remote]);
  const sessions = useLoad(server, s => get<Items<BundleUpload>>(s, '/v1/bundles/uploads'), []);
  const tasks = useTasks(server, 'import');
  const transfers = useTransfers(server.id);
  const [load, setLoad] = useState<LoadSource | null>(null);
  const [remove, setRemove] = useState<LibraryBundle | null>(null);
  const [removing, setRemoving] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const openLoadForTransfer = (t: Transfer) => {
    if (!t.upload) return;
    setLoad({ uploadId: t.upload.id, label: t.file.name, size: t.file.size, info: t.info ?? unknownInfo() });
  };

  const take = (files: FileList | File[] | null) => {
    setError(null);
    const f = Array.from(files ?? []).find(x => /\.zip$/i.test(x.name));
    if (!f) {
      if (files && Array.from(files).length) setError(new Error('Choose a run bundle: a .zip file.'));
      return;
    }
    startUpload(server, f, t => {
      sessions.reload();
      openLoadForTransfer(t);
    }).finally(() => sessions.reload());
    if (input.current) input.current.value = '';
  };

  // server-side sessions this tab is not uploading: paused by a reload, or complete and waiting
  const localIds = new Set(transfers.map(t => t.upload?.id).filter(Boolean));
  const localKeys = new Set(transfers.map(t => t.key));
  const orphans = (sessions.data?.items ?? []).filter(u => !localIds.has(u.id) && !localKeys.has(transferKey(server, { name: u.filename, size: u.size })));

  const staged = (t: BundleTask) => !!t.upload_id && (sessions.data?.items ?? []).some(u => u.id === t.upload_id);
  // running tasks and failures that can still be retried go on top; finished ones below the library
  const activeTasks = (tasks.tasks ?? []).filter(t => isActive(t) || (t.state === 'failed' && staged(t)));
  const recentTasks = useMemo(
    () => (tasks.tasks ?? []).filter(t => !isActive(t) && !(t.state === 'failed' && staged(t)) && Date.now() - new Date(t.updated_at).getTime() < RECENT_MS).slice(0, 3),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tasks.tasks, sessions.data],
  );

  const discard = async (u: BundleUpload) => {
    try {
      await del(server, `/v1/bundles/uploads/${encodeURIComponent(u.id)}`);
      for (const t of transfers) if (t.upload?.id === u.id) dropTransfer(t.key);
      sessions.reload();
    } catch (e) {
      setError(e);
    }
  };

  const doRemove = async () => {
    if (!remove) return;
    setRemoving(true);
    try {
      await del(server, `/v1/bundles/${encodeURIComponent(remove.id)}?remote=${encodeURIComponent(remote)}`);
      setRemove(null);
      library.reload();
    } catch (e) {
      setError(e);
      setRemove(null);
    } finally {
      setRemoving(false);
    }
  };

  const retryTask = (t: BundleTask) =>
    t.upload_id &&
    setLoad({
      uploadId: t.upload_id,
      label: t.key?.split('/').pop() ?? 'the staged zip',
      info: { ...unknownInfo(t.job_name ?? ''), name: t.job_name ?? '', datasets: t.dataset ? [t.dataset] : [] },
    });

  const busyCount = transfers.length + orphans.length + activeTasks.length;
  const items = library.data?.items ?? [];

  return (
    <>
      <div
        className={over ? 'dropzone dropzone-row is-over' : 'dropzone dropzone-row'}
        onDragOver={e => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={e => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files);
        }}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
      >
        <input
          ref={input}
          type="file"
          accept=".zip,application/zip"
          className="visually-hidden"
          aria-label="Choose a run bundle zip"
          onClick={e => e.stopPropagation()}
          onChange={e => take(e.target.files)}
        />
        <span className="dropzone-icon">
          <Upload aria-hidden />
        </span>
        <div className="dropzone-text">
          <p className="dropzone-title">Upload from computer</p>
          <p className="dropzone-hint">Drop a run bundle (.zip) here or click to choose. Big files go up in pieces and pick up where they stopped after a reload.</p>
        </div>
      </div>
      <ErrorNote error={error} />

      {busyCount > 0 && (
        <Section title="In progress">
          <ul className="task-list">
            {transfers.map(t => (
              <TransferRow key={t.key} t={t} server={server} onLoad={() => openLoadForTransfer(t)} onDiscard={() => (t.upload ? discard(t.upload) : dropTransfer(t.key))} />
            ))}
            {orphans.map(u => (
              <li key={u.id} className="card task">
                <div className="task-main">
                  <p className="task-title">
                    <FileArchive aria-hidden />
                    <span className="ellipsis">{u.filename}</span>
                  </p>
                  {u.complete ? (
                    <p className="task-meta">Uploaded {bytes(u.size)} · ready to load</p>
                  ) : (
                    <>
                      <Progress value={u.received} max={u.size} />
                      <p className="task-meta num">
                        Paused at {bytes(u.received)} of {bytes(u.size)}. Choose the same file again to continue.
                      </p>
                    </>
                  )}
                </div>
                <div className="task-actions">
                  {u.complete ? (
                    <button className="btn btn-sm btn-primary" onClick={() => setLoad({ uploadId: u.id, label: u.filename, size: u.size, info: unknownInfo() })}>
                      Load onto server
                    </button>
                  ) : (
                    <button className="btn btn-sm" onClick={() => input.current?.click()}>
                      <Play />
                      Resume
                    </button>
                  )}
                  <button className="btn btn-sm btn-ghost btn-danger" onClick={() => discard(u)}>
                    Discard
                  </button>
                </div>
              </li>
            ))}
            {activeTasks.map(t => (
              <ImportRow key={t.id} t={t} staged={staged(t)} onRetry={() => retryTask(t)} />
            ))}
          </ul>
        </Section>
      )}

      <Section
        title="Library"
        description={remote ? <>Run bundles in the bundles folder of {remote}. Save one from a run's Export menu.</> : undefined}
        actions={
          list.length > 1 && (
            <label className="field field-inline">
              <span>Remote</span>
              <select value={remote} onChange={e => setPicked(e.target.value)}>
                {list.map(r => (
                  <option key={r.id} value={r.name}>
                    {r.name} ({r.kind})
                  </option>
                ))}
              </select>
            </label>
          )
        }
      >
        <ErrorNote error={remotes.error || library.error} />
        {remotes.data && !list.length && (
          <Empty icon={<Archive />}>
            A library lives in a remote: an S3 bucket, a Hugging Face repo or a shared folder. Add one in the console under Remotes (or with AITK_S3_* on the server), then save runs to it.
          </Empty>
        )}
        {remote && library.data && !items.length && (
          <Empty icon={<Archive />}>Nothing in {remote} yet. Open a run, choose Export, then Save to library.</Empty>
        )}
        {items.length > 0 && (
          <ul className="bundle-grid">
            {items.map(b => (
              <BundleCard
                key={b.id}
                b={b}
                onLoad={() => setLoad({ remote, key: b.key, label: remote, size: b.size, info: infoFromSummary(b.summary, b.id) })}
                onDelete={() => setRemove(b)}
              />
            ))}
          </ul>
        )}
      </Section>

      {recentTasks.length > 0 && (
        <Section title="Recent loads">
          <ul className="task-list">
            {recentTasks.map(t => (
              <ImportRow key={t.id} t={t} staged={staged(t)} onRetry={() => retryTask(t)} />
            ))}
          </ul>
        </Section>
      )}

      {load && (
        <LoadDialog
          server={server}
          source={load}
          tasks={tasks}
          onClose={() => setLoad(null)}
          onLoaded={() => sessions.reload()}
        />
      )}
      {remove && (
        <Confirm title="Delete from the library" confirm="Delete" busy={removing} onConfirm={doRemove} onClose={() => setRemove(null)}>
          Delete <strong>{remove.summary?.name ?? remove.id}</strong> ({bytes(remove.size)}) from {remote}? Runs already loaded on a server are not affected.
        </Confirm>
      )}
    </>
  );
}

function BundleCard({ b, onLoad, onDelete }: { b: LibraryBundle; onLoad: () => void; onDelete: () => void }) {
  const s = b.summary;
  const kind = resumability(s);
  return (
    <li className="card bundle">
      <div className="bundle-head">
        <div className="bundle-titles">
          <p className="bundle-name">{s?.name ?? b.id}</p>
          <p className="bundle-id mono" title={b.key}>
            {b.id}
          </p>
        </div>
        {kind && (
          <span className={kind === 'resumable' ? 'badge tone-ok' : 'badge tone-quiet'} title={resumableHint[kind]}>
            {resumableLabel[kind]}
          </span>
        )}
      </div>
      <dl className="bundle-facts">
        <div>
          <dt>Step</dt>
          <dd className="num">
            {s ? s.step.toLocaleString() : '–'}
            {s?.total_steps ? <span className="of"> / {s.total_steps.toLocaleString()}</span> : null}
          </dd>
        </div>
        <div>
          <dt>Images</dt>
          <dd className="num">{s ? s.images.toLocaleString() : '–'}</dd>
        </div>
        <div>
          <dt>Size</dt>
          <dd className="num">{bytes(b.size)}</dd>
        </div>
        <div>
          <dt>Saved</dt>
          <dd title={new Date(s?.created_at ?? b.modified).toLocaleString()}>{ago(s?.created_at ?? b.modified)}</dd>
        </div>
        <div className="fact-wide">
          <dt>Base model</dt>
          <dd className="mono" title={s?.base_model}>
            {s?.base_model || '–'}
          </dd>
        </div>
      </dl>
      {!s && <p className="quiet-note">No summary next to this zip; its details show once it is loaded.</p>}
      <div className="bundle-actions">
        <button className="btn btn-primary btn-sm" onClick={onLoad}>
          Load onto server
        </button>
        <button className="btn btn-ghost btn-sm btn-danger" onClick={onDelete} aria-label={`Delete ${s?.name ?? b.id}`}>
          <Trash2 />
          Delete
        </button>
      </div>
    </li>
  );
}

function TransferRow({ t, server, onLoad, onDiscard }: { t: Transfer; server: Server; onLoad: () => void; onDiscard: () => void }) {
  const left = t.rate > 0 ? (t.file.size - t.sent) / t.rate : NaN;
  const pct = t.file.size ? Math.floor((t.sent / t.file.size) * 100) : 0;
  return (
    <li className="card task">
      <div className="task-main">
        <p className="task-title">
          <FileArchive aria-hidden />
          <span className="ellipsis">{t.file.name}</span>
          {t.state === 'paused' && <span className="badge tone-quiet">Paused</span>}
          {t.state === 'failed' && <span className="badge tone-danger">Failed</span>}
        </p>
        {t.state === 'complete' ? (
          <p className="task-meta">Uploaded {bytes(t.file.size)} · ready to load</p>
        ) : (
          <>
            <Progress value={t.sent} max={t.file.size} />
            <p className="task-meta num">
              {bytes(t.sent)} of {bytes(t.file.size)} · {pct}%
              {t.state === 'uploading' && t.rate > 0 && (
                <>
                  {' '}
                  · {bytes(t.rate)}/s · {duration(left)} left
                </>
              )}
              {t.state === 'starting' && ' · starting'}
            </p>
          </>
        )}
        {t.error && <p className="task-error">{t.error}</p>}
      </div>
      <div className="task-actions">
        {t.state === 'complete' && (
          <button className="btn btn-sm btn-primary" onClick={onLoad}>
            Load onto server
          </button>
        )}
        {(t.state === 'uploading' || t.state === 'starting') && (
          <button className="btn btn-sm" onClick={() => pauseUpload(t.key)}>
            <Pause />
            Pause
          </button>
        )}
        {(t.state === 'paused' || t.state === 'failed') && (
          <button className="btn btn-sm" onClick={() => startUpload(server, t.file, () => onLoad())}>
            <Play />
            Resume
          </button>
        )}
        <button className="btn btn-sm btn-ghost btn-icon" onClick={onDiscard} aria-label={`Discard ${t.file.name}`} title="Discard">
          <X />
        </button>
      </div>
    </li>
  );
}

function ImportRow({ t, staged, onRetry }: { t: BundleTask; staged: boolean; onRetry: () => void }) {
  const title = t.job_name ?? t.key?.split('/').pop() ?? (t.source === 'upload' ? 'Uploaded bundle' : (t.url ?? 'Bundle'));
  const from = t.source === 'remote' ? `from ${t.remote}` : t.source === 'url' ? 'from a link' : t.source === 'upload' ? 'from an upload' : 'at boot';
  return (
    <li className="card task">
      <div className="task-main">
        <p className="task-title">
          <span className="ellipsis">{title}</span>
          <StatePill state={t.state === 'done' ? 'completed' : t.state} label={t.state === 'done' ? 'Loaded' : t.state === 'failed' ? 'Failed' : t.state === 'queued' ? 'Queued' : 'Loading'} />
        </p>
        {isActive(t) ? (
          <TaskProgress task={t} />
        ) : (
          <p className="task-meta">
            {from} · {ago(t.updated_at)}
            {t.state === 'done' && t.dataset ? ` · dataset ${t.dataset}` : ''}
          </p>
        )}
        {t.error && <p className="task-error">{t.error.message}</p>}
      </div>
      <div className="task-actions">
        {t.state === 'done' && t.job_id && (
          <a className="btn btn-sm" href={href({ page: 'run', id: t.job_id })}>
            Open run
            <ArrowRight />
          </a>
        )}
        {t.state === 'failed' && staged && (
          <button className="btn btn-sm" onClick={onRetry}>
            {t.error?.code === 'conflict' || t.error?.code === 'dataset_exists' ? 'Pick another name' : 'Try again'}
          </button>
        )}
      </div>
    </li>
  );
}

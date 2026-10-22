import { ArrowRight } from 'lucide-react';
import { useEffect, useState } from 'react';
import { ApiError, get, post, type BundleTask, type Server, type SystemInfo } from '../lib/api';
import { dropTransfersFor, phaseLabel, resumableHint, resumableLabel, type BundleInfo, type useTasks } from '../lib/bundles';
import { bytes } from '../lib/format';
import { go } from '../lib/route';
import { Dialog } from './Dialog';
import { Callout, Progress, useLoad } from './ui';

/** Where the bundle comes from: a library object, or a staged upload. */
export type LoadSource = {
  label: string;
  info: BundleInfo;
  size?: number;
} & ({ remote: string; key: string; uploadId?: undefined } | { uploadId: string; remote?: undefined; key?: undefined });

type Tasks = ReturnType<typeof useTasks>;

const conflictCodes = ['conflict', 'dataset_exists'];

/**
 * "Load onto server": names the job and dataset, picks the GPU, imports and
 * follows the task. A name or dataset clash lets the user pick another and
 * retries from the staged zip (the task's upload_id), with no second
 * download.
 */
export function LoadDialog({ server, source, tasks, onClose, onLoaded }: { server: Server; source: LoadSource; tasks: Tasks; onClose: () => void; onLoaded?: () => void }) {
  const info = source.info;
  const [name, setName] = useState(info.name);
  const [dataset, setDataset] = useState(info.datasets[0] ?? '');
  const [gpu, setGpu] = useState('0');
  const [start, setStart] = useState(info.resumable !== 'dataset');
  const [uploadId, setUploadId] = useState<string | null>(source.uploadId ?? null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nameError, setNameError] = useState<string | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const system = useLoad(server, s => get<SystemInfo>(s, '/v1/system').catch(() => null), []);
  const gpus = system.data?.gpus ?? [];

  const task = taskId ? (tasks.tasks?.find(t => t.id === taskId) ?? null) : null;

  const explain = (code: string, message: string) => {
    if (code === 'conflict') setNameError(`A run named ${name.trim() || info.guess || 'like this bundle'} is already on this server. Pick another name.`);
    else if (code === 'dataset_exists') setDatasetError(`A dataset named ${dataset.trim() || info.datasets[0] || name} already holds other files. Pick another name.`);
    else if (code === 'too_large') setError(`Not enough disk space on the server: ${message}`);
    else setError(message);
  };

  // a failed task with a clash goes back to the form, keeping its staged zip
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
    if (task?.state !== 'done') return;
    if (uploadId) dropTransfersFor(uploadId);
    onLoaded?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task?.state]);

  const submit = async () => {
    setBusy(true);
    setError(null);
    setNameError(null);
    setDatasetError(null);
    const from = uploadId ? { upload_id: uploadId } : { remote: source.remote, key: source.key };
    try {
      const t = await post<BundleTask>(server, '/v1/bundles/imports', {
        ...from,
        name: name.trim() || undefined,
        dataset: dataset.trim() || undefined,
        gpu_ids: gpu,
        start,
      });
      tasks.upsert(t);
      setTaskId(t.id);
    } catch (e) {
      if (e instanceof ApiError) explain(e.code, e.message);
      else setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const running = task && task.state !== 'failed' && task.state !== 'done';
  const failed = task?.state === 'failed';
  const done = task?.state === 'done';

  const footer = done ? (
    <>
      <button className="btn btn-ghost" onClick={onClose}>
        Close
      </button>
      {task.job_id && (
        <button className="btn btn-primary" onClick={() => go({ page: 'run', id: task.job_id! })}>
          Open run
          <ArrowRight />
        </button>
      )}
    </>
  ) : running ? (
    <button className="btn" onClick={onClose}>
      Continue in the background
    </button>
  ) : (
    <>
      <button className="btn btn-ghost" onClick={onClose}>
        Cancel
      </button>
      <button className="btn btn-primary" onClick={submit} disabled={busy}>
        {failed ? 'Try again' : busy ? 'Starting…' : 'Load onto server'}
      </button>
    </>
  );

  return (
    <Dialog title="Load onto server" description={<>Restores the dataset and the run folder from {source.label}.</>} onClose={onClose} footer={footer}>
      <dl className="bundle-brief">
        <div>
          <dt>Run</dt>
          <dd>{info.guess || source.label}</dd>
        </div>
        <div>
          <dt>Step</dt>
          <dd className="num">
            {info.step != null ? info.step.toLocaleString() : '–'}
            {info.total ? <span className="of"> / {info.total.toLocaleString()}</span> : null}
          </dd>
        </div>
        {info.images != null && (
          <div>
            <dt>Images</dt>
            <dd className="num">{info.images.toLocaleString()}</dd>
          </div>
        )}
        {source.size != null && (
          <div>
            <dt>Size</dt>
            <dd className="num">{bytes(source.size)}</dd>
          </div>
        )}
      </dl>
      {info.resumable && (
        <p className="bundle-kind">
          <span className={info.resumable === 'resumable' ? 'badge tone-ok' : 'badge tone-quiet'}>{resumableLabel[info.resumable]}</span>
          <span>{resumableHint[info.resumable]}</span>
        </p>
      )}

      {done ? (
        <Callout kind="ok">
          <p>
            Loaded <strong>{task.job_name}</strong>
            {task.dataset ? (
              <>
                {' '}
                with the dataset <strong>{task.dataset}</strong>
              </>
            ) : null}
            .{start ? ' It is queued and continues from its newest checkpoint.' : ' Start it from the run page when you are ready.'}
          </p>
          {!!task.skipped?.length && <p className="muted">Skipped {task.skipped.length} junk files.</p>}
          {task.warnings?.map(w => (
            <p key={w} className="muted">
              {w}
            </p>
          ))}
        </Callout>
      ) : running ? (
        <TaskProgress task={task} />
      ) : (
        <form
          className="dialog-form"
          onSubmit={e => {
            e.preventDefault();
            submit();
          }}
        >
          <label className="field">
            <span>Run name</span>
            <input value={name} onChange={e => setName(e.target.value)} placeholder={info.guess || 'The bundle’s own name'} aria-invalid={!!nameError} />
            {nameError ? <small className="field-error">{nameError}</small> : <small>Checkpoints are renamed to match, so resuming still finds them.</small>}
          </label>
          <label className="field">
            <span>Dataset name</span>
            <input value={dataset} onChange={e => setDataset(e.target.value)} placeholder={info.datasets[0] ?? (name || info.guess || 'The bundle’s own name')} aria-invalid={!!datasetError} />
            {datasetError ? <small className="field-error">{datasetError}</small> : <small>A dataset with the same files is reused.</small>}
          </label>
          <label className="field">
            <span>GPU</span>
            <select value={gpu} onChange={e => setGpu(e.target.value)}>
              {gpus.length ? (
                gpus.map(g => (
                  <option key={g.index} value={String(g.index)}>
                    GPU {g.index} · {g.name}
                  </option>
                ))
              ) : (
                <option value="0">GPU 0</option>
              )}
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={start} onChange={e => setStart(e.target.checked)} />
            {info.resumable === 'dataset' ? 'Start training now' : 'Continue training now'}
          </label>
          {failed && task.error && !conflictCodes.includes(task.error.code) && <Callout kind="danger">{task.error.message}</Callout>}
          {error && <Callout kind="danger">{error}</Callout>}
          {uploadId && !source.uploadId && <p className="quiet-note">The zip is already on the server; trying again does not download it again.</p>}
          <button type="submit" hidden />
        </form>
      )}
    </Dialog>
  );
}

/** Phase, bar and bytes of one task. */
export function TaskProgress({ task }: { task: BundleTask }) {
  const known = task.bytes > 0;
  return (
    <div className="task-progress" aria-live="polite">
      <div className="task-progress-head">
        <span>{phaseLabel(task)}</span>
        <span className="num muted">
          {known ? `${bytes(task.bytes_done)} of ${bytes(task.bytes)}` : task.bytes_done ? bytes(task.bytes_done) : ''}
        </span>
      </div>
      <Progress value={known ? task.bytes_done : 0} max={known ? task.bytes : 1} indeterminate={!known} />
    </div>
  );
}

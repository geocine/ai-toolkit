import { Download, UploadCloud } from 'lucide-react';
import { useEffect, useState } from 'react';
import { ApiError, get, post, signedDownload, url, type BundleTask, type Items, type Job, type Remote, type Server } from '../lib/api';
import { isActive, useTasks } from '../lib/bundles';
import { ago, bytes } from '../lib/format';
import { Dialog } from './Dialog';
import { TaskProgress } from './LoadDialog';
import { Callout, useLoad } from './ui';

type Ckpts = 'latest' | 'all' | 'none';

const ckptOptions: { value: Ckpts; label: string }[] = [
  { value: 'latest', label: 'Latest' },
  { value: 'all', label: 'All' },
  { value: 'none', label: 'None' },
];

/**
 * Export a run as a bundle: download the zip (a signed link, streamed to
 * disk) or save it to a remote's library, with the push followed live.
 */
export function ExportDialog({ server, job, onClose }: { server: Server; job: Job; onClose: () => void }) {
  const [checkpoints, setCheckpoints] = useState<Ckpts>('latest');
  const [optimizer, setOptimizer] = useState(true);
  const [samples, setSamples] = useState(true);
  const [caches, setCaches] = useState(true);
  const [size, setSize] = useState<number | null>(null);
  const [remote, setRemote] = useState('');
  const [bundle, setBundle] = useState('');
  const [overwrite, setOverwrite] = useState(false);
  const [exists, setExists] = useState(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState<unknown>(null);
  const [pushId, setPushId] = useState<string | null>(null);
  const remotes = useLoad(server, s => get<Items<Remote>>(s, '/v1/remotes').catch(() => ({ items: [] as Remote[] })), []);
  const pushes = useTasks(server, 'push', job.id);
  const list = remotes.data?.items ?? [];
  const target = remote || list[0]?.name || '';

  const query = new URLSearchParams({
    checkpoints,
    optimizer: checkpoints !== 'none' && optimizer ? '1' : '0',
    samples: samples ? '1' : '0',
    caches: caches ? '1' : '0',
  }).toString();
  const path = `/v1/jobs/${encodeURIComponent(job.id)}/bundle?${query}`;

  // the zip is stored, so the server knows its exact size up front (HEAD)
  useEffect(() => {
    setSize(null);
    const ctl = new AbortController();
    const t = setTimeout(() => {
      fetch(url(server, path), { method: 'HEAD', headers: server.token ? { Authorization: `Bearer ${server.token}` } : {}, signal: ctl.signal })
        .then(r => (r.ok ? Number(r.headers.get('Content-Length')) : NaN))
        .then(n => setSize(Number.isFinite(n) && n > 0 ? n : null), () => undefined);
    }, 200);
    return () => {
      clearTimeout(t);
      ctl.abort();
    };
  }, [server, path]);

  const defaultName = `${job.name}_step${job.latest_checkpoint?.step ?? job.step}`;

  const downloadZip = async () => {
    setBusy('download');
    setError(null);
    try {
      await signedDownload(server, path);
    } catch (e) {
      setError(e);
    } finally {
      setBusy('');
    }
  };

  const push = async () => {
    setBusy('push');
    setError(null);
    try {
      const t = await post<BundleTask>(server, `/v1/jobs/${encodeURIComponent(job.id)}/bundle/push`, {
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
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) setExists(true);
      else setError(e);
    } finally {
      setBusy('');
    }
  };

  const current = pushId ? pushes.tasks?.find(t => t.id === pushId) : pushes.tasks?.find(isActive);
  const recent = pushes.tasks?.find(t => t.state === 'done');
  const clash = current?.state === 'failed' && current.error?.code === 'conflict';

  // the server may only find the clash once the task runs
  useEffect(() => {
    if (clash) setExists(true);
  }, [clash]);

  return (
    <Dialog title="Export run" description="One zip with the dataset and the run folder, in the layout ai-toolkit resumes from." onClose={onClose}>
      <div className="dialog-form">
        <div className="field">
          <span>Checkpoints</span>
          <div className="seg" role="radiogroup" aria-label="Checkpoints">
            {ckptOptions.map(o => (
              <button key={o.value} type="button" role="radio" aria-checked={checkpoints === o.value} className={checkpoints === o.value ? 'is-on' : ''} onClick={() => setCheckpoints(o.value)}>
                {o.label}
              </button>
            ))}
          </div>
          <small>
            {checkpoints === 'latest'
              ? 'The newest save, enough to continue training.'
              : checkpoints === 'all'
                ? 'Every save, to compare steps later.'
                : 'No weights: a dataset-only preset that trains from step 0.'}
          </small>
        </div>
        <div className="checks">
          <label className="check check-hint">
            <input type="checkbox" checked={checkpoints !== 'none' && optimizer} disabled={checkpoints === 'none'} onChange={e => setOptimizer(e.target.checked)} />
            <span>
              Optimizer state
              <small>Needed to resume exactly; without it training continues from the weights.</small>
            </span>
          </label>
          <label className="check check-hint">
            <input type="checkbox" checked={samples} onChange={e => setSamples(e.target.checked)} />
            <span>
              Samples
              <small>The images rendered during training.</small>
            </span>
          </label>
          <label className="check check-hint">
            <input type="checkbox" checked={caches} onChange={e => setCaches(e.target.checked)} />
            <span>
              Dataset caches
              <small>Latents and text embeddings, so the next server skips re-caching.</small>
            </span>
          </label>
        </div>

        <div className="export-size">
          <span className="muted">Zip size</span>
          <span className="num">{size ? bytes(size) : '…'}</span>
        </div>

        <div className="export-actions">
          <button className="btn" onClick={downloadZip} disabled={busy === 'download'}>
            <Download />
            {busy === 'download' ? 'Preparing…' : 'Download zip'}
          </button>
        </div>

        <div className="export-library">
          <h3>Save to library</h3>
          {list.length === 0 ? (
            <p className="quiet-note">No remote on this server. Add one in the console under Remotes to keep a library.</p>
          ) : (
            <>
              <div className="export-grid">
                {list.length > 1 && (
                  <label className="field">
                    <span>Remote</span>
                    <select value={target} onChange={e => setRemote(e.target.value)}>
                      {list.map(r => (
                        <option key={r.id} value={r.name}>
                          {r.name} ({r.kind})
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <label className="field">
                  <span>Name in the library</span>
                  <input
                    value={bundle}
                    placeholder={defaultName}
                    onChange={e => {
                      setBundle(e.target.value);
                      setExists(false);
                    }}
                  />
                </label>
              </div>
              {exists && (
                <Callout kind="warn">
                  <p>{bundle.trim() || defaultName} is already in the library.</p>
                  <label className="check">
                    <input type="checkbox" checked={overwrite} onChange={e => setOverwrite(e.target.checked)} />
                    Replace it
                  </label>
                </Callout>
              )}
              {current && isActive(current) ? (
                <TaskProgress task={current} />
              ) : (
                <div className="export-actions">
                  <button className="btn btn-primary" onClick={push} disabled={busy === 'push' || (exists && !overwrite)}>
                    <UploadCloud />
                    {busy === 'push' ? 'Starting…' : `Save to ${target}`}
                  </button>
                </div>
              )}
              {current?.state === 'failed' && current.error && !clash && <Callout kind="danger">{current.error.message}</Callout>}
              {current?.state === 'done' ? (
                <Callout kind="ok">
                  Saved as <span className="mono">{current.key?.split('/').pop()}</span> in {current.remote}.
                </Callout>
              ) : (
                recent &&
                !current && (
                  <p className="quiet-note">
                    Last saved {ago(recent.updated_at)} as <span className="mono">{recent.key?.split('/').pop()}</span>.
                  </p>
                )
              )}
            </>
          )}
        </div>
        {error ? <Callout kind="danger">{error instanceof Error ? error.message : String(error)}</Callout> : null}
      </div>
    </Dialog>
  );
}

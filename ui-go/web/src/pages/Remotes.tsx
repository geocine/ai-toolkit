import { useEffect, useState } from 'react';
import { CheckCircle2, Cloud, DownloadCloud, Plus, RefreshCw, Trash2, XCircle } from 'lucide-react';
import { api, fmtAgo, fmtBytes, fmtTime } from '../api';
import { go } from '../router';
import type { Items, Job, PortableJob, Remote, RemoteKind, Upload } from '../types';
import { Bar, Empty, Field, Loading, Modal, Mono, Section, Spinner, StateBadge, usePoll, useConfirm, useToast } from '../ui';

export function Remotes() {
  const toast = useToast();
  const confirm = useConfirm();
  const [remotes, setRemotes] = useState<Remote[] | null>(null);
  const [uploads, setUploads] = useState<Upload[] | null>(null);
  const [creating, setCreating] = useState(false);
  const [tests, setTests] = useState<Record<string, { ok: boolean; detail: string } | 'busy'>>({});
  const [browse, setBrowse] = useState<string>('');

  const refresh = usePoll(async signal => {
    const [r, u] = await Promise.all([api.get<Items<Remote>>('/v1/remotes', signal), api.get<Items<Upload>>('/v1/uploads', signal)]);
    setRemotes(r.items);
    setUploads(u.items);
  }, 5000);

  useEffect(() => {
    if (!browse && remotes?.length) setBrowse(remotes[0].id);
  }, [remotes, browse]);

  const test = async (r: Remote) => {
    setTests(t => ({ ...t, [r.id]: 'busy' }));
    try {
      const res = await api.post<{ ok: boolean; detail: string }>(`/v1/remotes/${encodeURIComponent(r.id)}/test`);
      setTests(t => ({ ...t, [r.id]: res }));
    } catch (e: any) {
      setTests(t => ({ ...t, [r.id]: { ok: false, detail: e?.message ?? String(e) } }));
    }
  };

  const del = async (r: Remote) => {
    if (!(await confirm({ title: 'Remove remote', body: <>Remove <b>{r.name}</b>? Files already on it are kept.</>, confirm: 'Remove', danger: true }))) return;
    try {
      await api.del(`/v1/remotes/${encodeURIComponent(r.id)}`);
      toast.success('Remote removed');
      refresh();
    } catch (e) {
      toast.error(e);
    }
  };

  const remoteName = (id: string) => remotes?.find(r => r.id === id)?.name ?? id;

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Remotes</h1>
          <div className="page-sub">Storage outside this machine: checkpoint uploads and portable jobs.</div>
        </div>
        <button className="btn btn-primary" onClick={() => setCreating(true)}>
          <Plus size={12} /> Add remote
        </button>
      </div>

      <Section title="Configured">
        {!remotes ? (
          <Loading />
        ) : remotes.length === 0 ? (
          <Empty icon={<Cloud size={28} />} title="No remotes">
            Add Cloudflare R2 / S3, a private Hugging Face repo or a shared folder, or set AITK_R2_* / AITK_HF_REPO in the server's environment.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Kind</th>
                  <th>Target</th>
                  <th>Check</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {remotes.map(r => {
                  const t = tests[r.id];
                  return (
                    <tr key={r.id}>
                      <td style={{ fontWeight: 700 }}>{r.name}</td>
                      <td>
                        <span className="badge">{r.kind}</span>
                      </td>
                      <td className="mono small ellipsis" style={{ maxWidth: 360 }}>
                        {target(r)}
                      </td>
                      <td className="small">
                        {t === 'busy' ? (
                          <Spinner size={12} />
                        ) : t ? (
                          <span className={t.ok ? 'ok-text row' : 'danger-text row'} title={t.detail}>
                            {t.ok ? <CheckCircle2 size={12} /> : <XCircle size={12} />} {t.ok ? 'ok' : t.detail}
                          </span>
                        ) : (
                          <span className="dim">—</span>
                        )}
                      </td>
                      <td className="actions">
                        <div className="row" style={{ justifyContent: 'flex-end' }}>
                          <button className="btn btn-sm" onClick={() => test(r)}>
                            Test
                          </button>
                          <button className="icon-btn" onClick={() => del(r)} aria-label="Remove" title="Remove">
                            <Trash2 size={12} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {remotes && remotes.length > 0 && (
        <Section
          title="Portable jobs"
          actions={
            <select value={browse} onChange={e => setBrowse(e.target.value)} style={{ width: 200 }}>
              {remotes.map(r => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          }
        >
          {browse && <PortableJobs remote={browse} />}
        </Section>
      )}

      <Section title="Uploads">
        {!uploads ? (
          <Loading />
        ) : uploads.length === 0 ? (
          <Empty title="No uploads">Checkpoint uploads (manual or a job's auto-upload policy) show up here.</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Remote</th>
                  <th>State</th>
                  <th style={{ minWidth: 140 }}>Progress</th>
                  <th>Updated</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {uploads.map(u => (
                  <tr key={u.id}>
                    <td>
                      <div className="mono small ellipsis" style={{ maxWidth: 380 }} title={u.path}>
                        {u.path.split(/[\\/]/).pop()}
                      </div>
                      {u.error && <div className="danger-text small">{u.error}</div>}
                      {u.url && (
                        <div className="dim small mono ellipsis" style={{ maxWidth: 380 }} title={u.url}>
                          {u.url}
                        </div>
                      )}
                    </td>
                    <td>{remoteName(u.remote_id)}</td>
                    <td>
                      <StateBadge state={u.state === 'failed' ? 'failed' : u.state === 'running' ? 'running' : u.state === 'done' ? 'completed' : 'queued'} />
                    </td>
                    <td>
                      <div className="mono small">
                        {fmtBytes(u.bytes_done)} / {fmtBytes(u.bytes)}
                      </div>
                      <Bar value={u.bytes_done} max={u.bytes} />
                    </td>
                    <td className="small muted nowrap">{fmtAgo(u.updated_at)}</td>
                    <td className="actions">
                      {(u.state === 'failed' || u.state === 'done') && (
                        <button
                          className="btn btn-sm"
                          onClick={() =>
                            api.post(`/v1/uploads/${encodeURIComponent(u.id)}/retry`).then(() => {
                              toast.success('Queued again');
                              refresh();
                            }, toast.error)
                          }
                        >
                          <RefreshCw size={12} /> Retry
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {creating && (
        <CreateRemote
          onClose={() => setCreating(false)}
          onCreated={() => {
            setCreating(false);
            refresh();
          }}
        />
      )}
    </>
  );
}

const target = (r: Remote) => {
  const c = r.config ?? {};
  if (r.kind === 's3') return `${c.endpoint ?? ''}/${c.bucket ?? ''}${c.prefix ? '/' + c.prefix : ''}`;
  if (r.kind === 'hf') return `hf.co/${c.repo_type === 'dataset' ? 'datasets/' : ''}${c.repo_id ?? ''}${c.path_in_repo ? '/' + c.path_in_repo : ''}`;
  return c.path ?? '';
};

function PortableJobs({ remote }: { remote: string }) {
  const toast = useToast();
  const confirm = useConfirm();
  const [items, setItems] = useState<PortableJob[] | null>(null);
  const [error, setError] = useState('');
  const [importing, setImporting] = useState('');

  useEffect(() => setItems(null), [remote]);
  const refresh = usePoll(
    async signal => {
      try {
        setItems((await api.get<Items<PortableJob>>(`/v1/portable/jobs?remote=${encodeURIComponent(remote)}`, signal)).items);
        setError('');
      } catch (e: any) {
        if (e?.name !== 'AbortError') setError(e?.message ?? String(e));
        setItems([]);
      }
    },
    60000,
    [remote],
  );

  const doImport = async (pj: PortableJob) => {
    const held = pj.lease && !pj.lease.expired && new Date(pj.lease.expires_at).getTime() > Date.now();
    let force = false;
    if (held) {
      force = await confirm({
        title: 'Take over the lease',
        body: (
          <>
            <b>{pj.name}</b> is leased by <span className="mono">{pj.lease!.node_label}</span> until {fmtTime(pj.lease!.expires_at)}. Only take it over if that
            server is gone; otherwise both would train it.
          </>
        ),
        confirm: 'Force import',
        danger: true,
      });
      if (!force) return;
    }
    setImporting(pj.name);
    try {
      const job = await api.post<Job>('/v1/portable/import', { remote, name: pj.name, start: true, force });
      toast.success(`Imported ${job.name}; continuing from step ${job.step}`);
      go('jobs', job.id);
    } catch (e) {
      toast.error(e);
      refresh();
    } finally {
      setImporting('');
    }
  };

  if (error) return <div className="danger-text small">{error}</div>;
  if (!items) return <Loading />;
  if (items.length === 0) return <Empty title="No portable jobs on this remote">Jobs with this remote as their sync remote appear here.</Empty>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Job</th>
            <th>State</th>
            <th>Step</th>
            <th>Lease</th>
            <th>Updated</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {items.map(pj => {
            const held = pj.lease && new Date(pj.lease.expires_at).getTime() > Date.now();
            return (
              <tr key={pj.name}>
                <td>
                  <div style={{ fontWeight: 700 }}>{pj.name}</div>
                  <div className="dim small">{pj.datasets.join(', ')}</div>
                </td>
                <td>
                  <StateBadge state={pj.state} />
                </td>
                <td className="mono">{pj.step}</td>
                <td className="small">
                  {held ? (
                    <>
                      <Mono>{pj.lease!.node_label}</Mono> <span className="muted">until {fmtTime(pj.lease!.expires_at)}</span>
                    </>
                  ) : (
                    <span className="muted">free</span>
                  )}
                </td>
                <td className="small muted nowrap">{fmtAgo(pj.updated_at)}</td>
                <td className="actions">
                  <button className="btn btn-sm" onClick={() => doImport(pj)} disabled={!!importing}>
                    {importing === pj.name ? <Spinner size={12} /> : <DownloadCloud size={12} />} Import & start
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CreateRemote({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const toast = useToast();
  const [kind, setKind] = useState<RemoteKind>('s3');
  const [name, setName] = useState('r2');
  const [f, setF] = useState<Record<string, string>>({ region: 'auto', repo_type: 'model' });
  const [pathStyle, setPathStyle] = useState(false);
  const [priv, setPriv] = useState(true);
  const [busy, setBusy] = useState(false);
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setF({ ...f, [k]: e.target.value });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    let config: Record<string, unknown>;
    let credentials: Record<string, string> | undefined;
    if (kind === 's3') {
      config = { endpoint: f.endpoint, bucket: f.bucket, region: f.region || 'auto', prefix: f.prefix || '', path_style: pathStyle };
      credentials = { access_key_id: f.access_key_id ?? '', secret_access_key: f.secret_access_key ?? '' };
    } else if (kind === 'hf') {
      config = { repo_id: f.repo_id, repo_type: f.repo_type || 'model', private: priv, path_in_repo: f.path_in_repo || '' };
      credentials = f.token ? { token: f.token } : undefined;
    } else {
      config = { path: f.path };
    }
    setBusy(true);
    try {
      await api.post('/v1/remotes', { name: name.trim(), kind, config, ...(credentials ? { credentials } : {}) });
      toast.success(`Added ${name}`);
      onCreated();
    } catch (err) {
      toast.error(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title="Add remote" onClose={onClose}>
      <form onSubmit={submit}>
        <div className="modal-body stack">
          <div className="form-grid">
            <Field label="Name">
              <input type="text" value={name} onChange={e => setName(e.target.value)} required />
            </Field>
            <Field label="Kind">
              <select
                value={kind}
                onChange={e => {
                  const k = e.target.value as RemoteKind;
                  setKind(k);
                  setName(k === 's3' ? 'r2' : k === 'hf' ? 'hf' : 'shared');
                }}
              >
                <option value="s3">Cloudflare R2 / S3</option>
                <option value="hf">Hugging Face repo</option>
                <option value="fs">Shared folder</option>
              </select>
            </Field>
          </div>
          {kind === 's3' && (
            <>
              <Field label="Endpoint" hint="https://<account>.r2.cloudflarestorage.com">
                <input type="text" value={f.endpoint ?? ''} onChange={set('endpoint')} required />
              </Field>
              <div className="form-grid">
                <Field label="Bucket">
                  <input type="text" value={f.bucket ?? ''} onChange={set('bucket')} required />
                </Field>
                <Field label="Region">
                  <input type="text" value={f.region ?? ''} onChange={set('region')} />
                </Field>
                <Field label="Prefix">
                  <input type="text" value={f.prefix ?? ''} onChange={set('prefix')} placeholder="optional" />
                </Field>
              </div>
              <div className="form-grid">
                <Field label="Access key ID">
                  <input type="text" value={f.access_key_id ?? ''} onChange={set('access_key_id')} required autoComplete="off" />
                </Field>
                <Field label="Secret access key">
                  <input type="password" value={f.secret_access_key ?? ''} onChange={set('secret_access_key')} required autoComplete="new-password" />
                </Field>
              </div>
              <label className="check small">
                <input type="checkbox" checked={pathStyle} onChange={e => setPathStyle(e.target.checked)} /> Path-style URLs (MinIO)
              </label>
            </>
          )}
          {kind === 'hf' && (
            <>
              <div className="form-grid">
                <Field label="Repo">
                  <input type="text" value={f.repo_id ?? ''} onChange={set('repo_id')} placeholder="user/my-loras" required />
                </Field>
                <Field label="Repo type">
                  <select value={f.repo_type} onChange={set('repo_type')}>
                    <option value="model">model</option>
                    <option value="dataset">dataset</option>
                  </select>
                </Field>
                <Field label="Path in repo">
                  <input type="text" value={f.path_in_repo ?? ''} onChange={set('path_in_repo')} placeholder="optional" />
                </Field>
              </div>
              <Field label="Token" hint="Empty: the server's HF_TOKEN setting">
                <input type="password" value={f.token ?? ''} onChange={set('token')} autoComplete="new-password" />
              </Field>
              <label className="check small">
                <input type="checkbox" checked={priv} onChange={e => setPriv(e.target.checked)} /> Private (created if missing)
              </label>
            </>
          )}
          {kind === 'fs' && (
            <Field label="Folder" hint="A path on the server, e.g. a mounted network volume">
              <input type="text" value={f.path ?? ''} onChange={set('path')} required />
            </Field>
          )}
          <div className="dim small">Credentials are stored on the server and never shown again.</div>
        </div>
        <div className="modal-actions">
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" disabled={busy}>
            {busy ? <Spinner size={12} /> : <Plus size={12} />} Add
          </button>
        </div>
      </form>
    </Modal>
  );
}

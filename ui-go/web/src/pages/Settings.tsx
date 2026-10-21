import { useEffect, useState } from 'react';
import { KeyRound, Plus, Save, Trash2 } from 'lucide-react';
import { api, fmtAgo, fmtTime } from '../api';
import type { ApiKey, Items, Settings } from '../types';
import { CopyButton, Empty, Field, Loading, Modal, Mono, Section, Spinner, usePoll, useConfirm, useToast } from '../ui';

export function SettingsPage() {
  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Settings</h1>
          <div className="page-sub">Server folders, job defaults and API keys.</div>
        </div>
      </div>
      <ServerSettings />
      <Keys />
    </>
  );
}

function ServerSettings() {
  const toast = useToast();
  const [s, setS] = useState<Settings | null>(null);
  const [form, setForm] = useState<Record<string, string | boolean>>({});
  const [hfToken, setHfToken] = useState('');
  const [saving, setSaving] = useState(false);

  const load = async () => {
    const v = await api.get<Settings>('/v1/settings');
    setS(v);
    setForm({ ...v, max_auto_resumes_default: String(v.max_auto_resumes_default) });
  };
  useEffect(() => {
    load().catch(toast.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      const body: Record<string, unknown> = {
        training_folder: form.training_folder,
        datasets_folder: form.datasets_folder,
        data_root: form.data_root,
        models_path: form.models_path,
        auto_resume_default: !!form.auto_resume_default,
        max_auto_resumes_default: Number(form.max_auto_resumes_default),
      };
      if (hfToken) body.hf_token = hfToken;
      const v = await api.patch<Settings>('/v1/settings', body);
      setS(v);
      setHfToken('');
      toast.success('Settings saved');
    } catch (err) {
      toast.error(err);
    } finally {
      setSaving(false);
    }
  };

  if (!s) return <Loading what="settings" />;
  const text = (k: keyof Settings) => (
    <input type="text" value={String(form[k] ?? '')} onChange={e => setForm({ ...form, [k]: e.target.value })} />
  );
  return (
    <Section title="Server">
      <form className="card" onSubmit={save}>
        <div className="form-grid">
          <Field label="Training folder">{text('training_folder')}</Field>
          <Field label="Datasets folder">{text('datasets_folder')}</Field>
          <Field label="Data root">{text('data_root')}</Field>
          <Field label="Models path" hint="MODELS_PATH in the server env wins over this">
            {text('models_path')}
          </Field>
          <Field label="Hugging Face token" hint={s.hf_token_set ? 'Set. Type to replace.' : 'Not set'}>
            <input type="password" value={hfToken} onChange={e => setHfToken(e.target.value)} placeholder={s.hf_token_set ? '••••••••' : 'hf_…'} autoComplete="new-password" />
          </Field>
          <Field label="Max auto-resumes (default)">
            <input type="number" min={0} value={String(form.max_auto_resumes_default ?? '')} onChange={e => setForm({ ...form, max_auto_resumes_default: e.target.value })} />
          </Field>
        </div>
        <label className="check" style={{ marginTop: 14 }}>
          <input type="checkbox" checked={!!form.auto_resume_default} onChange={e => setForm({ ...form, auto_resume_default: e.target.checked })} />
          Auto-resume interrupted jobs by default
        </label>
        <div className="form-actions">
          <button className="btn btn-primary" disabled={saving}>
            {saving ? <Spinner size={12} /> : <Save size={12} />} Save
          </button>
        </div>
      </form>
    </Section>
  );
}

function Keys() {
  const toast = useToast();
  const confirm = useConfirm();
  const [keys, setKeys] = useState<ApiKey[] | null>(null);
  const [name, setName] = useState('');
  const [creating, setCreating] = useState(false);
  const [secret, setSecret] = useState<{ name: string; key: string } | null>(null);

  const refresh = usePoll(async signal => setKeys((await api.get<Items<ApiKey>>('/v1/keys', signal)).items), 30000);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    try {
      const k = await api.post<ApiKey & { key: string }>('/v1/keys', { name: name.trim() });
      setSecret({ name: k.name, key: k.key });
      setName('');
      refresh();
    } catch (err) {
      toast.error(err);
    } finally {
      setCreating(false);
    }
  };

  const revoke = async (k: ApiKey) => {
    if (!(await confirm({ title: 'Revoke key', body: <>Revoke <b>{k.name}</b>? Clients using it lose access immediately.</>, confirm: 'Revoke', danger: true }))) return;
    try {
      await api.del(`/v1/keys/${encodeURIComponent(k.id)}`);
      toast.success('Key revoked');
      refresh();
    } catch (err) {
      toast.error(err);
    }
  };

  return (
    <Section title="API keys">
      <form className="row-wrap" onSubmit={create} style={{ marginBottom: 12 }}>
        <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="Key name, e.g. studio-laptop" style={{ maxWidth: 280 }} />
        <button className="btn btn-primary" disabled={creating || !name.trim()}>
          {creating ? <Spinner size={12} /> : <Plus size={12} />} Create key
        </button>
      </form>
      {!keys ? (
        <Loading />
      ) : keys.length === 0 ? (
        <Empty icon={<KeyRound size={28} />} title="No API keys">
          Give each app or agent its own key so it can be revoked on its own. Once a key exists, the server always requires a token.
        </Empty>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Prefix</th>
                <th>Created</th>
                <th>Last used</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {keys.map(k => (
                <tr key={k.id}>
                  <td style={{ fontWeight: 700 }}>{k.name}</td>
                  <td>
                    <Mono>{k.prefix}…</Mono>
                  </td>
                  <td className="small muted nowrap" title={fmtTime(k.created_at)}>
                    {fmtAgo(k.created_at)}
                  </td>
                  <td className="small muted nowrap">{k.last_used_at ? fmtAgo(k.last_used_at) : 'never'}</td>
                  <td className="actions">
                    <button className="btn btn-sm btn-ghost-danger" onClick={() => revoke(k)}>
                      <Trash2 size={12} /> Revoke
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {secret && (
        <Modal title={`Key "${secret.name}"`} onClose={() => setSecret(null)}>
          <div className="modal-body stack">
            <div className="warn-text small">Copy it now: it is not shown again.</div>
            <div className="secret">
              <span className="grow">{secret.key}</span>
              <CopyButton text={secret.key} />
            </div>
            <div className="muted small">
              Use it as <span className="mono">Authorization: Bearer {secret.key.slice(0, 10)}…</span> for /v1, /mcp and this console.
            </div>
          </div>
          <div className="modal-actions">
            <button className="btn btn-primary" onClick={() => setSecret(null)}>
              Done
            </button>
          </div>
        </Modal>
      )}
    </Section>
  );
}

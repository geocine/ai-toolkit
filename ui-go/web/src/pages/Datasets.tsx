import { useState } from 'react';
import { Database, Plus } from 'lucide-react';
import { api, fmtAgo, fmtBytes, plural } from '../api';
import { go } from '../router';
import type { Dataset, Items } from '../types';
import { Empty, Field, Loading, Modal, Spinner, usePoll, useToast } from '../ui';

export function Datasets() {
  const toast = useToast();
  const [items, setItems] = useState<Dataset[] | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);

  usePoll(async signal => setItems((await api.get<Items<Dataset>>('/v1/datasets', signal)).items), 15000);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      const ds = await api.post<Dataset>('/v1/datasets', { name: name.trim() });
      toast.success(`Created ${ds.name}`);
      go('datasets', ds.name);
    } catch (err) {
      toast.error(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Datasets</h1>
          <div className="page-sub">{items ? plural(items.length, 'dataset') : ''}</div>
        </div>
        <button className="btn btn-primary" onClick={() => setCreating(true)}>
          <Plus size={12} /> New dataset
        </button>
      </div>
      {!items ? (
        <Loading what="datasets" />
      ) : items.length === 0 ? (
        <Empty icon={<Database size={28} />} title="No datasets">
          Create one, then upload images and captions (or a zip).
        </Empty>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Images</th>
                <th>Captions</th>
                <th>Size</th>
                <th>Modified</th>
              </tr>
            </thead>
            <tbody>
              {items.map(d => (
                <tr key={d.name} className="clickable" onClick={() => go('datasets', d.name)}>
                  <td style={{ fontWeight: 700 }}>{d.name}</td>
                  <td className="mono">{d.images}</td>
                  <td className="mono">
                    {d.captions}
                    {d.images > 0 && d.captions < d.images && <span className="warn-text"> · {d.images - d.captions} missing</span>}
                  </td>
                  <td className="mono nowrap">{fmtBytes(d.size)}</td>
                  <td className="small muted nowrap">{fmtAgo(d.modified)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {creating && (
        <Modal title="New dataset" onClose={() => setCreating(false)}>
          <form onSubmit={create}>
            <div className="modal-body">
              <Field label="Name" hint="Becomes the folder name: letters, digits, - and _">
                <input type="text" value={name} onChange={e => setName(e.target.value)} autoFocus placeholder="my_subject" />
              </Field>
            </div>
            <div className="modal-actions">
              <button type="button" className="btn" onClick={() => setCreating(false)}>
                Cancel
              </button>
              <button className="btn btn-primary" disabled={busy || !name.trim()}>
                {busy ? <Spinner size={12} /> : <Plus size={12} />} Create
              </button>
            </div>
          </form>
        </Modal>
      )}
    </>
  );
}

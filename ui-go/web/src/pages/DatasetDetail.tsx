import { useCallback, useEffect, useRef, useState } from 'react';
import { FileArchive, Save, Trash2, Upload as UploadIcon } from 'lucide-react';
import { api, ApiError, fmtBytes, imgUrl, joinServerPath, plural, upload } from '../api';
import { go, href } from '../router';
import type { DatasetDetail as Detail, DatasetItem } from '../types';
import { Bar, Empty, Loading, Spinner, useConfirm, useToast } from '../ui';

const PAGE = 120;
const isVideo = (f: string) => /\.(mp4|webm|mov|mkv|avi)$/i.test(f);
const isAudio = (f: string) => /\.(mp3|wav|flac|ogg)$/i.test(f);

export function DatasetDetail({ name }: { name: string }) {
  const toast = useToast();
  const confirm = useConfirm();
  const [ds, setDs] = useState<Detail | null>(null);
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [over, setOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const zipInput = useRef<HTMLInputElement>(null);
  const base = `/v1/datasets/${encodeURIComponent(name)}`;

  const load = useCallback(async () => {
    try {
      const d = await api.get<Detail>(`${base}?limit=${PAGE}`);
      setDs(d);
      setItems(d.items);
      setCursor(d.next_cursor);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) setMissing(true);
      else toast.error(e);
    }
  }, [base, toast]);

  useEffect(() => {
    load();
  }, [load]);

  const more = async () => {
    if (!cursor) return;
    try {
      const d = await api.get<Detail>(`${base}?limit=${PAGE}&cursor=${encodeURIComponent(cursor)}`);
      setItems(i => [...i, ...d.items]);
      setCursor(d.next_cursor);
    } catch (e) {
      toast.error(e);
    }
  };

  const send = async (files: File[]) => {
    if (!files.length) return;
    setProgress(0);
    try {
      const zips = files.filter(f => /\.zip$/i.test(f.name));
      const plain = files.filter(f => !/\.zip$/i.test(f.name));
      if (plain.length) {
        const res = await upload<{ items: string[] }>(`${base}/files`, plain, 'files[]', setProgress);
        toast.success(`Uploaded ${plural(res.items.length, 'file')}`);
      }
      for (const z of zips) {
        const res = await upload<{ added: number; skipped: string[] }>(`${base}/archive`, [z], 'file', setProgress);
        toast.success(`${z.name}: added ${res.added}${res.skipped?.length ? `, skipped ${res.skipped.length}` : ''}`);
      }
      await load();
    } catch (e) {
      toast.error(e);
    } finally {
      setProgress(null);
    }
  };

  const delDataset = async () => {
    if (!(await confirm({ title: 'Delete dataset', body: <>Delete <b>{name}</b> and every file in it?</>, confirm: 'Delete', danger: true }))) return;
    try {
      await api.del(base);
      toast.success('Dataset deleted');
      go('datasets');
    } catch (e) {
      toast.error(e);
    }
  };

  if (missing)
    return (
      <Empty title="Dataset not found">
        <a href={href('datasets')}>Back to datasets</a>
      </Empty>
    );
  if (!ds) return <Loading what="dataset" />;

  return (
    <>
      <div className="crumbs">
        <a href={href('datasets')}>Datasets</a>
      </div>
      <div className="page-head">
        <div>
          <h1 className="page-title">{ds.name}</h1>
          <div className="page-sub">
            {plural(ds.images, 'image')} · {plural(ds.captions, 'caption')} · {fmtBytes(ds.size)} · <span className="mono">{ds.path}</span>
          </div>
        </div>
        <div className="row-wrap">
          <button className="btn" onClick={() => fileInput.current?.click()} disabled={progress !== null}>
            <UploadIcon size={12} /> Upload files
          </button>
          <button className="btn" onClick={() => zipInput.current?.click()} disabled={progress !== null}>
            <FileArchive size={12} /> Import zip
          </button>
          <button className="btn btn-ghost-danger" onClick={delDataset}>
            <Trash2 size={12} /> Delete
          </button>
        </div>
      </div>
      <input ref={fileInput} type="file" multiple hidden onChange={e => send(Array.from(e.target.files ?? []))} />
      <input ref={zipInput} type="file" accept=".zip" hidden onChange={e => send(Array.from(e.target.files ?? []))} />

      <div
        className={over ? 'drop over' : 'drop'}
        style={{ marginBottom: 20 }}
        onClick={() => fileInput.current?.click()}
        onDragOver={e => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={e => {
          e.preventDefault();
          setOver(false);
          send(Array.from(e.dataTransfer.files));
        }}
      >
        {progress !== null ? (
          <div className="stack" style={{ alignItems: 'center' }}>
            <span className="row">
              <Spinner size={12} /> Uploading {Math.round(progress * 100)}%
            </span>
            <div style={{ width: 240 }}>
              <Bar value={progress} max={1} />
            </div>
          </div>
        ) : (
          'Drop images, videos, .txt captions or a .zip here'
        )}
      </div>

      {items.length === 0 ? (
        <Empty title="Empty dataset">Upload media with matching .txt captions (image.png + image.txt).</Empty>
      ) : (
        <>
          <div className="grid grid-media">
            {items.map(it => (
              <ItemCard key={it.file} dsName={name} folder={ds.path} item={it} onDeleted={() => setItems(x => x.filter(y => y.file !== it.file))} />
            ))}
          </div>
          {cursor && (
            <div className="row" style={{ justifyContent: 'center', marginTop: 16 }}>
              <button className="btn" onClick={more}>
                Load more
              </button>
            </div>
          )}
        </>
      )}
    </>
  );
}

function ItemCard({ dsName, folder, item, onDeleted }: { dsName: string; folder: string; item: DatasetItem; onDeleted: () => void }) {
  const toast = useToast();
  const confirm = useConfirm();
  const [caption, setCaption] = useState(item.caption ?? '');
  const [saved, setSaved] = useState(item.caption ?? '');
  const [saving, setSaving] = useState(false);
  // a caption uploaded or changed elsewhere replaces the shown one unless it is being edited
  const savedRef = useRef(item.caption ?? '');
  useEffect(() => {
    const prev = savedRef.current;
    const fresh = item.caption ?? '';
    savedRef.current = fresh;
    setSaved(fresh);
    setCaption(cur => (cur === prev ? fresh : cur));
  }, [item.caption]);
  const path = joinServerPath(folder, item.file);
  const fileRoute = `/v1/datasets/${encodeURIComponent(dsName)}`;
  const encFile = item.file.split('/').map(encodeURIComponent).join('/');

  const save = async () => {
    setSaving(true);
    try {
      await api.put(`${fileRoute}/captions/${encFile}`, { caption });
      savedRef.current = caption;
      setSaved(caption);
    } catch (e) {
      toast.error(e);
    } finally {
      setSaving(false);
    }
  };
  const del = async () => {
    if (!(await confirm({ title: 'Delete file', body: <>Delete <span className="mono">{item.file}</span> and its caption?</>, confirm: 'Delete', danger: true }))) return;
    try {
      await api.del(`${fileRoute}/files/${encFile}`);
      onDeleted();
    } catch (e) {
      toast.error(e);
    }
  };

  return (
    <div className="thumb">
      {isVideo(item.file) ? (
        <video src={imgUrl(path)} muted preload="metadata" style={{ width: '100%', aspectRatio: '1', objectFit: 'cover' }} />
      ) : isAudio(item.file) ? (
        <img src={`/api/audio/art/${encodeURIComponent(path.slice(0, Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'))))}/${encodeURIComponent(item.file.split('/').pop()!)}`} alt="" loading="lazy" />
      ) : (
        <img src={imgUrl(path, true)} alt={item.file} loading="lazy" />
      )}
      <div className="thumb-meta">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <span className="mono small ellipsis" title={item.file}>
            {item.file}
          </span>
          <button className="icon-btn ghost" onClick={del} aria-label="Delete file" title="Delete file">
            <Trash2 size={12} />
          </button>
        </div>
        <span className="dim small mono">
          {item.width && item.height ? `${item.width}×${item.height} · ` : ''}
          {fmtBytes(item.size)}
        </span>
        <textarea rows={3} value={caption} onChange={e => setCaption(e.target.value)} placeholder="No caption" style={{ fontSize: 11 }} />
        {caption !== saved && (
          <button className="btn btn-sm btn-primary" onClick={save} disabled={saving}>
            {saving ? <Spinner size={10} /> : <Save size={10} />} Save caption
          </button>
        )}
      </div>
    </div>
  );
}

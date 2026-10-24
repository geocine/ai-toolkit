import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronRight, FileArchive, FileX, ImagePlus, Save, Sparkles, Trash2, Upload as UploadIcon } from 'lucide-react';
import { api, ApiError, fmtBytes, imgUrl, joinServerPath, plural, upload } from '../api';
import { CaptionCheckBar, CaptionDialog, CaptionRun, captionActive } from '../captionUi';
import { CompareDialog, LeftOut, PhotoCheckBar, PhotoDot, PhotoFlags, PhotoItemMenu, usePhotoActions, usePhotoReview, type PhotoActions } from '../photoUi';
import { go, href } from '../router';
import type { DatasetDetail as Detail, DatasetItem, PhotoItem, PhotoReview } from '../types';
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
  const [captioning, setCaptioning] = useState(false);
  const [onlyFiles, setOnlyFiles] = useState<Set<string> | null>(null);
  const [photoFiles, setPhotoFiles] = useState<Set<string> | null>(null);
  const [compare, setCompare] = useState<[string, string] | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const zipInput = useRef<HTMLInputElement>(null);
  const base = `/v1/datasets/${encodeURIComponent(name)}`;

  // a refresh keeps as many items as are shown (pages loaded with "Load more")
  const shown = useRef(PAGE);
  shown.current = Math.max(PAGE, items.length);
  const load = useCallback(async () => {
    try {
      const d = await api.get<Detail>(`${base}?limit=${shown.current}`);
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

  // while a caption job or a face check runs, its progress and what it writes
  const active = captionActive(ds?.caption_job);
  const facing = captionActive(ds?.photo_job);
  const wasActive = useRef(false);
  const wasFacing = useRef(false);
  useEffect(() => {
    if (wasActive.current && !active && ds?.caption_job) {
      const j = ds.caption_job;
      if (j.state === 'completed') toast.success('Captioning finished');
      else if (j.state === 'stopped') toast.success('Captioning stopped');
    }
    if (wasFacing.current && !facing && ds?.photo_job?.state === 'completed') toast.success('Face check finished');
    wasActive.current = active;
    wasFacing.current = facing;
    if (!active && !facing) return;
    const id = setInterval(() => document.visibilityState !== 'hidden' && load(), 3000);
    return () => clearInterval(id);
  }, [active, facing, load]); // eslint-disable-line react-hooks/exhaustive-deps

  // the photo review follows the files, their captions and the face check
  const version = `${ds?.images}|${ds?.size}|${ds?.photo_job?.id}:${ds?.photo_job?.state}|${items.map(i => `${i.file}:${i.caption ?? ''}`).join('|')}`;
  const photos = usePhotoReview(name, version);
  const review = photos.review;
  const photoActions = usePhotoActions(name, photos.setReview, load);
  const byFile = useMemo(() => new Map((review?.items ?? []).map(i => [i.file, i])), [review]);
  const shownItems = items.filter(i => (!onlyFiles || onlyFiles.has(i.file)) && (!photoFiles || photoFiles.has(i.file)));

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
      <Empty
        icon={<FileX size={18} />}
        title="Dataset not found"
        action={
          <a className="btn btn-sm" href={href('datasets')}>
            Back to datasets
          </a>
        }
      >
        It may have been deleted or renamed.
      </Empty>
    );
  if (!ds) return <Loading what="dataset" />;

  return (
    <>
      <div className="crumbs">
        <a href={href('datasets')}>Datasets</a>
        <ChevronRight size={14} />
        <span>{ds.name}</span>
      </div>
      <div className="page-head">
        <div>
          <h1 className="page-title">{ds.name}</h1>
          <div className="page-sub ellipsis" title={ds.path}>
            {plural(ds.images, 'image')} · {plural(ds.captions, 'caption')} · {fmtBytes(ds.size)} · <span className="mono">{ds.path}</span>
          </div>
        </div>
        <div className="page-actions">
          <button className="btn btn-destructive" onClick={delDataset}>
            <Trash2 size={14} /> Delete
          </button>
          <button className="btn" onClick={() => setCaptioning(true)} disabled={active || ds.images === 0} title={active ? 'Captioning now' : undefined}>
            <Sparkles size={14} /> Auto caption
          </button>
          <button className="btn" onClick={() => zipInput.current?.click()} disabled={progress !== null}>
            <FileArchive size={14} /> Import zip
          </button>
          <button className="btn btn-primary" onClick={() => fileInput.current?.click()} disabled={progress !== null}>
            <UploadIcon size={14} /> Upload files
          </button>
        </div>
      </div>
      {ds.caption_job && <CaptionRun job={ds.caption_job} onChange={load} />}
      {captioning && (
        <CaptionDialog
          dataset={name}
          media={items.some(i => isVideo(i.file) || isAudio(i.file))}
          onStarted={j => setDs(d => (d ? { ...d, caption_job: j } : d))}
          onClose={() => setCaptioning(false)}
        />
      )}
      <input ref={fileInput} type="file" multiple hidden onChange={e => send(Array.from(e.target.files ?? []))} />
      <input ref={zipInput} type="file" accept=".zip" hidden onChange={e => send(Array.from(e.target.files ?? []))} />

      <div
        className={over ? 'drop over' : 'drop'}
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
          <div className="stack" style={{ alignItems: 'center', gap: 10 }}>
            <span className="row num">
              <Spinner size={14} /> Uploading {Math.round(progress * 100)}%
            </span>
            <div style={{ width: 240 }}>
              <Bar value={progress} max={1} />
            </div>
          </div>
        ) : (
          <>
            <ImagePlus size={16} /> Drop images, videos, .txt captions or a .zip here, or click to browse
          </>
        )}
      </div>

      {items.length === 0 ? (
        <Empty icon={<ImagePlus size={18} />} title="Empty dataset">
          Upload media with matching .txt captions (<span className="mono">image.png</span> + <span className="mono">image.txt</span>).
        </Empty>
      ) : (
        <>
          <CaptionCheckBar dataset={name} version={items.map(i => i.caption ?? '').join('|')} onFilter={setOnlyFiles} onFixed={load} />
          {review && review.total > 0 && (
            <PhotoCheckBar
              dataset={name}
              review={review}
              actions={photoActions}
              photoJob={ds.photo_job}
              onFilter={setPhotoFiles}
              onFaceJob={j => setDs(d => (d ? { ...d, photo_job: j } : d))}
              onChanged={load}
              onRefresh={photos.refresh}
            />
          )}
          <div className="grid grid-items">
            {shownItems.map(it => (
              <ItemCard
                key={it.file}
                dsName={name}
                folder={ds.path}
                item={it}
                photo={byFile.get(it.file)}
                review={review}
                actions={photoActions}
                onCompare={(a, b) => setCompare([a, b])}
                onSaved={caption => setItems(x => x.map(y => (y.file === it.file ? { ...y, caption } : y)))}
                onDeleted={() => {
                  setItems(x => x.filter(y => y.file !== it.file));
                  load();
                }}
              />
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
      {review && <LeftOut folder={ds.path} review={review} actions={photoActions} />}
      {review && compare && <CompareDialog folder={ds.path} review={review} a={compare[0]} b={compare[1]} actions={photoActions} onClose={() => setCompare(null)} />}
    </>
  );
}

function ItemCard({
  dsName,
  folder,
  item,
  photo,
  review,
  actions,
  onCompare,
  onSaved,
  onDeleted,
}: {
  dsName: string;
  folder: string;
  item: DatasetItem;
  photo?: PhotoItem;
  review: PhotoReview | null;
  actions: PhotoActions;
  onCompare: (a: string, b: string) => void;
  onSaved: (caption: string) => void;
  onDeleted: () => void;
}) {
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
      // the page's items carry the caption too, so the caption and photo checks see it
      onSaved(caption);
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
    <div className="card item">
      {isVideo(item.file) ? (
        <video className="item-media" src={imgUrl(path)} muted preload="metadata" />
      ) : isAudio(item.file) ? (
        <img className="item-media" src={`/api/audio/art/${encodeURIComponent(path.slice(0, Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'))))}/${encodeURIComponent(item.file.split('/').pop()!)}`} alt="" loading="lazy" />
      ) : (
        <div className="item-thumb">
          <img className="item-media" src={imgUrl(path, true)} alt={item.file} loading="lazy" />
          <PhotoDot item={photo} />
        </div>
      )}
      <div className="item-head">
        <div style={{ minWidth: 0 }}>
          <div className="mono ellipsis" title={item.file} style={{ fontSize: 12 }}>
            {item.file}
          </div>
          <div className="item-meta">
            {item.width && item.height ? `${item.width} × ${item.height} · ` : ''}
            {photo?.px ? <span title={`Trains at about ${photo.px} px`}>{photo.px} px · </span> : null}
            {fmtBytes(item.size)}
          </div>
        </div>
        <div className="item-actions">
          {photo && review && <PhotoItemMenu item={photo} review={review} actions={actions} onCompare={onCompare} />}
          <button className="icon-btn sm danger" onClick={del} aria-label="Delete file" title="Delete file">
            <Trash2 size={14} />
          </button>
        </div>
      </div>
      <PhotoFlags item={photo} />
      <textarea className="caption" rows={3} value={caption} onChange={e => setCaption(e.target.value)} placeholder="No caption" aria-label={`Caption for ${item.file}`} />
      {caption !== saved && (
        <button className="btn btn-sm btn-primary" onClick={save} disabled={saving}>
          {saving ? <Spinner size={14} /> : <Save size={14} />} Save caption
        </button>
      )}
    </div>
  );
}

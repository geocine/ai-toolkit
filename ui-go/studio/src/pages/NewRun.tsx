import { ArrowLeft, ArrowRight, Check, ChevronDown, ChevronRight, Copy, Rocket, Sparkles, Square, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import YAML from 'yaml';
import { CompareDialog } from '../components/CompareDialog';
import { Confirm } from '../components/Dialog';
import { Dropzone } from '../components/Dropzone';
import { FaceCheck } from '../components/FaceCheck';
import {
  DecidePairs,
  PhotoGrid,
  PhotoMenu,
  PhotoSummary,
  PhotoThumb,
  groupKey,
  partnerOf,
  suggested,
  type GridPhoto,
  type PhotoAction,
  type PhotoFilter,
} from '../components/PhotoGrid';
import { FaceCropOffer, Readiness } from '../components/Readiness';
import { Callout, ErrorNote, NeedServer, PageHeader, Progress, useLoad } from '../components/ui';
import {
  ApiError,
  call,
  get,
  mediaUrl,
  post,
  upload,
  type CaptionCatalog,
  type CaptionCheck,
  type CaptionFix,
  type CaptionFlag,
  type Captioner,
  type Preflight,
  type Dataset,
  type DatasetDetail,
  type DatasetItem,
  type Items,
  type Job,
  type Remote,
  type Settings,
  type SystemInfo,
} from '../lib/api';
import { useServer, useStudio } from '../lib/context';
import { buildConfig, knobsFor, validate, withSubject, type Knobs, type Preset, type Subject } from '../lib/presets';
import { go } from '../lib/route';
import { save, useStored } from '../lib/store';
import { bytes } from '../lib/format';
import { batches, prepareImages, trainable, type PrepNote } from '../lib/images';
import { assess, assessFailure, checkPhotos, fixPhotos, photoSettings } from '../lib/photos';
import type {
  PhotoAssess,
  PhotoDecision,
  PhotoFix,
  PhotoFixOp,
  PhotoFlag,
  PhotoGroup,
  PhotoItem,
  PhotoMetrics,
  PhotoPick,
  PhotoReview,
  PhotoSettingsPatch,
  UploadResult,
} from '../lib/photoTypes';
import { PresetNotes } from './Presets';

const steps = ['Subject', 'Images', 'Captions', 'Settings', 'Launch'] as const;

const joinServerPath = (folder: string, rel: string) => {
  const sep = folder.includes('\\') && !folder.includes('/') ? '\\' : '/';
  return folder.replace(/[\\/]+$/, '') + sep + rel.replace(/\//g, sep);
};

const datasetSafe = (s: string) => s.trim().replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._-]+/, '');

interface Draft {
  step: number;
  presetId: string;
  k: Knobs;
  dataset: string;
}

const DRAFT_KEY = 'aitk.studio.draft';

export function NewRun() {
  const server = useServer();
  const { presets } = useStudio();
  // the draft survives reloads (a long caption pass should not be lost)
  const [draft, setDraft] = useStored<Draft>(DRAFT_KEY, { step: 0, presetId: presets[0].id, k: knobsFor(presets[0]), dataset: '' });
  const preset = presets.find(p => p.id === draft.presetId) ?? presets[0];
  const { step, k, dataset } = draft;
  const setStep = (s: number) => setDraft(d => ({ ...d, step: s }));
  const setK = (fn: (prev: Knobs) => Knobs) => setDraft(d => ({ ...d, k: fn(d.k) }));
  const setPreset = (p: Preset) => setDraft(d => ({ ...d, presetId: p.id }));
  const setDataset = (name: string) => setDraft(d => ({ ...d, dataset: name }));

  if (!server) return <NeedServer title="New run" />;

  const set = (patch: Partial<Knobs>) => setK(prev => ({ ...prev, ...patch }));

  return (
    <div className="wizard">
      <PageHeader title="New run" description="The draft is kept in this browser until you launch it." />
      <ol className="stepper">
        {steps.map((s, i) => (
          <li key={s} className={i === step ? 'is-current' : i < step ? 'is-done' : undefined}>
            <button disabled={i > step || (i >= 2 && !dataset)} onClick={() => setStep(i)} aria-current={i === step ? 'step' : undefined}>
              <span className="stepper-n">{i < step ? <Check aria-hidden /> : i + 1}</span>
              <span className="stepper-label">{s}</span>
            </button>
          </li>
        ))}
      </ol>

      {step === 0 && (
        <StepSubject
          presets={presets}
          preset={preset}
          k={k}
          onPreset={p => {
            setPreset(p);
            setK(prev => ({ ...knobsFor(p, prev.name, prev.trigger) }));
          }}
          onReset={() => setDraft({ step: 0, presetId: presets[0].id, k: knobsFor(presets[0]), dataset: '' })}
          set={set}
          onSubject={s => setK(prev => withSubject(preset, prev, s))}
          next={() => setStep(1)}
        />
      )}
      {step === 1 && (
        <StepImages
          k={k}
          dataset={dataset}
          onDone={name => {
            setDataset(name);
            setStep(2);
          }}
          back={() => setStep(0)}
        />
      )}
      {step === 2 && dataset && <StepCaptions preset={preset} k={k} dataset={dataset} next={() => setStep(3)} back={() => setStep(1)} />}
      {step === 3 && <StepSettings preset={preset} k={k} set={set} next={() => setStep(4)} back={() => setStep(2)} />}
      {step === 4 && dataset && <StepLaunch preset={preset} k={k} dataset={dataset} back={() => setStep(3)} />}
    </div>
  );
}

function WizardStep({ title, description, children }: { title: string; description?: ReactNode; children: ReactNode }) {
  return (
    <section className="card wizard-step">
      <header className="wizard-head">
        <h2>{title}</h2>
        {description && <p className="section-desc">{description}</p>}
      </header>
      {children}
    </section>
  );
}

// ---- 1. subject ----

function StepSubject(props: {
  presets: Preset[];
  preset: Preset;
  k: Knobs;
  onPreset: (p: Preset) => void;
  set: (p: Partial<Knobs>) => void;
  onSubject: (s: Subject) => void;
  onReset: () => void;
  next: () => void;
}) {
  const { presets, preset, k, set } = props;
  const [nameTouched, setNameTouched] = useState(Boolean(k.name));
  const ok = k.trigger.trim() && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(k.name);
  return (
    <WizardStep title="Who are we training?">
      <div className="form-grid">
        <label className="field field-wide">
          <span>Preset</span>
          <select value={preset.id} onChange={e => props.onPreset(presets.find(p => p.id === e.target.value)!)}>
            {presets.map(p => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <small>{preset.description}</small>
        </label>
        <label className="field">
          <span>Trigger word</span>
          <input
            autoFocus
            value={k.trigger}
            placeholder="jihan"
            onChange={e => {
              const trigger = e.target.value.trim();
              set(nameTouched ? { trigger } : { trigger, name: trigger ? `${datasetSafe(trigger).toLowerCase()}_kr` : '' });
            }}
          />
          <small>Written as [trigger] in captions and sample prompts.</small>
        </label>
        <label className="field">
          <span>Run name</span>
          <input
            value={k.name}
            placeholder="jihan_kr"
            onChange={e => {
              setNameTouched(true);
              set({ name: e.target.value });
            }}
          />
          <small>Also the output folder and checkpoint file name.</small>
        </label>
        {preset.promptSets && (
          <fieldset className="field field-wide">
            <span>Subject</span>
            <div className="seg">
              {(['female', 'male', 'man'] as Subject[]).map(s => (
                <button key={s} type="button" className={k.subject === s ? 'is-on' : ''} onClick={() => props.onSubject(s)}>
                  {s === 'female' ? 'Young woman' : s === 'male' ? 'Young man' : 'Man'}
                </button>
              ))}
            </div>
            <small>Picks the sample prompts and the caption prompt template.</small>
          </fieldset>
        )}
      </div>
      <PresetNotes preset={preset} />
      <div className="wizard-nav">
        {k.name || k.trigger ? (
          <button className="btn btn-ghost" onClick={props.onReset}>
            Start over
          </button>
        ) : (
          <span />
        )}
        <button className="btn btn-primary" disabled={!ok} onClick={props.next}>
          Next: images
          <ArrowRight />
        </button>
      </div>
    </WizardStep>
  );
}

// ---- 2. images ----

/** A photo picked in this browser, not uploaded yet. Left out, it never uploads. */
type LocalPhoto = { key: string; file: File; metrics?: PhotoMetrics; leftOut: boolean };

let localSeq = 0;
const stemOf = (name: string) => name.replace(/\.[^.]+$/, '');
const isCaption = (f: File) => /\.txt$/i.test(f.name);
const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

/** The quiet note of what getting the photos ready did on its own. */
function prepLine(notes: PrepNote[]): string {
  const n = (w: PrepNote['what']) => notes.filter(x => x.what === w).length;
  return [
    n('converted') && `${plural(n('converted'), 'photo', 'photos')} converted to JPEG`,
    n('upright') && `${plural(n('upright'), 'rotated photo', 'rotated photos')} turned upright`,
    n('gray16') && `${plural(n('gray16'), '16-bit grey PNG', '16-bit grey PNGs')} made 8-bit`,
    n('flattened') && `${plural(n('flattened'), 'transparent PNG', 'transparent PNGs')} put on white`,
    n('renamed') && `${plural(n('renamed'), 'file', 'files')} renamed so names do not clash`,
    n('server_convert') && `${plural(n('server_convert'), 'photo', 'photos')} converted on the server after upload`,
  ]
    .filter(Boolean)
    .join(' · ');
}

const goodList = [
  'Just them, face clearly visible and in focus',
  'Close-ups and some half- and full-length',
  'Different days, outfits, places and light',
  'Some with the head turned partly away and one or two side profiles',
  'Different expressions',
];
const leaveOutList = [
  'Group photos (or crop the others out)',
  'Sunglasses, masks or hats that hide the face',
  'Heavy filters and beauty smoothing',
  'Screenshots, collages, photos with text or watermarks',
  'Several shots from the same moment (keep the best)',
  'Photos many years apart',
  'Copies saved from chat apps (use the originals)',
];

function GoodPhotos() {
  return (
    <details className="photo-advice">
      <summary>What makes a good photo</summary>
      <div className="photo-advice-lists">
        <div>
          <p className="label">Good</p>
          <ul>
            {goodList.map(s => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </div>
        <div>
          <p className="label">Leave out</p>
          <ul>
            {leaveOutList.map(s => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </div>
      </div>
    </details>
  );
}

/**
 * Groups seen in any review, kept while their photos are here: leaving one
 * out must not hide the pair. Of two groups of one kind where one holds the
 * other's files (a burst of 3 that is 2 once one is left out), only the
 * larger stays, so no tile shows twice.
 */
function mergeGroups(prev: PhotoGroup[], next: PhotoGroup[]): PhotoGroup[] {
  const within = (a: PhotoGroup, b: PhotoGroup) => a.files.every(f => b.files.includes(f));
  const out = new Map(prev.map(g => [groupKey(g), g]));
  for (const g of next) {
    if (g.kind === 'duplicate') continue;
    const same = [...out.values()].filter(o => o.kind === g.kind);
    if (same.some(o => within(g, o) && groupKey(o) !== groupKey(g))) continue;
    for (const o of same) if (within(o, g)) out.delete(groupKey(o));
    out.set(groupKey(g), g);
  }
  return [...out.values()];
}

function StepImages({ k, dataset, onDone, back }: { k: Knobs; dataset: string; onDone: (name: string) => void; back: () => void }) {
  const server = useServer()!;
  const [name, setName] = useState(dataset || datasetSafe(k.trigger.toLowerCase()) || datasetSafe(k.name));
  const existing = useLoad(server, s => get<Items<Dataset>>(s, '/v1/datasets'), []);
  const already = existing.data?.items.find(d => d.name === name);
  const resKey = k.resolution.join(',');
  // the dataset's folder, for a dropped copy's stored twin in the compare dialog
  const stored = useLoad(server, s => (already ? get<DatasetDetail>(s, `/v1/datasets/${encodeURIComponent(name)}?limit=1`) : Promise.resolve(null)), [name, !!already]);

  // the formats the server converts after upload (heic through ffmpeg, bmp, gif, tiff)
  const [convert, setConvert] = useState<string[]>([]);
  useEffect(() => {
    let live = true;
    assess(server, { items: [] }).then(
      r => live && setConvert(r.convert ?? []),
      () => {}, // an older server: no photo checks, nothing converted there
    );
    return () => {
      live = false;
    };
  }, [server]);

  const [photos, setPhotos] = useState<LocalPhoto[]>([]);
  const [captions, setCaptions] = useState<File[]>([]);
  const [captionText, setCaptionText] = useState<Record<string, string>>({}); // by image file
  const [serverConvert, setServerConvert] = useState<File[]>([]);
  const [notes, setNotes] = useState<PrepNote[]>([]);
  const [skipped, setSkipped] = useState<string[]>([]);
  // names: each card's provisional name in the live check made while measuring (prepareImages onMetrics)
  const [measuring, setMeasuring] = useState<{ done: number; total: number; files: File[]; names?: Map<File, string> } | null>(null);
  const [decisions, setDecisions] = useState<Record<string, PhotoDecision>>({});
  const [review, setReview] = useState<PhotoReview | null>(null);
  // the last judgement of every photo: a left-out photo is not sent again, but its card keeps its flags
  const [seen, setSeen] = useState<Record<string, PhotoItem>>({});
  const [groups, setGroups] = useState<PhotoGroup[]>([]);
  const [pick, setPick] = useState<PhotoPick | null>(null);
  const [filter, setFilter] = useState<PhotoFilter>(null);
  const [undo, setUndo] = useState<Set<string> | null>(null); // left-out keys before the last bulk action
  const [compare, setCompare] = useState<[string, string] | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [uploaded, setUploaded] = useState(false);
  const [error, setError] = useState<unknown>(null); // a drop or an upload that failed
  // the live check's own failure: a later check that works clears it, and a
  // drop's error is left alone
  const [assessError, setAssessError] = useState<unknown>(null);
  // one drop is measured, or one upload runs, at a time: a drop meanwhile is
  // turned away (it would race the names, and the end of an upload clears
  // the grid). The lock opens only once the render with the new photos is
  // committed, so the next drop sees their names.
  const busyRef = useRef(false);
  const drops = useRef(0); // drops so far
  const measuringDrop = useRef(0); // the drop being measured, 0 when none
  const [settled, setSettled] = useState(0);
  const [turnedAway, setTurnedAway] = useState(false);
  useEffect(() => {
    busyRef.current = false;
  }, [settled]);
  const dropBusy = measuring !== null || progress !== null;

  const local = photos.length + captions.length + serverConvert.length > 0 || measuring !== null;
  const kept = photos.filter(p => !p.leftOut && p.metrics);
  const leftOut = photos.filter(p => p.leftOut).length;

  const assessBody = (n?: number): PhotoAssess => {
    const names = new Set(kept.map(p => p.file.name));
    const only = <T,>(m: Record<string, T>) => Object.fromEntries(Object.entries(m).filter(([f]) => names.has(f)));
    return {
      items: kept.map(p => p.metrics!),
      resolution: k.resolution,
      dataset: already ? name : undefined,
      decisions: only(decisions),
      captions: only(captionText),
      pick: n,
    };
  };
  const take = (r: PhotoReview) => {
    setReview(r);
    setSeen(prev => ({ ...prev, ...Object.fromEntries(r.items.map(i => [i.file, i])) }));
    setGroups(prev => mergeGroups(prev, r.groups));
  };

  // the server's judge on what would upload; again when the photos, the decisions or the run's resolution change
  const assessKey = [kept.map(p => p.file.name).join('\u0000'), JSON.stringify(decisions), resKey, already ? name : '', Object.keys(captionText).length].join('\u0001');
  useEffect(() => {
    if (measuring) return;
    if (!kept.length) {
      setReview(null);
      setAssessError(null);
      return;
    }
    let live = true;
    const t = setTimeout(() => {
      assess(server, assessBody()).then(
        r => {
          if (!live) return;
          setAssessError(null);
          take(r);
        },
        e => live && setAssessError(assessFailure(e)),
      );
    }, 250);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [assessKey, measuring === null]); // eslint-disable-line react-hooks/exhaustive-deps

  const storedNames = async (): Promise<string[]> => {
    if (!already) return [];
    try {
      const d = await get<DatasetDetail>(server, `/v1/datasets/${encodeURIComponent(name)}?limit=1000`);
      return d.items.map(i => i.file);
    } catch {
      return []; // the server renames on a clash anyway (on_conflict=rename)
    }
  };

  const onFiles = async (fs: File[]) => {
    if (!fs.length) return;
    if (busyRef.current) {
      setTurnedAway(true);
      return;
    }
    busyRef.current = true;
    setTurnedAway(false);
    setError(null);
    setPick(null);
    const shown = fs.filter(f => trainable.test(f.name));
    setMeasuring({ done: 0, total: shown.length, files: shown });
    // a live check about 500 ms in (or at the first photo measured after that),
    // so flags fill in while the rest is measured; the check at the end judges
    // the final names. A late answer, after measuring ends, is dropped.
    const drop = ++drops.current;
    measuringDrop.current = drop;
    let sofar: { metrics: PhotoMetrics[]; from: File[] } | null = null;
    let due = false;
    const early = () => {
      due = !sofar;
      if (!sofar) return;
      const { metrics, from } = sofar;
      const body = assessBody();
      assess(server, { ...body, items: [...body.items, ...metrics] }).then(
        r => {
          if (measuringDrop.current !== drop) return;
          setAssessError(null);
          setReview(r);
          setMeasuring(m => m && { ...m, names: new Map(from.map((f, i) => [f, metrics[i].file])) });
        },
        e => measuringDrop.current === drop && setAssessError(assessFailure(e)),
      );
    };
    const timer = setTimeout(early, 500);
    try {
      const taken = new Set<string>([...photos.map(p => p.file.name), ...serverConvert.map(f => f.name), ...(await storedNames())]);
      const r = await prepareImages(fs, {
        taken,
        serverConverts: convert,
        onProgress: (done, total) => setMeasuring(m => m && { ...m, done, total }),
        onMetrics: (metrics, from) => {
          sofar = { metrics, from };
          if (due) early();
        },
      });
      const byName = new Map(r.metrics.map(m => [m.file, m]));
      const imgs = r.ok.filter(f => !isCaption(f));
      const txt = r.ok.filter(isCaption);
      setPhotos(prev => [...prev, ...imgs.map(f => ({ key: `p${++localSeq}`, file: f, metrics: byName.get(f.name), leftOut: false }))]);
      setCaptions(prev => [...prev, ...txt]);
      setServerConvert(prev => [...prev, ...r.serverConvert]);
      setNotes(prev => [...prev, ...r.notes]);
      setSkipped(prev => [...prev, ...r.skipped]);
      // caption words are judged before upload too (sunglasses, other people, …)
      const all = [...photos.map(p => p.file), ...imgs];
      const texts: Record<string, string> = {};
      for (const t of txt) {
        const img = all.find(f => stemOf(f.name) === stemOf(t.name));
        if (!img) continue;
        try {
          texts[img.name] = await t.text();
        } catch {
          /* unreadable caption: judged after upload */
        }
      }
      if (Object.keys(texts).length) setCaptionText(prev => ({ ...prev, ...texts }));
    } catch (e) {
      setError(e);
    } finally {
      clearTimeout(timer);
      measuringDrop.current = 0;
      setMeasuring(null);
      setSettled(n => n + 1);
    }
  };

  const clear = () => {
    setPhotos([]);
    setCaptions([]);
    setCaptionText({});
    setServerConvert([]);
    setNotes([]);
    setSkipped([]);
    setDecisions({});
    setReview(null);
    setSeen({});
    setGroups([]);
    setPick(null);
    setFilter(null);
    setUndo(null);
  };

  // leaving out and putting back: a single click drops the last bulk undo and any pick preview
  const setOut = (files: string[], out: boolean, bulk = false) => {
    const want = new Set(files);
    setUndo(bulk ? new Set(photos.filter(p => p.leftOut).map(p => p.key)) : null);
    setPick(null);
    setPhotos(ps => ps.map(p => (want.has(p.file.name) ? { ...p, leftOut: out } : p)));
  };
  const toggle = (file: string) => {
    const p = photos.find(x => x.file.name === file);
    if (p) setOut([file], !p.leftOut);
  };
  const decide = (file: string, d: PhotoDecision | null) =>
    setDecisions(prev => {
      const next = { ...prev };
      if (d) next[file] = d;
      else delete next[file];
      return next;
    });

  const reviewItems = new Map(review?.items.map(i => [i.file, i]) ?? []);
  const rank = new Map(pick?.kept.map((f, i) => [f, i + 1]) ?? []);
  const why = new Map(pick?.left_out.map(l => [l.file, l.reason]) ?? []);
  const grid: GridPhoto[] = [
    ...photos.map(p => ({
      key: p.key,
      file: p.file.name,
      src: '',
      blob: p.file,
      item: (!p.leftOut && reviewItems.get(p.file.name)) || seen[p.file.name],
      leftOut: p.leftOut,
      rank: rank.get(p.file.name),
      reason: why.get(p.file.name),
      stored: false,
    })),
    ...(measuring?.files ?? []).map((f, i) => {
      const live = measuring?.names?.get(f);
      return { key: `m${i}`, file: f.name, src: '', blob: f, item: live ? reviewItems.get(live) : undefined, leftOut: false, stored: false };
    }),
  ];
  const byFile = (f: string) => grid.find(x => x.file === f);
  // a photo of the grid, or the stored photo a dropped copy duplicates (assess with dataset sets same_as to it)
  const storedPath = stored.data?.path ?? '';
  const photoFor = (f: string): GridPhoto | undefined =>
    byFile(f) ??
    (storedPath && already
      ? {
          key: `s:${f}`,
          file: f,
          src: mediaUrl(server, joinServerPath(storedPath, f), true),
          full: mediaUrl(server, joinServerPath(storedPath, f), false),
          leftOut: false,
          stored: true,
        }
      : undefined);

  const onAction = (a: PhotoAction, file: string) => {
    if (a === 'leave_out') setOut([file], true);
    else if (a === 'restore') setOut([file], false);
    else if (a === 'keep' || a === 'intended') decide(file, a);
    else if (a === 'unkeep' || a === 'unintend') decide(file, null);
    else if (a === 'compare') {
      const other = partnerOf(byFile(file)?.item, groups);
      if (other) setCompare([file, other]);
    }
  };
  const onSuggested = () => {
    const s = suggested(grid);
    if (s.files.length) setOut(s.files, true, true);
  };
  const onPick = async (n: number) => {
    setBusy(true);
    setError(null);
    try {
      const r = await assess(server, assessBody(n));
      take(r);
      setPick(r.pick ?? null);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  const keepPick = () => {
    if (!pick) return;
    const keep = new Set(pick.kept);
    setOut(
      kept.filter(p => !keep.has(p.file.name)).map(p => p.file.name),
      true,
      true,
    );
  };
  const onUndo = () => {
    if (!undo) return;
    setPhotos(ps => ps.map(p => ({ ...p, leftOut: undo.has(p.key) })));
    setUndo(null);
    setPick(null);
  };

  const send = async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    setTurnedAway(false);
    setError(null);
    try {
      if (!already) {
        try {
          await post(server, '/v1/datasets', { name });
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 409)) throw e;
        }
      }
      const keptFiles = photos.filter(p => !p.leftOut).map(p => p.file);
      const keptStems = new Set([...keptFiles, ...serverConvert].map(f => stemOf(f.name)));
      const outStems = new Set(photos.filter(p => p.leftOut).map(p => stemOf(p.file.name)));
      // a left-out photo's caption stays behind; each caption goes right after
      // its photo, so a batch rarely splits a pair (the server renames a
      // caption with its photo only inside one request)
      const caps = new Map<string, File[]>();
      for (const c of captions) {
        const s = stemOf(c.name);
        if (outStems.has(s) && !keptStems.has(s)) continue;
        caps.set(s, [...(caps.get(s) ?? []), c]);
      }
      const ordered: File[] = [];
      for (const f of [...keptFiles, ...serverConvert]) {
        ordered.push(f, ...(caps.get(stemOf(f.name)) ?? []));
        caps.delete(stemOf(f.name));
      }
      for (const rest of caps.values()) ordered.push(...rest);

      const renamed = new Map<string, string>();
      if (ordered.length) {
        // in batches, so a proxy's request size limit never refuses the lot
        setProgress(0);
        const total = ordered.reduce((n, f) => n + f.size, 0) || 1;
        let sent = 0;
        for (const batch of batches(ordered)) {
          const size = batch.reduce((n, f) => n + f.size, 0);
          const r = await upload<UploadResult>(server, `/v1/datasets/${encodeURIComponent(name)}/files?on_conflict=rename`, batch, (d, t) =>
            setProgress((sent + (d / (t || 1)) * size) / total),
          );
          for (const x of r?.renamed ?? []) renamed.set(x.from, x.to);
          sent += size;
        }
      }
      const final = (f: string) => renamed.get(f) ?? f;
      // what was decided here follows the photos, under their stored names
      try {
        const keptNames = new Set(keptFiles.map(f => f.name));
        const dec = Object.fromEntries(Object.entries(decisions).filter(([f]) => keptNames.has(f)).map(([f, d]) => [final(f), d]));
        if (Object.keys(dec).length) await photoSettings(server, name, { decisions: dec });
        if (serverConvert.length) await fixPhotos(server, name, { ops: [{ op: 'convert', files: serverConvert.map(f => final(f.name)) }], apply: true });
      } catch (e) {
        setError(e);
      }
      clear();
      setUploaded(true);
      existing.reload();
    } catch (e) {
      setError(e);
    } finally {
      setProgress(null);
      setSettled(n => n + 1);
    }
  };

  const uploads = photos.length - leftOut + serverConvert.length;
  const bytesTotal = [...photos.filter(p => !p.leftOut).map(p => p.file), ...captions, ...serverConvert].reduce((n, f) => n + f.size, 0);
  const prep = prepLine(notes);
  const a = compare && photoFor(compare[0]);
  const b = compare && photoFor(compare[1]);

  return (
    <WizardStep title="Images">
      <div className="form-grid">
        <label className="field">
          <span>Dataset name</span>
          <input
            value={name}
            onChange={e => {
              setName(datasetSafe(e.target.value));
              setUploaded(false);
            }}
          />
          <small>{already ? `Exists on the server (${already.images} images). New files are added to it.` : 'Created on the server.'}</small>
        </label>
        {existing.data && existing.data.items.length > 0 && (
          <label className="field">
            <span>Or reuse a dataset</span>
            <select value={already ? name : ''} onChange={e => e.target.value && setName(e.target.value)}>
              <option value="">Choose a dataset</option>
              {existing.data.items.map(d => (
                <option key={d.name} value={d.name}>
                  {d.name} · {d.images} images
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {/* while busy the file chooser stays shut; a drop is still caught (so the browser does not open the file) and turned away */}
      <div
        className={dropBusy ? 'dropzone-wrap is-busy' : 'dropzone-wrap'}
        aria-disabled={dropBusy || undefined}
        onClickCapture={e => {
          if (!dropBusy) return;
          e.preventDefault();
          e.stopPropagation();
        }}
        onKeyDownCapture={e => {
          if (!dropBusy || (e.key !== 'Enter' && e.key !== ' ')) return;
          e.preventDefault();
          e.stopPropagation();
        }}
      >
        <Dropzone onFiles={onFiles} />
      </div>
      {turnedAway && dropBusy && (
        <p className="quiet-note photo-note">{progress !== null ? 'Add more photos once the upload has finished.' : 'Add more photos once this check has finished.'}</p>
      )}
      {!local && !already && <p className="quiet-note photo-hint">10-30 photos of just them: close-ups and some full-length, from different days, outfits and places.</p>}
      <GoodPhotos />
      {skipped.length > 0 && (
        <Callout kind="warn">
          {skipped.length} file{skipped.length === 1 ? '' : 's'} left out: the trainer reads JPEG, PNG and WebP, and neither this browser nor the server could convert{' '}
          {skipped.slice(0, 3).join(', ')}
          {skipped.length > 3 ? ', …' : ''}. Export them as JPEG and add them again.
        </Callout>
      )}
      {local && (
        <>
          <div className="tally">
            <p>
              <strong className="num">{photos.length + serverConvert.length}</strong> photos
              {captions.length > 0 && <> · {plural(captions.length, 'caption', 'captions')}</>} · {bytes(bytesTotal)}
              {review && review.flagged > 0 && (
                <>
                  {' · '}
                  <button type="button" className="link" onClick={() => setFilter(f => (f === 'flagged' ? null : 'flagged'))}>
                    {review.flagged} to look at
                  </button>
                </>
              )}
            </p>
            <button type="button" className="btn btn-ghost btn-sm" onClick={clear} disabled={measuring !== null || progress !== null}>
              Clear
            </button>
          </div>
          {measuring && (
            <div className="upload-progress">
              <Progress value={measuring.done} max={measuring.total || null} indeterminate={!measuring.total} />
              <span className="num">
                Checking {measuring.done} of {measuring.total}…
              </span>
            </div>
          )}
          <PhotoSummary
            review={review}
            photos={grid}
            filter={filter}
            setFilter={setFilter}
            onSuggested={onSuggested}
            onPick={onPick}
            undo={undo ? onUndo : undefined}
            busy={busy || measuring !== null}
            pick={pick}
            onKeepPick={keepPick}
            onCancelPick={() => setPick(null)}
          />
          {prep && <p className="quiet-note photo-note">{prep}</p>}
          <p className="quiet-note photo-note">Sunglasses, group photos, outfits and places are checked after captions or the face check.</p>
          {review && <Readiness review={review} />}
          <PhotoGrid
            photos={grid}
            filter={filter}
            onToggle={toggle}
            onAction={onAction}
            busy={measuring !== null}
            groups={groups}
            picking={!!pick}
            canCompare={f => !!photoFor(f)}
          />
          {review && <DecidePairs review={{ ...review, groups }} photos={grid} onToggle={toggle} onCompare={(x, y) => setCompare([x, y])} />}
        </>
      )}
      {!local && (already || uploaded) && <StoredPhotos key={name} name={name} k={k} />}
      {a && b && (
        <CompareDialog
          a={a}
          b={b}
          onClose={() => setCompare(null)}
          onLeaveOut={f => setOut([f], true)}
          onKeep={f => setOut([f], false)}
          lockStored
        />
      )}
      {progress !== null && (
        <div className="upload-progress">
          <Progress value={Math.round(progress * 100)} max={100} />
          <span className="num">Uploading {Math.round(progress * 100)}%</span>
        </div>
      )}
      <ErrorNote error={error} />
      <ErrorNote error={assessError} />
      <div className="wizard-nav">
        <button className="btn btn-ghost" onClick={back}>
          <ArrowLeft />
          Back
        </button>
        {local ? (
          <span className="photo-send">
            {leftOut > 0 && <span className="muted num">{leftOut} left out</span>}
            <button className="btn btn-primary" disabled={!name || measuring !== null || progress !== null || (!uploads && !captions.length)} onClick={send}>
              {uploads ? `Upload ${plural(uploads, 'photo', 'photos')}` : `Upload ${plural(captions.length, 'caption', 'captions')}`}
              <ArrowRight />
            </button>
          </span>
        ) : (
          <button className="btn btn-primary" disabled={!name || !(already || uploaded)} onClick={() => onDone(name)}>
            {uploaded ? 'Next: captions' : 'Use this dataset'}
            <ArrowRight />
          </button>
        )}
      </div>
    </WizardStep>
  );
}

/**
 * The photos already on the server (after upload, or a reused dataset):
 * the server's review, polled while photos are still being analysed. Every
 * change here is real: leaving out moves a photo into .set_aside/ (restore
 * puts it back), and decisions are stored with the dataset. The originals a
 * convert or a turn upright kept are listed apart: they are backups, which
 * Restore all leaves, and putting one back replaces the photo made from it.
 */
function StoredPhotos({ name, k }: { name: string; k: Knobs }) {
  const server = useServer()!;
  const resKey = k.resolution.join(',');
  const detail = useLoad(server, s => get<DatasetDetail>(s, `/v1/datasets/${encodeURIComponent(name)}?limit=1`), [name]);
  const [review, setReview] = useState<PhotoReview | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState<PhotoFilter>(null);
  const [pick, setPick] = useState<PhotoPick | null>(null);
  const [undo, setUndo] = useState<string[] | null>(null); // .set_aside/ paths the last bulk action made
  const [compare, setCompare] = useState<[string, string] | null>(null);
  const seq = useRef(0);

  // only the newest answer lands (a poll must not undo a fix)
  const load = useCallback(
    async (wait?: number) => {
      const n = ++seq.current;
      try {
        const r = await checkPhotos(server, name, { resolution: k.resolution, wait_ms: wait });
        if (n === seq.current) {
          setReview(r);
          setError(null);
        }
      } catch (e) {
        if (n === seq.current) setError(e);
      }
    },
    [server, name, resKey], // eslint-disable-line react-hooks/exhaustive-deps
  );
  useEffect(() => {
    setPick(null);
    load(3000);
  }, [load]);
  // photos still being analysed: ask again every 2 s
  useEffect(() => {
    if (!review || review.pending === 0) return;
    const t = setTimeout(() => load(0), 2000);
    return () => clearTimeout(t);
  }, [review, load]);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  const fix = async (ops: PhotoFixOp[]): Promise<PhotoFix> => {
    const r = await fixPhotos(server, name, { ops, apply: true, resolution: k.resolution });
    ++seq.current;
    setReview(r.review);
    setPick(null);
    return r;
  };
  const asideOf = (r: PhotoFix) => r.changes.filter(c => c.op === 'set_aside' && c.to).map(c => c.to!);
  const settings = async (patch: PhotoSettingsPatch) => {
    await photoSettings(server, name, patch);
    await load(0);
  };

  if (!review) {
    return (
      <>
        {!error && <p className="quiet-note photo-note">Checking the photos on the server…</p>}
        {error instanceof ApiError && error.status === 404 ? null : <ErrorNote error={error} />}
      </>
    );
  }

  const path = detail.data?.path ?? '';
  const src = (rel: string, thumb: boolean) => (path ? mediaUrl(server, joinServerPath(path, rel), thumb) : '');
  const rank = new Map(pick?.kept.map((f, i) => [f, i + 1]) ?? []);
  const why = new Map(pick?.left_out.map(l => [l.file, l.reason]) ?? []);
  const photos: GridPhoto[] = review.items.map(it => ({
    key: it.file,
    file: it.file,
    src: src(it.file, true),
    full: src(it.file, false),
    item: it,
    leftOut: false,
    rank: rank.get(it.file),
    reason: why.get(it.file),
    stored: true,
  }));
  // a photo left out from the compare dialog is in .set_aside/ now
  const photoFor = (file: string): GridPhoto | undefined => {
    const p = photos.find(x => x.file === file);
    if (p) return p;
    const s = review.set_aside.find(x => x.from === file);
    return s && { key: s.file, file, src: src(s.file, true), full: src(s.file, false), leftOut: true, stored: true };
  };
  const restore = (file: string) => {
    const s = review.set_aside.find(x => x.from === file || x.file === file);
    return run(async () => {
      await fix([{ op: 'restore', files: [s?.file ?? file] }]);
    });
  };
  const leaveOut = (files: string[]) =>
    run(async () => {
      await fix([{ op: 'set_aside', files }]);
      setUndo(null);
    });
  const onAction = (a: PhotoAction, file: string) => {
    if (a === 'leave_out') leaveOut([file]);
    else if (a === 'restore') restore(file);
    else if (a === 'keep' || a === 'intended') run(() => settings({ decisions: { [file]: a } }));
    else if (a === 'unkeep' || a === 'unintend') run(() => settings({ decisions: { [file]: null } }));
    else if (a === 'reference') run(() => settings({ reference: file }));
    else if (a === 'compare') {
      const other = partnerOf(review.items.find(i => i.file === file), review.groups);
      if (other) setCompare([file, other]);
    }
  };
  const bulk = (ops: PhotoFixOp[]) =>
    run(async () => {
      const r = await fix(ops);
      const moved = asideOf(r);
      setUndo(moved.length ? moved : null);
    });
  const onSuggested = () => {
    const s = suggested(photos);
    if (s.flags.length) bulk([{ op: 'set_aside_flagged', flags: s.flags }]);
  };
  const onPick = (n: number) =>
    run(async () => {
      const r = await checkPhotos(server, name, { resolution: k.resolution, pick: n, wait_ms: 0 });
      ++seq.current;
      setReview(r);
      setPick(r.pick ?? null);
    });
  const keepPick = () => pick && bulk([{ op: 'keep_best', n: pick.n }]);
  const onUndo = () =>
    undo &&
    run(async () => {
      await fix([{ op: 'restore', files: undo }]);
      setUndo(null);
    });

  const person = review.kind === 'person';
  const originals = review.originals ?? []; // an older server lists them under set_aside
  const a = compare && photoFor(compare[0]);
  const b = compare && photoFor(compare[1]);
  return (
    <>
      <div className="tally">
        <p>
          <strong className="num">{review.total}</strong> photos on the server
          {review.pending > 0 && <span className="muted num"> · {review.pending} still being checked</span>}
          {review.flagged > 0 && (
            <>
              {' · '}
              <button type="button" className="link" onClick={() => setFilter(f => (f === 'flagged' ? null : 'flagged'))}>
                {review.flagged} to look at
              </button>
            </>
          )}
        </p>
      </div>
      <FaceCheck server={server} dataset={name} review={review} onChange={() => load(0)} />
      <PhotoSummary
        review={review}
        photos={photos}
        filter={filter}
        setFilter={setFilter}
        onSuggested={onSuggested}
        onPick={onPick}
        undo={undo ? onUndo : undefined}
        busy={busy}
        pick={pick}
        onKeepPick={keepPick}
        onCancelPick={() => setPick(null)}
      />
      <p className="quiet-note photo-note">Sunglasses, group photos, outfits and places are checked after captions or the face check.</p>
      <Readiness review={review} onOp={op => (op.op === 'keep_best' ? onPick(op.n ?? 25) : bulk([op]))} busy={busy} />
      {person && (
        <FaceCropOffer
          review={review}
          onPreview={() => fixPhotos(server, name, { ops: [{ op: 'face_crops' }], resolution: k.resolution })}
          onApply={async () => {
            await fix([{ op: 'face_crops' }]);
          }}
          onRemove={async () => {
            await fix([{ op: 'remove_face_crops' }]);
          }}
        />
      )}
      <ErrorNote error={error} />
      {photos.length > 0 && <PhotoGrid photos={photos} filter={filter} onToggle={f => leaveOut([f])} onAction={onAction} person={person} busy={busy} groups={review.groups} picking={!!pick} />}
      <DecidePairs review={review} photos={photos} onToggle={f => leaveOut([f])} onCompare={(x, y) => setCompare([x, y])} busy={busy} />
      {review.set_aside.length > 0 && (
        <details className="photo-aside">
          <summary>Left out ({review.set_aside.length})</summary>
          <p className="quiet-note">In the dataset's .set_aside folder: the trainer and the captioner skip it, and nothing is deleted.</p>
          <ul className="photo-aside-list">
            {review.set_aside.map(s => (
              <li key={s.file}>
                <PhotoThumb photo={{ key: s.file, file: s.from, src: src(s.file, true), leftOut: true, stored: true }} />
                <span className="mono photo-aside-name" title={s.file}>
                  {s.from}
                </span>
                <button type="button" className="btn btn-sm" disabled={busy} onClick={() => restore(s.file)}>
                  Restore
                </button>
              </li>
            ))}
          </ul>
          {review.set_aside.length > 1 && (
            <button type="button" className="btn btn-sm btn-ghost" disabled={busy} onClick={() => run(async () => void (await fix([{ op: 'restore', all: true }])))}>
              Restore all
            </button>
          )}
        </details>
      )}
      {originals.length > 0 && (
        <details className="photo-aside">
          <summary>Originals kept ({originals.length})</summary>
          <p className="quiet-note">
            The files a convert or a turn upright replaced, kept in the .set_aside folder as a backup. They are not left out, so Restore all leaves them. Put back swaps one in for
            the photo made from it.
          </p>
          <ul className="photo-aside-list">
            {originals.map(o => (
              <li key={o.file}>
                <PhotoThumb photo={{ key: o.file, file: o.from, src: src(o.file, true), leftOut: true, stored: true }} />
                <span className="photo-aside-name" title={o.file}>
                  <span className="mono">{o.from}</span>
                  <span className="muted"> · original of {o.of}</span>
                </span>
                <button type="button" className="btn btn-sm" disabled={busy} onClick={() => restore(o.file)}>
                  Put back
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}
      {a && b && <CompareDialog a={a} b={b} onClose={() => setCompare(null)} onLeaveOut={f => leaveOut([f])} onKeep={restore} />}
    </>
  );
}

// ---- 3. captions ----

function StepCaptions({ preset, k, dataset, next, back }: { preset: Preset; k: Knobs; dataset: string; next: () => void; back: () => void }) {
  const server = useServer()!;
  const detail = useLoad(server, s => get<DatasetDetail>(s, `/v1/datasets/${encodeURIComponent(dataset)}?limit=1000`), [dataset]);
  const [fallback, setFallback] = useState('a photo of [trigger]');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [copied, setCopied] = useState(false);
  const template = preset.captionTemplates?.[k.subject];
  const items = detail.data?.items ?? [];
  const missing = items.filter(i => !i.caption?.trim());
  const job = detail.data?.caption_job ?? null;
  const running = !!job && captionBusy.has(job.state);
  // what is wrong with the captions, re-read whenever they change
  const captionsKey = items.map(i => i.file + '\u0000' + (i.caption ?? '')).join('\u0001');
  const check = useLoad(server, s => get<CaptionCheck>(s, `/v1/datasets/${encodeURIComponent(dataset)}/captions/check?trigger=${encodeURIComponent(k.trigger)}`), [dataset, k.trigger, captionsKey]);
  const flagsOf = useMemo(() => new Map((check.data?.items ?? []).map(i => [i.file, i.flags])), [check.data]);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  // the photo check over the captions (sunglasses, other people, …) and the readiness line
  const photo = useLoad(server, s => checkPhotos(s, dataset, { resolution: k.resolution }).catch(() => null), [dataset, k.resolution.join(','), captionsKey]);
  const mentions = photo.data
    ? (Object.entries(mentionWhat) as [PhotoFlag, string][])
        .map(([flag, what]) => ({ flag, what, files: photo.data!.items.filter(i => i.flags.includes(flag)).map(i => i.file) }))
        .filter(m => m.files.length > 0)
    : [];
  const [only, setOnly] = useState<{ flag: PhotoFlag; files: Set<string> } | null>(null);
  const [askOut, setAskOut] = useState<{ what: string; files: string[] } | null>(null);
  // nothing left flagged: the filter lets everything through again
  const shown = only
    ? items.filter(i => only.files.has(i.file))
    : onlyFlagged && (check.data?.flagged ?? 0) > 0
      ? items.filter(i => (flagsOf.get(i.file) ?? []).length > 0)
      : items;
  const triggerIssues = (check.data?.counts.no_trigger ?? 0) + (check.data?.counts.trigger_case ?? 0);

  const fixTriggers = async () => {
    setBusy(true);
    setError(null);
    try {
      await post<CaptionFix>(server, `/v1/datasets/${encodeURIComponent(dataset)}/captions/fix`, {
        ops: [{ op: 'fix_trigger_case' }, { op: 'ensure_trigger' }],
        trigger: k.trigger,
        apply: true,
      });
      detail.reload();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  // the captions land while the caption job runs
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => document.visibilityState !== 'hidden' && detail.reload(), 3000);
    return () => clearInterval(id);
  }, [running]); // eslint-disable-line react-hooks/exhaustive-deps

  const saveCaption = async (item: DatasetItem, caption: string) => {
    await call(server, 'PUT', `/v1/datasets/${encodeURIComponent(dataset)}/captions/${item.file.split('/').map(encodeURIComponent).join('/')}`, {
      caption,
    });
    detail.setData(d => (d ? { ...d, items: d.items.map(i => (i.file === item.file ? { ...i, caption } : i)) } : d));
  };

  // leaving out moves photos (and their captions) into .set_aside/; restore is in the Images step
  const leaveOut = async (files: string[]) => {
    setBusy(true);
    setError(null);
    try {
      await fixPhotos(server, dataset, { ops: [{ op: 'set_aside', files }], apply: true, resolution: k.resolution });
      setOnly(null);
      setAskOut(null);
      detail.reload();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const fillMissing = async () => {
    setBusy(true);
    setError(null);
    try {
      for (const item of missing) await saveCaption(item, fallback);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <WizardStep title="Captions" description={photo.data ? <Readiness review={photo.data} compact /> : undefined}>
      <AutoCaption dataset={dataset} template={template} job={job} onChange={detail.reload} />
      {template && (
        <details className="template">
          <summary className="template-head">
            <p>
              Caption prompt for your own VLM <span className="muted">· {k.subject}</span>
            </p>
            <button
              className="btn btn-sm"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(template);
                  setCopied(true);
                  setTimeout(() => setCopied(false), 1500);
                } catch {
                  /* clipboard blocked */
                }
              }}
            >
              {copied ? <Check /> : <Copy />}
              {copied ? 'Copied' : 'Copy prompt'}
            </button>
          </summary>
          <pre className="code-block">{template}</pre>
        </details>
      )}
      <div className="tally">
        <p>
          <strong className="num">{items.length - missing.length}</strong> of {items.length} captioned
          {check.data && check.data.flagged > 0 && (
            <>
              {' · '}
              <button type="button" className="link" onClick={() => setOnlyFlagged(v => !v)}>
                {onlyFlagged ? 'show all' : `${check.data.flagged} to look at`}
              </button>
            </>
          )}
        </p>
        {triggerIssues > 0 && (
          <button className="btn btn-sm" disabled={busy || running} onClick={fixTriggers} title="[Trigger] and wrong-case trigger words become exact; captions without one get [trigger] first">
            Fix the trigger in {triggerIssues}
          </button>
        )}
        {missing.length > 0 && (
          <div className="fill">
            <label htmlFor="fill-missing">Fill {missing.length} missing with</label>
            <input id="fill-missing" value={fallback} onChange={e => setFallback(e.target.value)} />
            <button className="btn" disabled={busy || running || !fallback.trim()} onClick={fillMissing} title={running ? 'Wait for the caption run to finish' : undefined}>
              {busy ? 'Saving…' : 'Fill'}
            </button>
          </div>
        )}
      </div>
      {check.data && check.data.flagged > 0 && (
        <p className="flag-summary">
          {(Object.entries(check.data.counts) as [CaptionFlag, number][]).map(([f, n]) => (
            <span key={f} className="badge tone-warn" title={flagHelp[f]}>
              {n} {flagLabel[f]}
            </span>
          ))}
        </p>
      )}
      {mentions.map(m => (
        <p key={m.flag} className="photo-mention">
          <span>
            Captions mention {m.what} in {m.files.length === 1 ? '1 photo' : `${m.files.length} photos`}
          </span>
          <button type="button" className="link" onClick={() => setOnly(o => (o?.flag === m.flag ? null : { flag: m.flag, files: new Set(m.files) }))}>
            {only?.flag === m.flag ? 'Show all' : 'Show'}
          </button>
          <button type="button" className="link" disabled={busy || running} onClick={() => setAskOut({ what: m.what, files: m.files })}>
            Leave them out
          </button>
        </p>
      ))}
      {askOut && (
        <Confirm
          title={`Leave out ${askOut.files.length === 1 ? '1 photo' : `${askOut.files.length} photos`}?`}
          confirm="Leave them out"
          busy={busy}
          onConfirm={() => leaveOut(askOut.files)}
          onClose={() => setAskOut(null)}
        >
          Their captions mention {askOut.what}. The photos and their captions move to the dataset's .set_aside folder, which the trainer skips. Nothing is deleted:
          the Images step puts them back.
        </Confirm>
      )}
      <ErrorNote error={error || detail.error} />
      <div className="caption-grid">
        {shown.map(item => (
          <CaptionCard
            key={item.file}
            item={item}
            src={detail.data ? mediaUrl(server, joinServerPath(detail.data.path, item.file), true) : ''}
            onSave={saveCaption}
            locked={running}
            flags={flagsOf.get(item.file)}
            onLeaveOut={photo.data ? () => leaveOut([item.file]) : undefined}
          />
        ))}
      </div>
      <div className="wizard-nav">
        <button className="btn btn-ghost" onClick={back}>
          <ArrowLeft />
          Back
        </button>
        <button className="btn btn-primary" disabled={!items.length} onClick={next}>
          Next: settings
          <ArrowRight />
        </button>
      </div>
    </WizardStep>
  );
}

const captionBusy = new Set(['queued', 'starting', 'running', 'stopping']);

/** What the photo check found the captions mention (caption-word flags, PHOTO-PLAN 4.4). */
const mentionWhat: Partial<Record<PhotoFlag, string>> = {
  mentions_covered: 'sunglasses, masks or hats',
  mentions_people: 'other people',
  mentions_text: 'text or a logo',
  mentions_tone: 'black and white',
  mentions_screen: 'a screen',
};

const flagLabel: Record<CaptionFlag, string> = {
  missing: 'missing',
  no_trigger: 'without the trigger',
  trigger_case: 'with a misspelled trigger',
  truncated: 'cut off',
  hedging: 'unsure',
  preamble: 'with a preamble',
  duplicate: 'duplicate',
};
const flagHelp: Record<CaptionFlag, string> = {
  missing: 'No caption: the trainer uses the default one.',
  no_trigger: 'No [trigger] and no trigger word: the trainer puts the trigger word first itself.',
  trigger_case: '[Trigger] or a differently cased trigger word: matched exactly, so it is not replaced and the trigger is added again.',
  truncated: 'Ends mid-sentence: the captioner ran out of tokens.',
  hedging: '"appears to be", "possibly": the model was unsure.',
  preamble: '"Here is a caption", "This image shows".',
  duplicate: 'The same text as another caption.',
};
const CAPTION = 'config.process[0].caption.';

// Runs one of the server's captioner models on the dataset (a caption job);
// the preset's caption prompt is the starting prompt.
function AutoCaption({ dataset, template, job, onChange }: { dataset: string; template?: string; job: Job | null; onChange: () => void }) {
  const server = useServer()!;
  const cat = useLoad(server, s => get<CaptionCatalog>(s, '/v1/templates/captioners'), []);
  const [name, setName] = useState('');
  const [model, setModel] = useState('');
  const [prompt, setPrompt] = useState(template ?? '');
  const [recaption, setRecaption] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const c = cat.data?.captioners.find(x => x.name === name);
  const running = !!job && captionBusy.has(job.state);

  const choose = (next: Captioner, keepPrompt: boolean) => {
    setName(next.name);
    setModel(String(next.defaults[CAPTION + 'model_name_or_path'] ?? next.model_options[0]?.value ?? ''));
    if (!keepPrompt) setPrompt(String(next.defaults[CAPTION + 'caption_prompt'] ?? ''));
  };
  useEffect(() => {
    const list = cat.data?.captioners ?? [];
    const first = list.find(x => x.name === 'Qwen3VLCaptioner') ?? list[0];
    if (first && !name) choose(first, !!template);
  }, [cat.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async () => {
    if (!c) return;
    setBusy(true);
    setError(null);
    try {
      await post(server, `/v1/datasets/${encodeURIComponent(dataset)}/caption`, {
        captioner: c.name,
        model,
        prompt: c.additional_sections.includes('caption.caption_prompt') ? prompt : undefined,
        recaption,
        // the preset's prompts ask for a detailed caption; 128 tokens cut them off
        overrides: c.additional_sections.includes('caption.max_new_tokens') && Number(c.defaults[CAPTION + 'max_new_tokens'] ?? 0) < 384 ? { 'caption.max_new_tokens': 384 } : undefined,
      });
      onChange();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  const stop = async () => {
    if (!job) return;
    try {
      await post(server, `/v1/jobs/${encodeURIComponent(job.id)}/stop`, {});
      onChange();
    } catch (e) {
      setError(e);
    }
  };

  if (cat.error || (cat.data && !cat.data.captioners.length)) return null;
  const prompts: [string, string][] = [...(template ? [['This preset', template] as [string, string]] : []), ...Object.entries(c?.caption_prompts ?? {})];
  return (
    <div className="card auto-caption">
      <div className="auto-caption-head">
        <Sparkles aria-hidden />
        <p>Caption with a model on the server</p>
      </div>
      {running && job ? (
        <div className="auto-caption-run">
          <p className="auto-caption-status">
            {job.state === 'queued' ? 'Waiting for the GPU' : job.info || 'Captioning'}
            {job.total_steps ? (
              <span className="muted num">
                {' '}
                · {job.step} / {job.total_steps}
              </span>
            ) : null}
          </p>
          <Progress value={job.step} max={job.total_steps} indeterminate={!job.total_steps} />
          <button className="btn btn-sm" onClick={stop} disabled={job.state === 'stopping'}>
            <Square />
            Stop
          </button>
        </div>
      ) : (
        c && (
          <>
            <div className="auto-caption-grid">
              <label className="field">
                <span>Captioner</span>
                <select value={name} onChange={e => choose(cat.data!.captioners.find(x => x.name === e.target.value)!, !!template)}>
                  {cat.data!.captioners.map(x => (
                    <option key={x.name} value={x.name}>
                      {x.label} · {x.group}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Model</span>
                {c.model_options.length ? (
                  <select value={model} onChange={e => setModel(e.target.value)}>
                    {c.model_options.map(o => (
                      <option key={String(o.value)} value={String(o.value)}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input value={model} onChange={e => setModel(e.target.value)} spellCheck={false} />
                )}
              </label>
            </div>
            {c.additional_sections.includes('caption.caption_prompt') && (
              <label className="field">
                <span>Prompt</span>
                {prompts.length > 1 && (
                  <select value={prompts.find(([, v]) => v === prompt)?.[0] ?? ''} onChange={e => setPrompt(prompts.find(([k]) => k === e.target.value)?.[1] ?? prompt)} aria-label="Prompt preset">
                    {!prompts.some(([, v]) => v === prompt) && <option value="">Your own</option>}
                    {prompts.map(([k]) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                  </select>
                )}
                <textarea rows={5} value={prompt} onChange={e => setPrompt(e.target.value)} />
                <small>
                  <span className="mono">[trigger]</span> in a caption becomes the trigger word when training.
                </small>
              </label>
            )}
            <div className="auto-caption-foot">
              <label className="check">
                <input type="checkbox" checked={recaption} onChange={e => setRecaption(e.target.checked)} />
                Replace existing captions
              </label>
              <button className="btn btn-primary" onClick={start} disabled={busy || !model.trim()}>
                <Sparkles />
                {busy ? 'Starting…' : 'Caption'}
              </button>
            </div>
            {job?.state === 'failed' && <Callout kind="danger">The last caption run failed{job.info ? `: ${job.info}` : ''}.</Callout>}
          </>
        )
      )}
      <ErrorNote error={error} />
    </div>
  );
}

// locked: a caption job is writing captions; an edit now would race it
function CaptionCard({
  item,
  src,
  onSave,
  locked,
  flags,
  onLeaveOut,
}: {
  item: DatasetItem;
  src: string;
  onSave: (i: DatasetItem, c: string) => Promise<void>;
  locked?: boolean;
  flags?: CaptionFlag[];
  onLeaveOut?: () => void;
}) {
  const [text, setText] = useState(item.caption ?? '');
  const [state, setState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const saved = useRef(item.caption ?? '');
  useEffect(() => {
    // a caption written elsewhere (the caption job) replaces this one unless it is being edited
    const fresh = item.caption ?? '';
    setText(cur => (cur === saved.current ? fresh : cur));
    saved.current = fresh;
  }, [item.caption]);
  const commit = async () => {
    if (text === saved.current) return;
    setState('saving');
    try {
      await onSave(item, text);
      saved.current = text;
      setState('saved');
      setTimeout(() => setState('idle'), 1200);
    } catch {
      setState('error');
    }
  };
  return (
    <figure className="caption-card">
      <img src={src} alt="" loading="lazy" />
      <figcaption>
        <div className="caption-meta">
          <span className="mono caption-file" title={item.file}>
            {item.file}
          </span>
          {state !== 'idle' ? (
            <span className={`save-state save-${state}`}>{state === 'saving' ? 'Saving…' : state === 'saved' ? 'Saved' : 'Not saved'}</span>
          ) : !text.trim() ? (
            <span className="badge tone-warn">No caption</span>
          ) : (
            flags &&
            flags.length > 0 && (
              <span className="badge tone-warn" title={flags.map(f => flagHelp[f]).join(' ')}>
                {flags.map(f => flagLabel[f]).join(', ')}
              </span>
            )
          )}
          {onLeaveOut && (
            <PhotoMenu label={`More for ${item.file}`} disabled={locked}>
              <button role="menuitem" onClick={onLeaveOut}>
                Leave out
              </button>
            </PhotoMenu>
          )}
        </div>
        <textarea value={text} rows={3} placeholder="No caption; the default caption is used" readOnly={locked} onChange={e => setText(e.target.value)} onBlur={commit} />
      </figcaption>
    </figure>
  );
}

// ---- 4. settings ----

const resolutionChoices: { label: string; value: number[] }[] = [
  { label: '1024 only', value: [1024] },
  { label: '768 + 1024', value: [768, 1024] },
  { label: '512 + 768 + 1024', value: [512, 768, 1024] },
];

function Num({ label, value, onChange, hint, step = 1 }: { label: string; value: number; onChange: (n: number) => void; hint?: string; step?: number }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" step={step} value={value} onChange={e => onChange(Number(e.target.value))} />
      {hint && <small>{hint}</small>}
    </label>
  );
}

function StepSettings({ preset, k, set, next, back }: { preset: Preset; k: Knobs; set: (p: Partial<Knobs>) => void; next: () => void; back: () => void }) {
  const saves = Math.floor(k.steps / Math.max(1, k.saveEvery));
  // a preset's own resolution is offered even when it is not a standard choice
  const own = preset.defaults.resolution;
  const choices = resolutionChoices.some(r => r.value.join() === own.join())
    ? resolutionChoices
    : [{ label: own.join(' + '), value: own }, ...resolutionChoices];
  const lokr = String(preset.base.config.process[0]?.network?.type ?? '').toLowerCase() === 'lokr';
  return (
    <WizardStep title="Settings" description={`Starting from ${preset.name}. Only the knobs you usually change per run are here.`}>
      <div className="form-section">
        <h3>Training</h3>
        <div className="form-grid">
          <Num label="Steps" value={k.steps} onChange={steps => set({ steps })} />
          <Num label="Save every" value={k.saveEvery} onChange={saveEvery => set({ saveEvery })} hint={`${saves} checkpoints to pick from`} />
          <label className="field">
            <span>Optimizer</span>
            <select value={k.optimizer} onChange={e => set({ optimizer: e.target.value })}>
              {Array.from(new Set(['automagic2', 'automagic3', 'adamw8bit', 'adafactor', k.optimizer])).map(o => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </label>
          <Num label="Rank (linear / alpha)" value={k.rank} onChange={rank => set({ rank })} />
          {lokr && <Num label="LoKr factor" value={k.lokrFactor} onChange={lokrFactor => set({ lokrFactor })} hint="-1 = auto" />}
          <fieldset className="field field-wide">
            <span>Resolution</span>
            <div className="seg">
              {choices.map(r => (
                <button key={r.label} type="button" className={r.value.join() === k.resolution.join() ? 'is-on' : ''} onClick={() => set({ resolution: r.value })}>
                  {r.label}
                </button>
              ))}
            </div>
          </fieldset>
        </div>
      </div>
      <div className="form-section">
        <h3>Memory</h3>
        <div className="checks">
          <label className="check">
            <input type="checkbox" checked={k.cacheTextEmbeddings} onChange={e => set({ cacheTextEmbeddings: e.target.checked })} />
            <span>Cache text embeddings</span>
          </label>
          <label className="check">
            <input type="checkbox" checked={k.lowVram} onChange={e => set({ lowVram: e.target.checked })} />
            <span>Low VRAM</span>
          </label>
        </div>
      </div>
      <div className="form-section">
        <h3>Samples</h3>
        <div className="checks">
          <label className="check">
            <input type="checkbox" checked={k.sampling} onChange={e => set({ sampling: e.target.checked })} />
            <span>Generate samples while training</span>
          </label>
        </div>
        {k.sampling && (
          <>
            <div className="form-grid">
              <Num label="Sample every" value={k.sampleEvery} onChange={sampleEvery => set({ sampleEvery })} />
            </div>
            <div className="prompts">
              <p className="label">Sample prompts</p>
              {k.prompts.map((p, i) => (
                <div key={i} className="prompt-row">
                  <span className="prompt-n num">{i + 1}</span>
                  <textarea rows={2} value={p} onChange={e => set({ prompts: k.prompts.map((x, j) => (j === i ? e.target.value : x)) })} />
                  <button className="btn btn-ghost btn-icon" onClick={() => set({ prompts: k.prompts.filter((_, j) => j !== i) })} aria-label="Remove prompt" title="Remove prompt">
                    <X />
                  </button>
                </div>
              ))}
              <div>
                <button className="btn" onClick={() => set({ prompts: [...k.prompts, 'A portrait photograph of [trigger]'] })}>
                  Add prompt
                </button>
              </div>
            </div>
          </>
        )}
      </div>
      <div className="wizard-nav">
        <button className="btn btn-ghost" onClick={back}>
          <ArrowLeft />
          Back
        </button>
        <button className="btn btn-primary" onClick={next}>
          Next: review
          <ArrowRight />
        </button>
      </div>
    </WizardStep>
  );
}

// ---- 5. launch ----

function StepLaunch({ preset, k, dataset, back }: { preset: Preset; k: Knobs; dataset: string; back: () => void }) {
  const server = useServer()!;
  const info = useLoad(
    server,
    async s => {
      const [ds, settings, remotes, system] = await Promise.all([
        get<DatasetDetail>(s, `/v1/datasets/${encodeURIComponent(dataset)}?limit=1`),
        get<Settings>(s, '/v1/settings'),
        get<Items<Remote>>(s, '/v1/remotes').catch(() => ({ items: [] as Remote[] })),
        get<SystemInfo>(s, '/v1/system').catch(() => null),
      ]);
      return { ds, settings, remotes: remotes.items, system };
    },
    [dataset],
  );
  const [gpu, setGpu] = useState('0');
  const [syncRemote, setSyncRemote] = useState('');
  const [autoResume, setAutoResume] = useState(true);
  // every checkpoint by default: the best step is picked after training, and a
  // destroyed pod takes its local copies with it
  const [keep, setKeep] = useState('-1');
  const [showConfig, setShowConfig] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const problems = validate(k);

  const config = info.data ? buildConfig(preset, k, { datasetPath: info.data.ds.path, trainingFolder: info.data.settings.training_folder }) : null;
  // the server's own look at this config before a GPU is paid for
  const configKey = config ? JSON.stringify(config) : '';
  const intended: Record<string, unknown> = { auto_resume: autoResume };
  if (syncRemote) intended.sync_remote = syncRemote === '-' ? '' : syncRemote;
  if (syncRemote !== '-') intended.keep_checkpoints = Number(keep);
  const preflight = useLoad(server, s => (config ? post<Preflight>(s, '/v1/jobs/preflight', { config, policy: intended }) : Promise.resolve(null)), [configKey, syncRemote, keep, autoResume]);
  const pf = preflight.data?.checks ?? [];
  const blocked = pf.some(c => c.level === 'block');

  // the job this page created: a retry after a failed start starts it
  // instead of creating it again (which the taken name would refuse)
  const created = useRef<{ id: string; name: string } | null>(null);
  const launch = async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    try {
      const policy: Record<string, unknown> = { auto_resume: autoResume };
      if (syncRemote) policy.sync_remote = syncRemote === '-' ? '' : syncRemote;
      if (syncRemote !== '-') policy.keep_checkpoints = Number(keep);
      let id = created.current?.name === k.name ? created.current.id : '';
      if (id) {
        await call<Job>(server, 'PATCH', `/v1/jobs/${id}`, { gpu_ids: gpu, config, policy });
      } else {
        id = (await post<Job>(server, '/v1/jobs', { name: k.name, gpu_ids: gpu, config, policy })).id;
        created.current = { id, name: k.name };
      }
      await post<Job>(server, `/v1/jobs/${id}/start`);
      const job = { id };
      save(DRAFT_KEY, null);
      go({ page: 'run', id: job.id });
    } catch (e) {
      setError(e);
      setBusy(false);
    }
  };

  return (
    <WizardStep title="Review and launch">
      <ErrorNote error={info.error} />
      {info.data && (
        <>
          <dl className="summary">
            <div>
              <dt>Run</dt>
              <dd>
                {k.name} <span className="muted">· trigger</span> {k.trigger}
              </dd>
            </div>
            <div>
              <dt>Dataset</dt>
              <dd>
                {dataset}{' '}
                <span className="muted">
                  · {info.data.ds.images} images, {info.data.ds.captions} captions
                </span>
              </dd>
            </div>
            <div>
              <dt>Training</dt>
              <dd className="num">
                {k.steps} steps <span className="muted">· save every</span> {k.saveEvery} <span className="muted">·</span> {k.optimizer}{' '}
                <span className="muted">·</span> {k.resolution.join('+')}px
              </dd>
            </div>
            <div>
              <dt>Samples</dt>
              <dd>{k.sampling ? `${k.prompts.length} prompts every ${k.sampleEvery} steps` : 'Off'}</dd>
            </div>
          </dl>
          <div className="form-grid">
            <label className="field">
              <span>GPU</span>
              <select value={gpu} onChange={e => setGpu(e.target.value)}>
                {(info.data.system?.gpus?.length ? info.data.system.gpus : [{ index: 0, name: 'GPU 0' }]).map(g => (
                  <option key={g.index} value={String(g.index)}>
                    {g.index} · {g.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Keep portable on</span>
              <select value={syncRemote} onChange={e => setSyncRemote(e.target.value)}>
                <option value="">Server default</option>
                <option value="-">None</option>
                {info.data.remotes.map(r => (
                  <option key={r.id} value={r.name}>
                    {r.name} ({r.kind})
                  </option>
                ))}
              </select>
              <small>Synced runs continue on another server if this one goes away.</small>
            </label>
            <label className="field">
              <span>Checkpoints kept there</span>
              <select value={keep} onChange={e => setKeep(e.target.value)} disabled={syncRemote === '-'}>
                <option value="-1">All, to pick the best step later</option>
                <option value="3">The newest and 3 before it</option>
                <option value="0">Only the newest</option>
              </select>
              <small>Only the newest is needed to resume.</small>
            </label>
            <label className="check field-wide">
              <input type="checkbox" checked={autoResume} onChange={e => setAutoResume(e.target.checked)} />
              <span>Resume automatically after an interruption</span>
            </label>
          </div>
          {problems.length > 0 && (
            <Callout kind="warn">
              <ul className="problems">
                {problems.map(p => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
            </Callout>
          )}
          {(['block', 'warn'] as const).map(level => {
            const list = pf.filter(c => c.level === level);
            return (
              list.length > 0 && (
                <Callout key={level} kind={level === 'block' ? 'danger' : 'warn'}>
                  <ul className="problems">
                    {list.map(c => (
                      <li key={c.message}>
                        {c.message}
                        {c.fix && <span className="muted"> · {c.fix}</span>}
                      </li>
                    ))}
                  </ul>
                </Callout>
              )
            );
          })}
          {pf.some(c => c.level === 'info') && (
            <ul className="problems preflight-info">
              {pf
                .filter(c => c.level === 'info')
                .map(c => (
                  <li key={c.message}>
                    {c.message}
                    {c.fix && <span className="muted"> · {c.fix}</span>}
                  </li>
                ))}
            </ul>
          )}
          <div className="config-toggle">
            <button className="btn btn-ghost btn-sm" aria-expanded={showConfig} onClick={() => setShowConfig(v => !v)}>
              {showConfig ? <ChevronDown /> : <ChevronRight />}
              {showConfig ? 'Hide' : 'Show'} the job config
            </button>
          </div>
          {showConfig && config && <pre className="code-block config-preview">{YAML.stringify(config, { lineWidth: 0 })}</pre>}
        </>
      )}
      <ErrorNote error={error} />
      <div className="wizard-nav">
        <button className="btn btn-ghost" onClick={back}>
          <ArrowLeft />
          Back
        </button>
        <button className="btn btn-primary" disabled={!config || problems.length > 0 || blocked || busy} onClick={launch} title={blocked ? 'Fix what blocks the run first' : undefined}>
          <Rocket />
          {busy ? 'Starting…' : 'Create and start training'}
        </button>
      </div>
    </WizardStep>
  );
}

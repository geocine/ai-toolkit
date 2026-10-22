import { ArrowLeft, ArrowRight, Check, ChevronDown, ChevronRight, Copy, Rocket, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import YAML from 'yaml';
import { Dropzone } from '../components/Dropzone';
import { Callout, ErrorNote, NeedServer, PageHeader, Progress, useLoad } from '../components/ui';
import {
  ApiError,
  call,
  get,
  mediaUrl,
  post,
  upload,
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

function StepImages({ k, dataset, onDone, back }: { k: Knobs; dataset: string; onDone: (name: string) => void; back: () => void }) {
  const server = useServer()!;
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState(dataset || datasetSafe(k.trigger.toLowerCase()) || datasetSafe(k.name));
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);
  const existing = useLoad(server, s => get<Items<Dataset>>(s, '/v1/datasets'), []);
  const images = files.filter(f => /\.(jpe?g|png|webp|bmp|gif|avif|heic)$/i.test(f.name));
  const captions = files.filter(f => /\.txt$/i.test(f.name));
  const urls = useMemo(() => images.slice(0, 60).map(f => URL.createObjectURL(f)), [images]);
  useEffect(() => () => urls.forEach(u => URL.revokeObjectURL(u)), [urls]);
  const already = existing.data?.items.find(d => d.name === name);

  const go = async () => {
    setError(null);
    try {
      if (!already) {
        try {
          await post(server, '/v1/datasets', { name });
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 409)) throw e;
        }
      }
      if (files.length) {
        setProgress(0);
        await upload(server, `/v1/datasets/${encodeURIComponent(name)}/files`, files, (d, t) => setProgress(d / t));
      }
      onDone(name);
    } catch (e) {
      setError(e);
      setProgress(null);
    }
  };

  return (
    <WizardStep title="Images">
      <div className="form-grid">
        <label className="field">
          <span>Dataset name</span>
          <input value={name} onChange={e => setName(datasetSafe(e.target.value))} />
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
      <Dropzone onFiles={fs => setFiles(prev => [...prev, ...fs])} />
      {files.length > 0 && (
        <>
          <div className="tally">
            <p>
              <strong className="num">{images.length}</strong> images
              {captions.length > 0 && <> · {captions.length} captions</>} · {bytes(files.reduce((n, f) => n + f.size, 0))}
              {images.length > 0 && images.length < 15 && <span className="warn"> · fewer than ~15 images rarely holds a likeness</span>}
              {images.length > 40 && <span className="muted"> · ~25 was enough in your notes</span>}
            </p>
            <button className="btn btn-ghost btn-sm" onClick={() => setFiles([])}>
              Clear
            </button>
          </div>
          <div className="thumb-grid">
            {urls.map(u => (
              <span key={u} className="thumb">
                <img src={u} alt="" />
              </span>
            ))}
          </div>
        </>
      )}
      {progress !== null && (
        <div className="upload-progress">
          <Progress value={Math.round(progress * 100)} max={100} />
          <span className="num">Uploading {Math.round(progress * 100)}%</span>
        </div>
      )}
      <ErrorNote error={error} />
      <div className="wizard-nav">
        <button className="btn btn-ghost" onClick={back}>
          <ArrowLeft />
          Back
        </button>
        <button className="btn btn-primary" disabled={!name || (!files.length && !already) || progress !== null} onClick={go}>
          {files.length ? `Upload ${files.length} files` : 'Use this dataset'}
          <ArrowRight />
        </button>
      </div>
    </WizardStep>
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

  const saveCaption = async (item: DatasetItem, caption: string) => {
    await call(server, 'PUT', `/v1/datasets/${encodeURIComponent(dataset)}/captions/${item.file.split('/').map(encodeURIComponent).join('/')}`, {
      caption,
    });
    detail.setData(d => (d ? { ...d, items: d.items.map(i => (i.file === item.file ? { ...i, caption } : i)) } : d));
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
    <WizardStep title="Captions">
      {template && (
        <div className="template">
          <div className="template-head">
            <p>
              Caption prompt for your VLM <span className="muted">· {k.subject}</span>
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
          </div>
          <pre className="code-block">{template}</pre>
        </div>
      )}
      <div className="tally">
        <p>
          <strong className="num">{items.length - missing.length}</strong> of {items.length} captioned
        </p>
        {missing.length > 0 && (
          <div className="fill">
            <label htmlFor="fill-missing">Fill {missing.length} missing with</label>
            <input id="fill-missing" value={fallback} onChange={e => setFallback(e.target.value)} />
            <button className="btn" disabled={busy || !fallback.trim()} onClick={fillMissing}>
              {busy ? 'Saving…' : 'Fill'}
            </button>
          </div>
        )}
      </div>
      <ErrorNote error={error || detail.error} />
      <div className="caption-grid">
        {items.map(item => (
          <CaptionCard key={item.file} item={item} src={detail.data ? mediaUrl(server, joinServerPath(detail.data.path, item.file), true) : ''} onSave={saveCaption} />
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

function CaptionCard({ item, src, onSave }: { item: DatasetItem; src: string; onSave: (i: DatasetItem, c: string) => Promise<void> }) {
  const [text, setText] = useState(item.caption ?? '');
  const [state, setState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const saved = useRef(item.caption ?? '');
  useEffect(() => {
    setText(item.caption ?? '');
    saved.current = item.caption ?? '';
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
          ) : (
            !text.trim() && <span className="badge tone-warn">No caption</span>
          )}
        </div>
        <textarea value={text} rows={3} placeholder="No caption; the default caption is used" onChange={e => setText(e.target.value)} onBlur={commit} />
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
  // the Krea notes only apply to the built-in preset; imported ones keep their own resolution
  const krea = preset.builtIn;
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
          <Num label="Steps" value={k.steps} onChange={steps => set({ steps })} hint={krea ? 'Your best Krea results came at 2000–2750 of 3000.' : undefined} />
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
            {krea && <small>1024 only was the key in your notes.</small>}
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
  const [showConfig, setShowConfig] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const problems = validate(k);

  const config = info.data ? buildConfig(preset, k, { datasetPath: info.data.ds.path, trainingFolder: info.data.settings.training_folder }) : null;

  const launch = async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    try {
      const policy: Record<string, unknown> = { auto_resume: autoResume };
      if (syncRemote) policy.sync_remote = syncRemote === '-' ? '' : syncRemote;
      const job = await post<Job>(server, '/v1/jobs', { name: k.name, gpu_ids: gpu, config, policy });
      await post<Job>(server, `/v1/jobs/${job.id}/start`);
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
        <button className="btn btn-primary" disabled={!config || problems.length > 0 || busy} onClick={launch}>
          <Rocket />
          {busy ? 'Starting…' : 'Create and start training'}
        </button>
      </div>
    </WizardStep>
  );
}

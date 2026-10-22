// Auto-captioning a dataset with one of upstream's captioner models
// (GET /v1/templates/captioners, POST /v1/datasets/{name}/caption).

import { useEffect, useMemo, useState } from 'react';
import { ArrowRight, Square } from 'lucide-react';
import { api, ApiError } from './api';
import { href } from './router';
import type { CaptionCatalog, Captioner, Job } from './types';
import { Bar, Callout, Field, Modal, Spinner, useToast } from './ui';

const KEY = 'config.process[0].caption.';
const ACTIVE = new Set(['queued', 'starting', 'running', 'stopping']);
export const captionActive = (j: Job | null | undefined) => !!j && ACTIVE.has(j.state);

// what selecting the captioner sets, else upstream's default caption job
function initial(cat: CaptionCatalog, c: Captioner, key: string): unknown {
  if (KEY + key in c.defaults) return c.defaults[KEY + key];
  const p0 = (cat.default_config as any)?.config?.process?.[0];
  return p0?.type === c.name ? p0?.caption?.[key] : undefined;
}

// Qwen3-VL for still images; upstream's default (Qwen3-Omni) when there is video or sound
function pick(cat: CaptionCatalog, media: boolean): Captioner {
  const def = (cat.default_config as any)?.config?.process?.[0]?.type;
  return cat.captioners.find(c => c.name === (media ? def : 'Qwen3VLCaptioner')) ?? cat.captioners.find(c => c.name === def) ?? cat.captioners[0];
}

export function CaptionDialog({ dataset, media, onStarted, onClose }: { dataset: string; media: boolean; onStarted: (j: Job) => void; onClose: () => void }) {
  const [cat, setCat] = useState<CaptionCatalog | null>(null);
  const [name, setName] = useState('');
  const [model, setModel] = useState('');
  const [custom, setCustom] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [maxRes, setMaxRes] = useState('');
  const [tokens, setTokens] = useState('');
  const [recaption, setRecaption] = useState(false);
  const [gpu, setGpu] = useState('0');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get<CaptionCatalog>('/v1/templates/captioners').then(
      c => {
        setCat(c);
        if (c.captioners.length) choose(c, pick(c, media));
      },
      e => setError(e?.message ?? String(e)),
    );
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const c = cat?.captioners.find(x => x.name === name);
  const has = (s: string) => !!c?.additional_sections.includes('caption.' + s);
  const prompts = c ? Object.entries(c.caption_prompts) : [];

  function choose(cc: CaptionCatalog, next: Captioner) {
    setName(next.name);
    const m = String(initial(cc, next, 'model_name_or_path') ?? '');
    setModel(m);
    setCustom(!!m && !next.model_options.some(o => o.value === m));
    setPrompt(String(initial(cc, next, 'caption_prompt') ?? ''));
    setMaxRes(String(initial(cc, next, 'max_res') ?? ''));
    setTokens(String(initial(cc, next, 'max_new_tokens') ?? ''));
  }

  const groups = useMemo(() => {
    const out = new Map<string, Captioner[]>();
    for (const x of cat?.captioners ?? []) out.set(x.group, [...(out.get(x.group) ?? []), x]);
    return [...out];
  }, [cat]);

  const start = async () => {
    if (!c) return;
    setBusy(true);
    setError('');
    const overrides: Record<string, unknown> = {};
    if (has('max_res') && maxRes) overrides['caption.max_res'] = Number(maxRes);
    if (has('max_new_tokens') && tokens) overrides['caption.max_new_tokens'] = Number(tokens);
    try {
      const j = await api.post<Job>(`/v1/datasets/${encodeURIComponent(dataset)}/caption`, {
        captioner: c.name,
        model: model.trim() || undefined,
        prompt: has('caption_prompt') ? prompt : undefined,
        recaption,
        overrides,
        gpu_ids: gpu.trim() || undefined,
      });
      onStarted(j);
      onClose();
    } catch (e) {
      setError(e instanceof ApiError && e.status === 409 ? 'This dataset is already being captioned.' : ((e as Error)?.message ?? String(e)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title="Auto caption" onClose={onClose} wide>
      <div className="modal-body caption-dialog">
        <div className="small muted" style={{ marginBottom: 14 }}>
          Runs a captioner model on this server's GPU and writes a <span className="mono">.txt</span> next to each file. The model downloads on first use.
        </div>
        {!cat && !error && <Spinner />}
        {cat && c && (
          <div className="form-grid caption-form">
            <Field label="Captioner" hint={c.group}>
              <select value={name} onChange={e => choose(cat, cat.captioners.find(x => x.name === e.target.value)!)}>
                {groups.map(([g, list]) => (
                  <optgroup key={g} label={g}>
                    {list.map(x => (
                      <option key={x.name} value={x.name}>
                        {x.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </Field>
            <Field label="Model">
              {custom || c.model_options.length === 0 ? (
                <div className="row" style={{ gap: 8, flexWrap: 'nowrap' }}>
                  <input type="text" value={model} onChange={e => setModel(e.target.value)} placeholder="org/repo on Hugging Face, or a local path" spellCheck={false} />
                  {c.model_options.length > 0 && (
                    <button className="btn" type="button" onClick={() => (setCustom(false), setModel(String(c.model_options[0].value)))}>
                      List
                    </button>
                  )}
                </div>
              ) : (
                <select value={model} onChange={e => (e.target.value === '\u0000' ? setCustom(true) : setModel(e.target.value))}>
                  {c.model_options.map(o => (
                    <option key={String(o.value)} value={String(o.value)}>
                      {o.label}
                    </option>
                  ))}
                  <option value={'\u0000'}>Other…</option>
                </select>
              )}
            </Field>
            {has('caption_prompt') && (
              <div className="caption-span">
                <Field
                  label="Prompt"
                  hint={
                    <>
                      <span className="mono">[trigger]</span> written into a caption becomes the trigger word when training.
                    </>
                  }
                >
                  {prompts.length > 1 && (
                    <select className="sm" value={prompts.find(([, v]) => v === prompt)?.[0] ?? ''} onChange={e => setPrompt(c.caption_prompts[e.target.value] ?? prompt)} aria-label="Prompt preset">
                      {!prompts.some(([, v]) => v === prompt) && <option value="">Your own</option>}
                      {prompts.map(([k]) => (
                        <option key={k} value={k}>
                          {k}
                        </option>
                      ))}
                    </select>
                  )}
                  <textarea rows={6} value={prompt} onChange={e => setPrompt(e.target.value)} />
                </Field>
              </div>
            )}
            {has('max_res') && (
              <Field label="Max resolution">
                <select value={maxRes} onChange={e => setMaxRes(e.target.value)}>
                  {cat.options.max_res?.map(o => (
                    <option key={String(o.value)} value={String(o.value)}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </Field>
            )}
            {has('max_new_tokens') && (
              <Field label="Max caption tokens">
                <select value={tokens} onChange={e => setTokens(e.target.value)}>
                  {cat.options.max_new_tokens?.map(o => (
                    <option key={String(o.value)} value={String(o.value)}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </Field>
            )}
            <Field label="GPU">
              <input type="text" value={gpu} onChange={e => setGpu(e.target.value)} className="mono" />
            </Field>
            <label className="check caption-span">
              <input type="checkbox" checked={recaption} onChange={e => setRecaption(e.target.checked)} /> Replace existing captions (otherwise files with a caption are skipped)
            </label>
          </div>
        )}
        {error && (
          <div style={{ marginTop: 12 }}>
            <Callout tone="danger">{error}</Callout>
          </div>
        )}
      </div>
      <div className="modal-actions">
        <button className="btn" onClick={onClose}>
          Cancel
        </button>
        <button className="btn btn-primary" onClick={start} disabled={!c || busy}>
          {busy ? <Spinner /> : null} Start captioning
        </button>
      </div>
    </Modal>
  );
}

// The dataset's caption job while it runs, or how the last one ended.
export function CaptionRun({ job, onChange }: { job: Job; onChange: () => void }) {
  const toast = useToast();
  const [stopping, setStopping] = useState(false);
  const active = captionActive(job);
  const total = job.total_steps ?? 0;
  if (!active && job.state !== 'failed') return null;
  const stop = async () => {
    setStopping(true);
    try {
      await api.post(`/v1/jobs/${encodeURIComponent(job.id)}/stop`, {});
      onChange();
    } catch (e) {
      toast.error(e);
    } finally {
      setStopping(false);
    }
  };
  if (!active)
    return (
      <div className="caption-run-note">
        <Callout tone="danger">
          Captioning failed{job.info ? `: ${job.info}` : ''}.{' '}
          <a className="link" href={href('jobs', job.id)}>
            Open the log
          </a>
        </Callout>
      </div>
    );
  return (
    <div className="card caption-run">
      <div className="caption-run-head">
        <Spinner />
        <span className="cell-title">{job.state === 'queued' ? 'Waiting for the GPU' : job.info || 'Captioning'}</span>
        <span className="grow" />
        {total > 0 && (
          <span className="small muted num">
            {job.step} / {total}
          </span>
        )}
        <a className="btn btn-sm" href={href('jobs', job.id)}>
          Log <ArrowRight size={14} />
        </a>
        <button className="btn btn-sm" onClick={stop} disabled={stopping || job.state === 'stopping'}>
          <Square size={12} /> Stop
        </button>
      </div>
      {total > 0 && <Bar value={job.step} max={total} />}
    </div>
  );
}

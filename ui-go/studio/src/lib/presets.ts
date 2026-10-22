// Training presets: a full ai-toolkit job config plus the few knobs that
// change run to run. The built-in Krea 2 preset comes from the user's own
// configs (E:\AIML\ideogram\krea_lokr*.yml) with their bugs fixed.
import YAML from 'yaml';
import type { JobConfig } from './api';

export type Subject = 'female' | 'male' | 'man';

export interface Knobs {
  name: string;
  trigger: string;
  subject: Subject;
  steps: number;
  saveEvery: number;
  resolution: number[];
  optimizer: string;
  rank: number;
  lokrFactor: number;
  sampling: boolean;
  sampleEvery: number;
  prompts: string[];
  cacheTextEmbeddings: boolean;
  lowVram: boolean;
}

export interface Preset {
  id: string;
  name: string;
  description: string;
  builtIn: boolean;
  base: JobConfig;
  defaults: Omit<Knobs, 'name' | 'trigger'>;
  promptSets?: Partial<Record<Subject, string[]>>;
  captionTemplates?: Partial<Record<Subject, string>>;
  notes: { title: string; body: string }[];
}

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));

// ---- the user's Krea 2 character LoKr ----

const femalePrompts = [
  'A close-up portrait photograph of [trigger] as a young woman with soft pink cherry blossoms framing her face, gentle smile, and warm spring daylight.',
  'A cozy indoor portrait of [trigger] as a young woman curled up by a rain-streaked window, wearing an oversized cream sweater, holding a mug with both hands and smiling softly at the camera.',
  'A pastel studio portrait of [trigger] as a young woman wearing a delicate flower crown, soft pink backdrop, and a sweet playful expression with a slight head tilt.',
  'A golden-hour outdoor portrait of [trigger] as a young woman laughing with eyes squinted shut, backlit by warm sunset light in a sunlit meadow of wildflowers.',
  'A cute cafe portrait of [trigger] as a young woman sipping bubble tea through a wide straw, looking at the camera with an adorable surprised expression and wide eyes.',
  'A dreamy nighttime portrait of [trigger] as a young woman surrounded by warm fairy lights, soft glowing bokeh, and a gentle shy smile looking slightly off-camera.',
];

const malePrompts = [
  'A close-up portrait photograph of [trigger] as a man with sharp jawline and short styled hair, confident half-smile, soft natural window light, clean modern background.',
  'A casual indoor portrait of [trigger] as a man sitting by a rain-streaked window, wearing a charcoal hoodie, holding a black coffee mug, relaxed expression looking at the camera.',
  'A studio portrait of [trigger] as a man in a fitted black t-shirt, neutral gray backdrop, strong key light, calm composed expression with a slight head tilt.',
  'A golden-hour outdoor portrait of [trigger] as a man laughing with eyes squinted shut, backlit by warm sunset light on a city rooftop with a soft skyline bokeh.',
  'A cafe portrait of [trigger] as a man in a denim jacket, one hand on a ceramic coffee cup, looking at the camera with a subtle smirk and easy confidence.',
  'A nighttime street portrait of [trigger] as a man under neon city lights, cool blue and amber reflections, leather jacket, quiet thoughtful smile looking slightly off-camera.',
];

const captionIntro =
  'Caption this image as a detailed image-generation prompt. Describe all visible details clearly and confidently. ' +
  'Do not use uncertain phrases like “appears to be,” “possibly,” or “seems.”\n\n';
const captionOutro = ' Keep `[trigger]` exactly as written. No preamble. Output only the caption.';

const captionTemplates: Record<Subject, string> = {
  female: captionIntro + 'Start with “A photo of [trigger] as a young female” or “[trigger] as a young woman.”' + captionOutro,
  male: captionIntro + 'Start with “A photo of [trigger] as a young male” or “[trigger] as a young man.”' + captionOutro,
  man: captionIntro + 'Start with “A photo of [trigger] as a man” or “[trigger] as a man.”' + captionOutro,
};

const kreaBase: JobConfig = {
  job: 'extension',
  config: {
    name: 'my_lokr',
    process: [
      {
        type: 'diffusion_trainer',
        training_folder: 'output',
        sqlite_db_path: './aitk_db.db',
        device: 'cuda',
        trigger_word: 'trigger',
        performance_log_every: 10,
        network: {
          type: 'lokr',
          linear: 32,
          linear_alpha: 32,
          lokr_full_rank: true,
          lokr_factor: 16,
          network_kwargs: { ignore_if_contains: [] },
        },
        save: { dtype: 'bf16', save_every: 150, max_step_saves_to_keep: 15, save_format: 'diffusers', push_to_hub: false },
        datasets: [
          {
            folder_path: '',
            mask_path: null,
            mask_min_value: 0.1,
            default_caption: '',
            caption_ext: 'txt',
            caption_dropout_rate: 0.05,
            cache_latents_to_disk: true,
            is_reg: false,
            network_weight: 1,
            resolution: [1024],
            controls: [],
            shrink_video_to_frames: true,
            num_frames: 1,
            flip_x: false,
            flip_y: false,
            num_repeats: 1,
          },
        ],
        train: {
          batch_size: 1,
          bypass_guidance_embedding: false,
          steps: 3000,
          gradient_accumulation: 1,
          train_unet: true,
          train_text_encoder: false,
          gradient_checkpointing: true,
          noise_scheduler: 'flowmatch',
          optimizer: 'automagic2',
          timestep_type: 'sigmoid',
          content_or_style: 'balanced',
          optimizer_params: { weight_decay: 0.0001 },
          unload_text_encoder: false,
          cache_text_embeddings: false,
          lr: 0.0001,
          // fix: EMA is read from ema_config only (toolkit EMAConfig)
          ema_config: { use_ema: true, ema_decay: 0.99 },
          skip_first_sample: true,
          force_first_sample: false,
          disable_sampling: false,
          dtype: 'bf16',
          diff_output_preservation: false,
          diff_output_preservation_multiplier: 1,
          diff_output_preservation_class: 'person',
          switch_boundary_every: 1,
          // fix: these are TrainConfig keys; at process level they were ignored
          loss_type: 'mse',
          do_differential_guidance: true,
          differential_guidance_scale: 3,
        },
        logging: { log_every: 1, use_ui_logger: true },
        model: {
          name_or_path: 'krea/Krea-2-Raw',
          quantize: true,
          qtype: 'qfloat8',
          quantize_te: true,
          qtype_te: 'qfloat8',
          arch: 'krea2',
          low_vram: true,
          model_kwargs: {},
          compile: false,
          layer_offloading: false,
          layer_offloading_text_encoder_percent: 1,
          layer_offloading_transformer_percent: 1,
        },
        sample: {
          sampler: 'flowmatch',
          sample_every: 250,
          width: 1024,
          height: 1024,
          samples: femalePrompts.map(prompt => ({ prompt })),
          neg: '',
          seed: 42,
          walk_seed: true,
          guidance_scale: 4,
          sample_steps: 30,
          num_frames: 1,
          fps: 1,
        },
      },
    ],
  },
  meta: { name: '[name]', version: '1.0' },
};

export const kreaPreset: Preset = {
  id: 'builtin-krea2-lokr',
  name: 'Krea 2 · character LoKr',
  description: 'Face / character LoKr on Krea 2 Raw, ~25 images, Automagic, sigmoid timesteps, 1024 only.',
  builtIn: true,
  base: kreaBase,
  defaults: {
    subject: 'female',
    steps: 3000,
    saveEvery: 150,
    resolution: [1024],
    optimizer: 'automagic2',
    rank: 32,
    lokrFactor: 16,
    sampling: true,
    sampleEvery: 250,
    prompts: femalePrompts,
    cacheTextEmbeddings: false,
    lowVram: true,
  },
  promptSets: { female: femalePrompts, male: malePrompts, man: malePrompts },
  captionTemplates,
  notes: [
    {
      title: 'Differential guidance was off in krea_lokr.yml',
      body:
        '`loss_type`, `do_differential_guidance` and `differential_guidance_scale` sat at the process level. ' +
        'The trainer only reads them from `train` (TrainConfig), so they were silently ignored. Moved into `train`.',
    },
    {
      title: 'EMA was off in both files',
      body:
        'EMA is read only from `train.ema_config: {use_ema, ema_decay}` (EMAConfig). The files set `ema_config: null` ' +
        'plus loose `use_ema` / `ema_decay` (`ma_decay`, a typo, in the female file), which the trainer never reads. ' +
        'The preset uses `ema_config: {use_ema: true, ema_decay: 0.99}` as intended.',
    },
    {
      title: 'From your notes',
      body:
        '~25 images is enough; 1024 only was the key (768+1024 is an option); 3000 steps with the best checkpoints ' +
        'around 2000–2750, so every 150-step save is kept for picking; sampling is optional and slow.',
    },
  ],
};

export const builtInPresets: Preset[] = [kreaPreset];

// ---- knobs <-> config ----

const proc = (c: JobConfig) => c.config.process[0];

/** Reads the knobs out of any ai-toolkit config (for imported presets). */
export function readKnobs(c: JobConfig): Omit<Knobs, 'name' | 'trigger'> & { name: string; trigger: string } {
  const p = proc(c) ?? {};
  const train = p.train ?? {};
  const net = p.network ?? {};
  const sample = p.sample ?? {};
  const ds = (p.datasets ?? [])[0] ?? {};
  const samples: { prompt?: string }[] = Array.isArray(sample.samples)
    ? sample.samples
    : Array.isArray(sample.prompts)
      ? sample.prompts.map((prompt: string) => ({ prompt }))
      : [];
  return {
    name: c.config.name ?? '',
    trigger: p.trigger_word ?? '',
    subject: 'female',
    steps: Number(train.steps ?? 3000),
    saveEvery: Number(p.save?.save_every ?? 250),
    resolution: Array.isArray(ds.resolution) ? ds.resolution.map(Number) : [1024],
    optimizer: String(train.optimizer ?? 'adamw8bit'),
    rank: Number(net.linear ?? 32),
    lokrFactor: Number(net.lokr_factor ?? -1),
    sampling: !train.disable_sampling,
    sampleEvery: Number(sample.sample_every ?? 250),
    prompts: samples.map(s => s.prompt ?? '').filter(Boolean),
    cacheTextEmbeddings: Boolean(train.cache_text_embeddings),
    lowVram: Boolean(p.model?.low_vram),
  };
}

export interface BuildContext {
  datasetPath: string;
  trainingFolder?: string;
}

/** The job config for a run: the preset's base with the knobs applied. */
export function buildConfig(preset: Preset, k: Knobs, ctx: BuildContext): JobConfig {
  const c = clone(preset.base);
  const p = proc(c);
  c.config.name = k.name;
  p.trigger_word = k.trigger || null;
  if (ctx.trainingFolder) p.training_folder = ctx.trainingFolder;
  p.datasets = p.datasets?.length ? p.datasets : [{ caption_ext: 'txt' }];
  p.datasets[0].folder_path = ctx.datasetPath;
  p.datasets[0].resolution = [...k.resolution];
  p.train = { ...(p.train ?? {}), steps: k.steps, optimizer: k.optimizer, cache_text_embeddings: k.cacheTextEmbeddings, disable_sampling: !k.sampling };
  p.save = { ...(p.save ?? {}), save_every: k.saveEvery };
  p.network = { ...(p.network ?? {}), linear: k.rank, linear_alpha: k.rank };
  if ((p.network.type ?? '').toLowerCase() === 'lokr') p.network.lokr_factor = k.lokrFactor;
  p.model = { ...(p.model ?? {}), low_vram: k.lowVram };
  p.sample = { ...(p.sample ?? {}), sample_every: k.sampleEvery, samples: k.prompts.filter(s => s.trim()).map(prompt => ({ prompt })) };
  delete p.sample.prompts;
  return c;
}

/** Problems worth stopping for before a run is created. */
export function validate(k: Knobs): string[] {
  const out: string[] = [];
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(k.name)) out.push('Name: letters, digits, dot, dash or underscore, starting with a letter or digit.');
  if (!k.trigger.trim()) out.push('Trigger word is empty: captions and prompts use [trigger].');
  if (k.steps < 1) out.push('Steps must be positive.');
  if (k.saveEvery < 1 || k.saveEvery > k.steps) out.push('Save every must be between 1 and the step count.');
  if (!k.resolution.length) out.push('Pick at least one resolution.');
  if (k.sampling && !k.prompts.some(p => p.trim())) out.push('Sampling is on but there are no prompts.');
  return out;
}

// ---- import / export ----

/**
 * Parses an ai-toolkit job config (YAML, or JSON when `json`) and fixes the
 * two traps from the user's Krea files, with a note per fix.
 */
export function parseConfig(text: string, json = false): { cfg: JobConfig; notes: Preset['notes'] } {
  const parsed = json ? JSON.parse(text) : YAML.parse(text);
  const cfg: JobConfig = parsed?.config?.process ? parsed : parsed?.job_config?.config?.process ? parsed.job_config : null;
  if (!cfg || !Array.isArray(cfg.config.process) || !cfg.config.process[0]) {
    throw new Error('Not an ai-toolkit job config: expected config.process[0].');
  }
  const p = proc(cfg);
  const notes: Preset['notes'] = [];
  const misplaced = ['loss_type', 'do_differential_guidance', 'differential_guidance_scale'].filter(key => key in p && !(key in (p.train ?? {})));
  if (misplaced.length) {
    p.train = { ...(p.train ?? {}) };
    for (const key of misplaced) {
      p.train[key] = p[key];
      delete p[key];
    }
    notes.push({ title: 'Moved into train', body: `${misplaced.map(m => '`' + m + '`').join(', ')} sat at the process level, where the trainer ignores them.` });
  }
  const t = p.train ?? {};
  if ((t.use_ema || t.ma_decay !== undefined || t.ema_decay !== undefined) && !t.ema_config?.use_ema) {
    t.ema_config = { use_ema: Boolean(t.use_ema), ema_decay: Number(t.ema_decay ?? t.ma_decay ?? 0.99) };
    delete t.use_ema;
    delete t.ema_decay;
    delete t.ma_decay;
    notes.push({ title: 'EMA fixed', body: 'Loose `use_ema` / `ema_decay` were rewritten as `train.ema_config`, the only place the trainer reads EMA from.' });
  }
  return { cfg, notes };
}

const describe = (cfg: JobConfig) => {
  const p = proc(cfg);
  return `${p.model?.arch ?? 'unknown arch'} · ${p.network?.type ?? 'lora'} · ${p.model?.name_or_path ?? ''}`;
};

const knobDefaults = (cfg: JobConfig): Preset['defaults'] => {
  const { name: _n, trigger: _t, ...k } = readKnobs(cfg);
  return k;
};

export function presetFromText(text: string, filename: string): Preset {
  const { cfg, notes } = parseConfig(text, filename.endsWith('.json'));
  return {
    id: 'imported-' + Math.random().toString(36).slice(2, 8),
    name: (filename.replace(/\.(ya?ml|json)$/i, '') || cfg.config.name || 'Imported') + ' (imported)',
    description: describe(cfg),
    builtIn: false,
    base: cfg,
    defaults: knobDefaults(cfg),
    notes,
  };
}

/** An editable copy; the built-in one stays as it ships. */
export function duplicatePreset(p: Preset): Preset {
  return { ...clone(p), id: 'custom-' + Math.random().toString(36).slice(2, 8), name: `${p.name} (copy)`, builtIn: false };
}

/**
 * The preset with its config replaced by edited YAML. The run knobs are read
 * again from the new config; the notes are the fixes this save made.
 */
export function editPreset(p: Preset, name: string, text: string): Preset {
  const { cfg, notes } = parseConfig(text);
  return {
    ...p,
    name: name.trim() || p.name,
    // a written description (the built-in's, on a copy) stays; a generated one follows the config
    description: p.description === describe(p.base) ? describe(cfg) : p.description,
    base: cfg,
    defaults: { ...knobDefaults(cfg), subject: p.defaults.subject },
    notes,
  };
}

/** The preset's own config, as edited (no run knobs applied). */
export function presetConfigYAML(p: Preset): string {
  return YAML.stringify(p.base, { lineWidth: 0 });
}

export function presetToYAML(preset: Preset, k?: Knobs): string {
  const cfg = k ? buildConfig(preset, k, { datasetPath: proc(preset.base)?.datasets?.[0]?.folder_path ?? '' }) : preset.base;
  return '---\n' + YAML.stringify(cfg, { lineWidth: 0 });
}

export function knobsFor(preset: Preset, name = '', trigger = ''): Knobs {
  return { ...clone(preset.defaults), name, trigger };
}

/** Switching subject swaps the prompt set when the preset ships one. */
export function withSubject(preset: Preset, k: Knobs, subject: Subject): Knobs {
  const set = preset.promptSets?.[subject];
  return { ...k, subject, prompts: set ? [...set] : k.prompts };
}

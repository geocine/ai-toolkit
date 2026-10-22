// Shapes of the /v1 API (docs/headless.md; same as the code-mode declaration).

export type JobState =
  | 'draft'
  | 'queued'
  | 'starting'
  | 'running'
  | 'stopping'
  | 'stopped'
  | 'completed'
  | 'failed'
  | 'interrupted';

export type JobPolicy = {
  auto_resume?: boolean | null;
  max_auto_resumes?: number | null;
  auto_upload_remote?: string | null;
  sync_remote?: string | null;
};

export type Attempt = {
  n: number;
  started_at: string;
  ended_at: string | null;
  end_state: string | null;
  end_reason: string | null;
  start_step: number;
  end_step: number | null;
  auto: boolean;
  pid: number | null;
};

export type Checkpoint = { name: string; path: string; step: number | null; size: number; modified: string; sha256?: string };
export type Sample = { name: string; path: string; step: number | null; index: number | null; modified: string };

export type Job = {
  id: string;
  name: string;
  type: string;
  ref: string | null;
  gpu_ids: string;
  state: JobState;
  status: string;
  info: string;
  step: number;
  total_steps: number | null;
  speed: string;
  queue_position: number;
  pid: number | null;
  created_at: string;
  updated_at: string;
  config: any;
  policy: JobPolicy;
  attempts: number;
  last_attempt: Attempt | null;
  latest_checkpoint: Checkpoint | null;
};

export type Items<T> = { items: T[]; next_cursor?: string | null };
export type LogChunk = { offset: number; next_offset: number; size: number; text: string };
export type LossSeries = { keys: string[]; key: string; points: { step: number; value: number; wall_time: number }[] };
export type Queue = { gpu_ids: string; running: boolean; current_job_id: string | null; queued: number };
export type Dataset = { name: string; path: string; images: number; captions: number; size: number; modified: string };
export type DatasetItem = { file: string; caption: string | null; size: number; width?: number; height?: number };
export type DatasetDetail = Dataset & { items: DatasetItem[]; next_cursor: string | null; caption_job: Job | null };
export type CaptionOption = { value: unknown; label: string };
export type Captioner = {
  name: string;
  label: string;
  group: string;
  defaults: Record<string, unknown>;
  model_options: CaptionOption[];
  model2_options: CaptionOption[];
  caption_prompts: Record<string, string>;
  additional_sections: string[];
  supports_loras: boolean;
  no_compile: boolean;
  min_new_tokens: number | null;
  multi_line_prompts: boolean;
  python_accepted: boolean;
};
export type CaptionCatalog = { captioners: Captioner[]; default_config: unknown; options: Record<string, CaptionOption[]> };

export type RemoteKind = 's3' | 'hf' | 'fs';
export type Remote = { id: string; name: string; kind: RemoteKind; config: Record<string, any>; created_at: string };
export type Upload = {
  id: string;
  remote_id: string;
  job_id: string | null;
  root: string;
  path: string;
  url: string | null;
  state: 'queued' | 'running' | 'done' | 'failed';
  bytes: number;
  bytes_done: number;
  sha256: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
};
export type Lease = { node_id: string; node_label: string; heartbeat_at: string; expires_at: string; expired?: boolean } | null;
export type SyncStatus = { remote: string | null; last_sync_at: string | null; last_error: string | null; lease: Lease; step?: number | null };
export type PortableJob = { name: string; step: number; state: string; lease: Lease; updated_at: string; datasets: string[] };
export type Settings = {
  training_folder: string;
  datasets_folder: string;
  data_root: string;
  models_path: string;
  hf_token_set: boolean;
  auto_resume_default: boolean;
  max_auto_resumes_default: number;
};
export type ApiKey = { id: string; name: string; prefix: string; created_at: string; last_used_at: string | null };
export type Health = { ok: boolean; version: string; upstream_version: string; started_at: string };

export type Gpu = {
  index: number;
  name?: string;
  driverVersion?: string;
  temperature: number;
  utilization: { gpu: number; memory: number };
  memory: { total: number; free: number; used: number };
  power: { draw: number; limit: number };
};
export type SystemInfo = {
  gpus: Gpu[];
  has_nvidia_smi: boolean;
  cpu: { name: string; cores: number; temperature: number; totalMemory: number; availableMemory: number; freeMemory: number; currentLoad: number };
  disks: { name: string; path: string; free: number; total: number }[];
  toolkit_root: string;
};

// run bundles (docs/headless.md, Run bundles)
export type TaskState = 'queued' | 'running' | 'done' | 'failed';
export type BundleTask = {
  id: string;
  kind: 'import' | 'push';
  state: TaskState;
  phase: '' | 'downloading' | 'reading' | 'restoring' | 'building' | 'uploading';
  bytes: number;
  bytes_done: number;
  job_id: string | null;
  job_name: string | null;
  dataset: string | null;
  source: string;
  ref: string | null;
  remote: string | null;
  key: string | null;
  url: string | null;
  upload_id: string | null;
  skipped: string[] | null;
  warnings: string[] | null;
  error: { code: string; message: string } | null;
  created_at: string;
  updated_at: string;
};
export type BundleUpload = { id: string; filename: string; size: number; received: number; complete: boolean; chunk_size: number; created_at: string; updated_at: string; expires_at: string };
export type BundleSummary = {
  format: number;
  name: string;
  step: number;
  total_steps: number | null;
  size: number;
  sha256?: string;
  created_at: string;
  images: number;
  datasets: string[] | null;
  base_model?: string;
  arch?: string;
  checkpoints: string[] | null;
  optimizer: boolean;
  server_version: string;
};
export type LibraryBundle = { id: string; key: string; size: number; modified: string; summary: BundleSummary | null };

// unattended runs (docs/headless.md, Unattended runs)
export type LifeMode = 'keep' | 'stop' | 'destroy';
export type Lifecycle = {
  mode: LifeMode;
  on: 'completed' | 'any';
  delay_s: number;
  dry_run: boolean;
  state: 'off' | 'waiting' | 'countdown' | 'verifying' | 'acting' | 'held' | 'cancelled' | 'done' | 'failed';
  deadline: string | null;
  reason: string;
  provider: 'runpod' | 'vast' | 'command' | 'exit';
  job_id: string | null;
  job_name: string | null;
  results: { job_id: string; job_name: string; safe: boolean; via: string; detail: string }[] | null;
  updated_at: string;
};
export type NotifySink = { id: string; kind: 'ntfy' | 'discord' | 'telegram' | 'pushover' | 'webhook'; target: string; source: 'env' | 'api'; created_at: string };
export type NotifyConfig = { sinks: NotifySink[]; events: string[] | null; kinds: string[]; public_url: string; studio_url: string; sample_every_s: number };
export type NotifyTest = { sink: string; kind: string; target: string; ok: boolean; detail: string };

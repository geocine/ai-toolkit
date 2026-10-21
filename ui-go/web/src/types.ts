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
export type DatasetDetail = Dataset & { items: DatasetItem[]; next_cursor: string | null };

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

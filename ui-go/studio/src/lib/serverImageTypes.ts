// Image check and host fit shapes (internal/imagecheck, internal/hostfit; the
// /v1/images and /v1/hosts routes). Types only: the calls live in
// serverImage.ts. Keep in sync with internal/codemode/image_types.go.
export type CheckLevel = 'block' | 'warn' | 'info';
export type CheckAction = { kind: 'filter_hosts' | 'show_all' | 'set_disk' | 'use_image' | 'pick_gpu' | 'edit_template'; value?: string | number };
export type ImageCheckItem = { level: CheckLevel; code: string; message: string; fix?: string; action?: CheckAction };
export type ImageRequirements = { platform: string; cuda_min: string; driver_min: number; compat_branches: { brand: string; branch: number }[]; sm: number[]; ptx: number[]; flash_attn_sm: number[]; torch: string; flavor: string; port: number; api: number; sources: Record<string, string>; source?: 'fallback' };
export type ImageFacts = { ref: string; repo: string; tag?: string; digest: string; platform_digest: string; platforms: string[]; platform: string; download_bytes: number; largest_layer_bytes: number; layers: number; unpacked_bytes: number; unpacked_source: 'label' | 'estimate'; created: string; labels: Record<string, string>; env: Record<string, string>; exposed_ports: string[]; go_server: boolean; revision?: string; requires: ImageRequirements; checked_at: string; rate_limit?: { limit: number; remaining: number; window_s: number } };
export type HostFacts = { provider?: string; gpu_name?: string; gpu_count?: number; vram_gb?: number; sm?: number; sm_source?: 'provider' | 'table' | ''; gpu_class?: 'geforce' | 'pro' | 'datacenter' | ''; gpu_vendor?: 'nvidia' | 'amd' | ''; cpu_arch?: string; cuda_max?: string; cuda_source?: string; driver?: string; disk_gb?: number; inet_down_mbps?: number; disk_bw_mbps?: number; ram_gb?: number; placement_floor?: boolean };
export type ModelRef = { arch: string; name_or_path?: string; quantize?: boolean; qtype?: string; low_vram?: boolean; resolution?: number; frames?: number };
export type DiskNeed = { disk_gb: number; image_peak_gb: number; model_gb: number; work_gb: number; model_source: 'hf' | 'table' | 'none' | 'default' };
export type PullEstimate = { image_min: number; model_min: number | null; basis: string; estimate: true };
export type VramFit = { verdict: 'fits' | 'tight' | 'no' | 'unknown'; note: string; source: 'repo' | 'vendor' | 'estimate' | ''; recipe?: string };
export type ImageCheck = { ok: boolean; level: 'ok' | CheckLevel; image: ImageFacts | null; launch_ref: string | null; pinned: string | null; host: HostFacts | null; need: DiskNeed | null; pull: PullEstimate | null; vram: VramFit | null; checks: ImageCheckItem[] };
export type ImageTag = { tag: string; digest: string; created: string; flavor?: string; cuda_min?: string; revision?: string; moving: boolean };
export type ImageChoice = { channel: 'same' | 'latest' | 'flavor' | 'custom' | 'template'; label: string; image: string; launch_ref: string | null; digest: string; revision?: string; flavor?: string; level: 'ok' | CheckLevel; why: string[] };
export type ImageRecommendation = { choice: ImageChoice | null; alternatives: ImageChoice[]; checks: ImageCheckItem[]; action?: CheckAction };
export type GpuInfo = { name: string; sm: number; vram_gb: number; class: 'geforce' | 'pro' | 'datacenter'; vendor: 'nvidia' | 'amd'; platform: string; aliases: string[]; speed: number; speed_source: string; verdict: 'ok' | CheckLevel; note: string };
export type HostFit = { model: ModelRef; tiers: ({ vram_gb: number } & VramFit)[]; need: DiskNeed };
export type CudaCheck = { ok: boolean; checked_at: string; torch: string; cuda: string; arch_list: string[]; devices: { index: number; name: string; capability: string; driver: string }[]; error: string | null; failure: 'no_kernel_image' | 'driver_too_old' | 'no_gpu' | 'torch_missing' | 'timeout' | null };
export type OfferFit = { level: 'ok' | CheckLevel; codes: string[]; summary: string; pull_min: number | null; vram: VramFit['verdict'] };

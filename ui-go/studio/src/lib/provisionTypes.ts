// GPU servers created on RunPod, Vast.ai and QuickPod (internal/provision).
// These types mirror the Go JSON exactly; internal/codemode carries the same
// text for the MCP declaration.
import type { HostFacts, OfferFit, ModelRef, DiskNeed } from './serverImageTypes';

export type ProviderKind = 'runpod' | 'vast' | 'quickpod';
export type MachineStatus = 'creating' | 'starting' | 'running' | 'ready' | 'stopping' | 'stopped' | 'terminating' | 'terminated' | 'failed' | 'unknown';
export type MachineProblem = 'token_mismatch' | 'open_server' | 'no_port' | 'not_ready_timeout' | 'unreachable' | 'never_listed' | 'provider_error' | 'not_refreshed' | 'cuda_failed' | 'pull_slow' | 'image_pull_failed';
export type WhenDone = 'keep' | 'stop' | 'destroy';
export interface ProviderCaps { template_required: boolean; cloud_choice: boolean; count_choice: boolean; stop_start: boolean; stop_keeps_disk: boolean; https: 'yes' | 'no' | 'unconfirmed'; self_destroy: 'builtin' | 'key'; credit: boolean; env_count_max: number; env_value_max: number; env_total_max: number; env_flag_string: boolean; image_digest_ref: 'yes' | 'no' | 'unconfirmed'; reports_inet: boolean; reports_disk_bw: boolean }
export interface Provider { kind: ProviderKind; label: string; stage: 'stable' | 'beta'; capabilities: ProviderCaps; unconfirmed: string[]; console_url: string; keys_url: string; key_scopes: string[]; configured: boolean; key_source: 'api' | 'env' | 'sim' | ''; key_hint: string | null; max_running: number; running: number; simulated: boolean }
export interface ProviderUpdate { api_key?: string; max_running?: number }
export interface ProviderTest { ok: boolean; detail: string; credit: number | null }
export interface OfferQuery { gpu?: string; min_vram_gb?: number; max_price?: number; gpu_count?: number; cloud?: 'secure' | 'community'; disk_gb?: number; limit?: number; image?: string; model?: string; show_all?: boolean; fast?: boolean; max_vram_gb?: number }
export interface Offer { id: string; provider: ProviderKind; gpu: string; gpu_count: number; vram_gb: number; price_per_hour: number; stopped_price_per_hour: number | null; location: string; reliability: number | null; availability: 'high' | 'medium' | 'low' | 'available'; max_disk_gb: number | null; cloud?: string; https: boolean; host: HostFacts | null; fit: OfferFit | null }
export interface ProviderTemplate { id: string; name: string; image: string; ours: boolean; publishes_port: boolean; has_token: boolean }
export interface LaunchRequest { provider: ProviderKind; offer_id: string; max_price_per_hour: number; request_id?: string; name?: string; image?: string; disk_gb?: number; gpu_count?: number; cloud?: 'secure' | 'community'; template_id?: string; cors_origins?: string[]; studio_url?: string; copy_remote?: string; copy_hf_token?: boolean; copy_notify?: boolean; when_done?: WhenDone; self_destroy_key?: boolean; env?: Record<string, string>; skip_image_check?: boolean }
export interface Machine { id: string; name: string; label: string; provider: ProviderKind; provider_id: string; status: MachineStatus; problem: MachineProblem | null; detail: string; provider_status: string; gpu: string; gpu_count: number; vram_gb: number; location: string; image: string; image_digest: string | null; image_launch_ref: string | null; image_warnings: string[]; disk_gb: number; base_url: string | null; secure: boolean; relay_path: string; price_per_hour: number | null; stopped_price_per_hour: number | null; when_done: WhenDone; self_destroy_key: boolean; copied_remote: string; actions: ('stop' | 'start' | 'terminate' | 'forget')[]; self: boolean; simulated: boolean; warnings: string[]; created_at: string; updated_at: string; ready_at: string | null; ended_at: string | null }
export interface MachinePatch { name?: string; token?: string }
export interface MachineLink { name: string; base_url: string; token: string; secure: boolean; relay_path: string }
export interface MachineLogs { text: string; source: string; truncated: boolean }
export interface GuideEnv { name: string; value: string; secret: boolean; note: string }
export interface GuideField { label: string; value: string; note: string }
export interface SetupGuide { kind: ProviderKind; image: string; port: number; disk_gb: number; env: GuideEnv[]; env_block: string; docker_options: string; fields: GuideField[]; steps: string[]; url_pattern: string; console_url: string; keys_url: string; key_scopes: string[]; token?: string }

// GPU recommender (internal/gpupick): best value and best speed in the
// smallest VRAM tier the model fits. It only preselects; it never launches.
export interface GpuPick { kind: 'value' | 'speed'; offer: Offer; tier_gb: number; speed: number; speed_source: string; per_dollar: number; reason: string }
export interface GpuRecommendation {
  providers: ProviderKind[]; model: ModelRef | null; start_tier_gb: number; tier_gb: number; default: 'value' | '';
  value: GpuPick | null; speed: GpuPick | null; need: DiskNeed;
  tiers: { tier_gb: number; offers: number; qualified: number; rejected: Record<string, number> }[];
  // provider is '' for image and model notes
  notes: { provider: ProviderKind | ''; code: string; message: string; offers: number }[];
  thresholds: { min_inet_down_mbps: number; min_disk_bw_mbps: number; min_reliability: number; ladder: number[] };
  message: string;
}
export interface RecommendQuery { model?: string; image?: string; disk_gb?: number; cloud?: 'secure' | 'community'; max_price?: number }

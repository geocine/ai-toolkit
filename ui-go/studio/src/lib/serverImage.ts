// The server image a GPU server runs: check it against a host, pick it for
// an offer, list its tags, and the GPU and VRAM hints (/v1/images,
// /v1/hosts; docs/headless.md, "Images and hosts"). The shapes live in
// serverImageTypes.ts. lib/images.ts is about dataset images, not this.
import { call, get, post, type Items, type Server } from './api';
import { query } from './provision';
import type { ProviderKind } from './provisionTypes';
import type { CheckAction, GpuInfo, HostFacts, HostFit, ImageCheck, ImageCheckItem, ImageRecommendation, ImageTag, ModelRef } from './serverImageTypes';

/** The image GPU servers run unless another is picked (imagecheck.DefaultRepo). */
export const DEFAULT_REPO = 'ghcr.io/geocine/aitoolkit-go';
export const DEFAULT_IMAGE = `${DEFAULT_REPO}:latest`;

export type OfferRef = { provider: ProviderKind; offer_id: string };

export interface ImageCheckRequest {
  image: string;
  platform?: string;
  host?: HostFacts;
  /** An offer from providers.offers; it wins over host. */
  offer?: OfferRef;
  model?: ModelRef;
  /** The disk the launch will ask for. */
  disk_gb?: number;
  /** Re-resolve the tag now instead of using the cache. */
  refresh?: boolean;
}

export interface ImageRecommendRequest {
  offer?: OfferRef;
  host?: HostFacts;
  model?: ModelRef;
  prefer?: 'same' | 'latest';
  /** A custom image to check as the only candidate. */
  image?: string;
  /** QuickPod: the template whose image runs. */
  template_id?: string;
  disk_gb?: number;
}

export interface HostFitRequest {
  model: ModelRef;
  image?: string;
  gpus?: { name?: string; vram_gb: number; count?: number }[];
  host?: HostFacts;
}

/** Never a 5xx for registry trouble: that comes back as a warn check. */
export const checkImage = (s: Server, req: ImageCheckRequest, signal?: AbortSignal) => call<ImageCheck>(s, 'POST', '/v1/images/check', req, signal);
/** choice is null with action filter_hosts when nothing fits. */
export const recommendImage = (s: Server, req: ImageRecommendRequest = {}, signal?: AbortSignal) =>
  call<ImageRecommendation>(s, 'POST', '/v1/images/recommend', req, signal);
/** Newest first; limit 1..20 (default 8). */
export const imageTags = async (s: Server, image = DEFAULT_REPO, limit?: number) =>
  (await get<Items<ImageTag>>(s, `/v1/images/tags${query({ image, limit })}`)).items;
export const hostFit = (s: Server, req: HostFitRequest) => post<HostFit>(s, '/v1/hosts/fit', req);
export const hostGpus = async (s: Server) => (await get<Items<GpuInfo>>(s, '/v1/hosts/gpus')).items;

/**
 * The model archs the host checks know (internal/hostfit vram.go archNames),
 * for "What will you train?". The value travels as offers' model= and as
 * ModelRef.arch.
 */
export const modelArchs: { arch: string; name: string }[] = [
  { arch: 'flux', name: 'FLUX.1' },
  { arch: 'flux_kontext', name: 'FLUX.1 Kontext' },
  { arch: 'flux2', name: 'FLUX.2' },
  { arch: 'krea2', name: 'Krea 2' },
  { arch: 'zimage', name: 'Z-Image' },
  { arch: 'qwen_image', name: 'Qwen-Image' },
  { arch: 'qwen_image_edit', name: 'Qwen-Image-Edit' },
  { arch: 'qwen_image_edit_plus', name: 'Qwen-Image-Edit-2509' },
  { arch: 'hidream', name: 'HiDream' },
  { arch: 'wan21', name: 'Wan 2.1 14B' },
  { arch: 'wan22_5b', name: 'Wan 2.2 5B' },
  { arch: 'wan22_14b', name: 'Wan 2.2 14B' },
  { arch: 'ltx2', name: 'LTX-2' },
];

export const archName = (arch: string) => modelArchs.find(m => m.arch === arch)?.name ?? arch;

/** The repository of an image reference, without @digest or :tag (a registry port stays). */
export function imageRepo(ref: string): string {
  const r = ref.trim().split('@')[0];
  const colon = r.lastIndexOf(':');
  return colon > r.lastIndexOf('/') ? r.slice(0, colon) : r;
}

/** The tag of an image reference ("" for none or a digest). */
export function imageTag(ref: string): string {
  const r = ref.trim().split('@')[0];
  const colon = r.lastIndexOf(':');
  return colon > r.lastIndexOf('/') ? r.slice(colon + 1) : '';
}

/** "sha256:d89f…59e30b". */
export function shortDigest(d: string): string {
  const [alg, hex] = d.includes(':') ? d.split(':', 2) : ['sha256', d];
  return hex.length > 14 ? `${alg}:${hex.slice(0, 4)}…${hex.slice(-6)}` : d;
}

/** Bytes in decimal gigabytes, as registries and the server's copy count them ("6.6 GB"). */
export const gb = (n: number) => `${(n / 1e9).toFixed(1)} GB`;

/** "3 min", "1 h 5 min", "under a minute". */
export function minutes(n: number): string {
  if (!(n >= 1)) return 'under a minute';
  const m = Math.round(n);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h${m % 60 ? ` ${m % 60} min` : ''}`;
}

/** The label of the button a check action becomes. */
export function actionLabel(a: CheckAction): string {
  switch (a.kind) {
    case 'filter_hosts':
      return a.value === 'fast' ? 'Show only fast hosts' : 'Show only hosts that fit';
    case 'show_all':
      return 'Show all hosts';
    case 'set_disk':
      return `Use ${a.value} GB disk`;
    case 'use_image': {
      const tag = typeof a.value === 'string' ? imageTag(a.value) : '';
      return tag && tag !== 'latest' ? `Use build ${tag}` : 'Use this image';
    }
    case 'pick_gpu':
      return 'Pick another GPU';
    case 'edit_template':
      return 'Edit the template';
  }
}

/** Only registry trouble: the image could not be checked, which never blocks. */
export const uncheckedOnly = (checks: ImageCheckItem[]) =>
  checks.length > 0 && checks.every(c => c.level === 'info' || c.code === 'image_unreachable' || c.code === 'image_unchecked') && checks.some(c => c.level === 'warn');

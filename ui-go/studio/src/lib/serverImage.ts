// The server image a GPU server runs: check it against a host, pick it for
// an offer, list its tags, and the GPU and VRAM hints (/v1/images,
// /v1/hosts; docs/headless.md, "Images and hosts"). The shapes live in
// serverImageTypes.ts. lib/images.ts is about dataset images, not this.
import { call, get, post, type Items, type Server } from './api';
import { query } from './provision';
import type { ProviderKind } from './provisionTypes';
import type { CheckAction, GpuInfo, HostFacts, HostFit, ImageCheck, ImageCheckItem, ImageRecommendation, ImageTag, ModelInfo, ModelRef } from './serverImageTypes';

/** The image GPU servers run unless another is picked (imagecheck.DefaultRepo). */
export const DEFAULT_REPO = 'ghcr.io/geocine/aitoolkit-worker';
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
 * Upstream's model archs as the server has them built in (GET /v1/hosts/models),
 * for "What will you train?". The arch travels as offers' model= and as
 * ModelRef.arch; vram says the server has VRAM data for it.
 */
export const hostModels = async (s: Server) => (await get<Items<ModelInfo>>(s, '/v1/hosts/models')).items;

/** A model's label from the list, else the arch itself. */
export const archName = (models: readonly ModelInfo[] | null | undefined, arch: string) => models?.find(m => m.arch === arch)?.name ?? arch;

const groupOrder = ['image', 'instruction', 'video', 'audio', 'llm', 'experimental'];
const groupLabels: Record<string, string> = { image: 'Image', instruction: 'Image editing', video: 'Video', audio: 'Audio', llm: 'Language', experimental: 'Experimental' };

/** The list in upstream's groups (image first), each in the server's order. */
export function modelGroups(models: readonly ModelInfo[]): { group: string; label: string; models: ModelInfo[] }[] {
  const by = new Map<string, ModelInfo[]>();
  for (const m of models) by.set(m.group, [...(by.get(m.group) ?? []), m]);
  const rank = (g: string) => (groupOrder.includes(g) ? groupOrder.indexOf(g) : groupOrder.length);
  return [...by.keys()]
    .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
    .map(g => ({ group: g, label: groupLabels[g] ?? (g ? g[0].toUpperCase() + g.slice(1) : 'Other'), models: by.get(g)! }));
}

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

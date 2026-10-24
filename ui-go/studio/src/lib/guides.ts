// Manual template guides, bundled so they work with no server at all.
// guides.json is generated from internal/provision/guide
// (go test ./internal/provision/guide -update); GET /v1/providers/{kind}/setup
// serves the same text from a controller.
import bundled from './guides.json';
import type { ProviderKind, SetupGuide } from './provisionTypes';

// what the guides leave in place of a token and an origin (guide.TokenPlaceholder, guide.OriginPlaceholder)
export const TOKEN_PLACEHOLDER = '<a long random token>';
export const ORIGIN_PLACEHOLDER = '<your studio origin>';

const guides = bundled as Record<ProviderKind, SetupGuide>;

export interface GuideOptions {
  token?: string;
  origin?: string;
  image?: string;
  disk?: number;
}

// The same rules as guide.normalize in Go, so the bundled guide and
// GET /v1/providers/{kind}/setup accept the same values: nothing that would
// break the copied env block or docker options.
const MIN_DISK = 20;
const MAX_DISK = 2000;
const TOKEN_RE = /^[A-Za-z0-9._~-]{8,512}$/;
const ORIGIN_RE = /^https?:\/\/[A-Za-z0-9.-]+(:[0-9]{1,5})?$|^https?:\/\/\[[0-9A-Fa-f:.]+\](:[0-9]{1,5})?$/;
const IMAGE_BAD = /[\s"'`$\\;|&<>]/;

const validToken = (t: string) => TOKEN_RE.test(t);
const validImage = (s: string) => s.length <= 1024 && !IMAGE_BAD.test(s);
const validDisk = (n: number) => Number.isInteger(n) && n >= MIN_DISK && n <= MAX_DISK;
const originParts = (s: string) =>
  s
    .split(',')
    .map(p => p.trim())
    .filter(Boolean);
const validOrigin = (p: string) => p === '*' || ORIGIN_RE.test(p);

/** What is wrong with each option, by name; empty when the guide can use them all. Blank options are fine (they keep the placeholders and defaults). */
export function validateGuideOptions(o: GuideOptions): Partial<Record<keyof GuideOptions, string>> {
  const errs: Partial<Record<keyof GuideOptions, string>> = {};
  const token = o.token?.trim() ?? '';
  if (token && !validToken(token)) errs.token = 'Use 8 or more letters, digits or ._~- (no spaces).';
  if (originParts(o.origin ?? '').some(p => !validOrigin(p))) errs.origin = 'Use * or scheme://host[:port], comma separated, with no path.';
  const image = o.image?.trim() ?? '';
  if (image && !validImage(image)) errs.image = 'Use an image reference without spaces or quotes.';
  if (o.disk != null && o.disk !== 0 && !validDisk(o.disk)) errs.disk = `Use ${MIN_DISK} to ${MAX_DISK} GB.`;
  return errs;
}

/**
 * g with the token, origin, image and disk filled in. Values the server would
 * refuse are left out: a bad token or image keeps the placeholder or default,
 * each origin is trimmed and the invalid ones dropped, and a disk outside
 * 20-2000 GB keeps the guide's. Works on the bundled guides and on the ones a
 * controller serves.
 */
export function fillGuide(g: SetupGuide, o: GuideOptions = {}): SetupGuide {
  const token = o.token?.trim() ?? '';
  const tok = validToken(token) ? token : TOKEN_PLACEHOLDER;
  const origin = originParts(o.origin ?? '').filter(validOrigin).join(',') || ORIGIN_PLACEHOLDER;
  const raw = o.image?.trim() ?? '';
  const image = raw && validImage(raw) ? raw : g.image;
  const disk = o.disk != null && validDisk(o.disk) ? o.disk : g.disk_gb;
  const fill = (s: string) => s.split(TOKEN_PLACEHOLDER).join(tok).split(ORIGIN_PLACEHOLDER).join(origin).split(g.image).join(image);
  return {
    ...g,
    image,
    disk_gb: disk,
    env: g.env.map(e => ({ ...e, value: fill(e.value), note: fill(e.note) })),
    env_block: fill(g.env_block),
    docker_options: fill(g.docker_options),
    // the disk fields ("Container Disk", "Disk Space") hold the bare GB number
    fields: g.fields.map(f => ({ ...f, value: /disk/i.test(f.label) ? String(disk) : fill(f.value), note: fill(f.note) })),
    steps: g.steps.map(fill),
    key_scopes: [...g.key_scopes],
  };
}

/** The bundled guide for kind with the options filled in (fillGuide); empty options keep the placeholders. */
export function guideFor(kind: ProviderKind, o: GuideOptions = {}): SetupGuide {
  return fillGuide(guides[kind], o);
}

/** A fresh server token: "aitk_" + base64url of 32 random bytes (provision.NewToken). */
export function newToken(): string {
  const b = crypto.getRandomValues(new Uint8Array(32));
  let s = '';
  for (const x of b) s += String.fromCharCode(x);
  return 'aitk_' + btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

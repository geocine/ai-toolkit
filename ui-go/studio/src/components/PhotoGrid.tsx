import { Ellipsis } from 'lucide-react';
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import type { PhotoFlag, PhotoGroup, PhotoItem, PhotoPick, PhotoReview } from '../lib/photoTypes';

// The Images step's photo grid (PHOTO-PLAN 13.1, 13.2): every photo with its
// flags from the server's judge, the summary badges and bulk actions above
// it, and the similar pairs to decide below it. Silence is the good state:
// a photo with nothing wrong shows no tag at all.

/** A photo in the grid: picked in this browser (blob) or stored on the server (src). */
export type GridPhoto = {
  key: string;
  file: string;
  /** The thumbnail URL of a stored photo; empty for a local one (blob). */
  src: string;
  /** A local photo's file: its object URL is made only while the card is near the screen. */
  blob?: Blob;
  /** The full-size URL of a stored photo, for the compare dialog. */
  full?: string;
  item?: PhotoItem;
  leftOut: boolean;
  /** Its number in a pick preview. */
  rank?: number;
  /** Why a pick preview left it out. */
  reason?: string;
  stored: boolean;
};
export type PhotoAction = 'leave_out' | 'restore' | 'keep' | 'unkeep' | 'intended' | 'unintend' | 'reference' | 'compare';
export type PhotoFilter = PhotoFlag | 'flagged' | null;

// The labels of internal/photocheck/flags.go Catalog (they read inside
// "N {label}"), and which flags warn. The review carries counts and the
// per-photo sentences (notes), not the labels.
const flagInfo: Record<PhotoFlag, [label: string, warn: boolean]> = {
  unreadable: ['unreadable', true],
  unsupported: ['not read by the trainer', true],
  huge: ['too large', true],
  gray16: ['16-bit grey', true],
  transparent: ['transparent', true],
  animated: ['animated', false],
  cmyk: ['CMYK', false],
  same_stem: ['sharing a caption', true],
  rotated: ['stored sideways', false],
  generated: ['AI-generated', false],
  derived: ['face crop', false],
  duplicate: ['duplicates', true],
  similar: ['very similar', false],
  burst: ['same moment', false],
  tiny: ['too small', true],
  small: ['low resolution', true],
  upscaled: ['maybe enlarged', false],
  blurry: ['blurry', true],
  dark: ['too dark', true],
  bright: ['blown out', true],
  flat: ['flat', false],
  grayscale: ['black and white', false],
  strong_color: ['strong colour', false],
  noisy: ['grainy', false],
  compressed: ['compressed', false],
  screenshot: ['screenshots', true],
  from_screen: ['saved from a screen', false],
  bars: ['black bars', true],
  aspect: ['very wide or tall', false],
  mentions_covered: ['sunglasses or hat (caption)', false],
  mentions_people: ['other people (caption)', false],
  mentions_text: ['text (caption)', false],
  mentions_tone: ['black and white (caption)', false],
  mentions_screen: ['screen (caption)', false],
  no_face: ['no face', true],
  others: ['other people', true],
  small_face: ['small face', true],
  face_cut: ['face cut off', false],
  face_blurry: ['face blurry', true],
  backlit: ['face in shadow', false],
  not_them: ['someone else?', true],
  identity_check: ['check the face', false],
  smoothed: ['smoothed skin', false],
  text: ['text', false],
  watermark: ['watermark', true],
  eyes_hidden: ['eyes hidden', false],
  covered: ['face covered', false],
  filtered: ['filtered', false],
  collage: ['collage', false],
  back_view: ['back view', false],
  expression_extreme: ['strong expression', false],
  lookalike: ['look-alike', false],
};
const flagOrder = Object.keys(flagInfo) as PhotoFlag[];

export const flagLabel = (f: PhotoFlag) => flagInfo[f]?.[0] ?? f.replace(/_/g, ' ');
const flagWarns = (f: PhotoFlag) => flagInfo[f]?.[1] ?? false;

/** Flags "Keep anyway" never waives: the trainer cannot read the file at all (photocheck trainerUnreadable). */
const trainerUnreadable: PhotoFlag[] = ['unreadable', 'unsupported', 'huge'];
/** Flags "This is intended" waives (photocheck judge, section 4.7). */
const intendable: PhotoFlag[] = ['no_face', 'grayscale', 'mentions_covered', 'mentions_tone', 'eyes_hidden', 'covered', 'back_view'];
/** Flags "Leave out duplicates and too-small" takes (set_aside_flagged's defaults). */
export const suggestedFlags: PhotoFlag[] = ['duplicate', 'tiny', 'unreadable'];

/** "4 duplicates and 1 too small". */
export function countList(parts: [number, string][]): string {
  const s = parts.filter(([n]) => n > 0).map(([n, l]) => `${n} ${l}`);
  return s.length <= 1 ? (s[0] ?? '') : `${s.slice(0, -1).join(', ')} and ${s[s.length - 1]}`;
}

/** The photos "Leave out …" would take, and its button text. */
export function suggested(photos: GridPhoto[]): { files: string[]; flags: PhotoFlag[]; label: string } {
  const files: string[] = [];
  const by = new Map<PhotoFlag, number>();
  for (const p of photos) {
    if (p.leftOut || !p.item) continue;
    const f = suggestedFlags.find(fl => p.item!.flags.includes(fl) && !p.item!.waived?.includes(fl));
    if (!f) continue;
    files.push(p.file);
    by.set(f, (by.get(f) ?? 0) + 1);
  }
  const flags = suggestedFlags.filter(f => by.has(f));
  return { files, flags, label: countList(flags.map(f => [by.get(f)!, flagLabel(f)])) };
}

const matches = (p: GridPhoto, filter: PhotoFilter) =>
  !filter || (filter === 'flagged' ? (p.item?.flags.length ?? 0) > 0 : !!p.item?.flags.includes(filter));

/** A group's identity across reviews (ids are positional per review): its kind and its files. */
export const groupKey = (g: PhotoGroup) => g.kind + ':' + [...g.files].sort().join('\u0000');

/** The other photo of a duplicate, similar or burst group. */
export function partnerOf(item: PhotoItem | undefined, groups: PhotoGroup[]): string | undefined {
  if (!item) return undefined;
  if (item.same_as) return item.same_as;
  const g = groups.find(g => g.kind !== 'duplicate' && g.files.includes(item.file));
  return g?.files.find(f => f !== item.file);
}

/**
 * A photo's image. A local photo's object URL exists only while the card is
 * near the screen (a 300-photo drop would otherwise hold every decoded photo).
 */
export function PhotoThumb({ photo, full }: { photo: GridPhoto; full?: boolean }) {
  const box = useRef<HTMLSpanElement>(null);
  const [src, setSrc] = useState(photo.blob ? '' : full ? photo.full || photo.src : photo.src);
  useEffect(() => {
    if (!photo.blob) {
      setSrc(full ? photo.full || photo.src : photo.src);
      return;
    }
    const blob = photo.blob;
    let u = '';
    const show = (on: boolean) => {
      if (on && !u) {
        u = URL.createObjectURL(blob);
        setSrc(u);
      } else if (!on && u) {
        URL.revokeObjectURL(u);
        u = '';
        setSrc('');
      }
    };
    const el = box.current;
    let io: IntersectionObserver | null = null;
    if (full || !el || typeof IntersectionObserver === 'undefined') show(true);
    else {
      io = new IntersectionObserver(es => show(es.some(e => e.isIntersecting)), { rootMargin: '600px 0px' });
      io.observe(el);
    }
    return () => {
      io?.disconnect();
      if (u) URL.revokeObjectURL(u);
    };
  }, [photo.blob, photo.src, photo.full, full]);
  return (
    <span ref={box} className={full ? 'photo-img is-full' : 'photo-img'}>
      {src && <img src={src} alt="" loading="lazy" decoding="async" />}
    </span>
  );
}

/** The soft badge of a photo's flags; waived flags are struck through. Its title is the judge's notes. */
export function FlagBadge({ item }: { item: PhotoItem }) {
  const waived = item.waived ?? [];
  if (!item.flags.length && !waived.length) return null;
  const warn = item.flags.some(flagWarns);
  const title = [...item.flags, ...waived].map(f => item.notes[f] ?? flagLabel(f)).join(' ');
  return (
    <span className={`badge photo-flags ${warn ? 'tone-warn' : 'tone-quiet'}`} title={title}>
      {item.flags.map(flagLabel).join(', ')}
      {waived.map(f => (
        <s key={f}> {flagLabel(f)}</s>
      ))}
    </span>
  );
}

/**
 * A small "…" menu; Escape or a click outside closes it. It opens to the
 * left of its button, or to the right when that would leave the screen
 * (a card in the grid's first column on a phone).
 */
export function PhotoMenu({ label, children, disabled }: { label: string; children: ReactNode; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const [flip, setFlip] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const pop = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (!open) return setFlip(false);
    const r = pop.current?.getBoundingClientRect();
    if (r && r.left < 8) setFlip(true);
  }, [open]);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);
  return (
    <div className="menu" ref={box}>
      <button
        type="button"
        className="btn btn-sm btn-ghost btn-icon photo-menu-btn"
        aria-label={label}
        title="More"
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen(o => !o)}
      >
        <Ellipsis />
      </button>
      {open && (
        <div ref={pop} className={flip ? 'menu-pop photo-menu-pop is-flipped' : 'menu-pop photo-menu-pop'} role="menu" onClick={() => setOpen(false)}>
          {children}
        </div>
      )}
    </div>
  );
}

/**
 * The counts as soft badges (a click filters the grid), the bulk actions
 * (each names what it takes), "Pick the best N" and its preview bar.
 */
export function PhotoSummary(p: {
  review: PhotoReview | null;
  photos: GridPhoto[];
  filter: PhotoFilter;
  setFilter(f: PhotoFilter): void;
  onSuggested(): void;
  onPick(n: number): void;
  undo?: () => void;
  busy?: boolean;
  /** A pick preview to confirm or drop. */
  pick?: PhotoPick | null;
  onKeepPick?(): void;
  onCancelPick?(): void;
}) {
  const { review, photos, filter, setFilter, busy } = p;
  const [n, setN] = useState(25);
  const sug = suggested(photos);
  const counts = (Object.entries(review?.counts ?? {}) as [PhotoFlag, number][])
    .filter(([, c]) => c > 0)
    .sort(([a], [b]) => (flagWarns(b) ? 1 : 0) - (flagWarns(a) ? 1 : 0) || flagOrder.indexOf(a) - flagOrder.indexOf(b));
  const pickable = photos.filter(x => !x.leftOut && x.item).length;
  return (
    <div className="photo-summary">
      {counts.length > 0 && (
        <p className="photo-counts">
          {counts.map(([f, c]) => (
            <button
              key={f}
              type="button"
              className={`badge photo-count ${flagWarns(f) ? 'tone-warn' : 'tone-quiet'}${filter === f ? ' is-on' : ''}`}
              aria-pressed={filter === f}
              onClick={() => setFilter(filter === f ? null : f)}
            >
              {c} {flagLabel(f)}
            </button>
          ))}
          {filter && (
            <button type="button" className="link photo-count-all" onClick={() => setFilter(null)}>
              Show all
            </button>
          )}
        </p>
      )}
      {p.pick ? (
        <div className="photo-pickbar">
          <p>
            <strong className="num">{p.pick.kept.length}</strong> picked of {pickable}. Picked for sharpness first, then for variety: angles, framing, outfits,
            places.
          </p>
          {p.pick.note && <p className="quiet-note">{p.pick.note}</p>}
          <div className="photo-actions">
            <button type="button" className="btn btn-sm btn-primary" disabled={busy} onClick={p.onKeepPick}>
              Keep these {p.pick.kept.length}
            </button>
            <button type="button" className="btn btn-sm btn-ghost" disabled={busy} onClick={p.onCancelPick}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        (sug.files.length > 0 || pickable >= 10 || p.undo) && (
          <div className="photo-actions">
            {sug.files.length > 0 && (
              <button type="button" className="btn btn-sm" disabled={busy} onClick={p.onSuggested}>
                Leave out {sug.label}
              </button>
            )}
            {pickable >= 10 && (
              <span className="photo-pick">
                <button type="button" className="btn btn-sm" disabled={busy} onClick={() => p.onPick(Math.min(40, Math.max(10, n || 25)))}>
                  Pick the best
                </button>
                <input
                  type="number"
                  min={10}
                  max={40}
                  value={n}
                  aria-label="How many photos to pick"
                  onChange={e => setN(Math.round(Number(e.target.value)))}
                  onBlur={() => setN(v => Math.min(40, Math.max(10, v || 25)))}
                />
              </span>
            )}
            {p.undo && (
              <button type="button" className="link" disabled={busy} onClick={p.undo}>
                Undo
              </button>
            )}
          </div>
        )
      )}
    </div>
  );
}

/** Every photo as a card: a click leaves it out or puts it back; the "…" menu holds the rest. */
export function PhotoGrid(p: {
  photos: GridPhoto[];
  filter: PhotoFilter;
  onToggle(file: string): void;
  onAction(a: PhotoAction, file: string): void;
  /** Offer "This is them" (person sets, stored photos). */
  person?: boolean;
  busy?: boolean;
  /** The review's groups, for "Compare with" on similar and burst photos. */
  groups?: PhotoGroup[];
  /** A pick preview is showing: photos without a number dim. */
  picking?: boolean;
  /** Whether "Compare with" can show a partner (default: always). */
  canCompare?(file: string): boolean;
}) {
  const shown = p.photos.filter(x => matches(x, p.filter));
  if (!shown.length) return <p className="quiet-note photo-none">No photo matches this filter.</p>;
  return (
    <div className="photo-grid">
      {shown.map(x => (
        <PhotoCard
          key={x.key}
          photo={x}
          onToggle={p.onToggle}
          onAction={p.onAction}
          person={p.person}
          busy={p.busy}
          groups={p.groups}
          picking={p.picking}
          canCompare={p.canCompare}
        />
      ))}
    </div>
  );
}

function PhotoCard({
  photo,
  onToggle,
  onAction,
  person,
  busy,
  groups = [],
  picking,
  canCompare,
}: {
  photo: GridPhoto;
  onToggle(file: string): void;
  onAction(a: PhotoAction, file: string): void;
  person?: boolean;
  busy?: boolean;
  groups?: PhotoGroup[];
  picking?: boolean;
  canCompare?(file: string): boolean;
}) {
  const { item, file, leftOut, rank } = photo;
  const warn = item?.flags.some(flagWarns);
  const dim = leftOut || (picking && rank === undefined);
  const other = item && item.flags.some(f => f === 'duplicate' || f === 'similar' || f === 'burst') ? partnerOf(item, groups) : undefined;
  const partner = other && (!canCompare || canCompare(other)) ? other : undefined;
  // "Keep anyway" on a file the trainer cannot read would store a decision that changes nothing
  const hasWarn = !!item?.flags.some(f => flagWarns(f) && !trainerUnreadable.includes(f));
  const canIntend = !!item && (item.flags.some(f => intendable.includes(f)) || item.decision === 'intended');
  const title = photo.reason ?? (leftOut ? `${file} · left out: click to put it back` : `${file} · click to leave it out`);
  return (
    <figure className={`photo-card${dim ? ' is-dim' : ''}${leftOut ? ' is-out' : ''}`}>
      <button type="button" className="photo-hit" onClick={() => onToggle(file)} disabled={busy} title={title} aria-pressed={leftOut} aria-label={title}>
        <PhotoThumb photo={photo} />
        {item && item.flags.length > 0 && <span className={`photo-dot ${warn ? 'tone-warn' : 'tone-quiet'}`} aria-hidden />}
        {rank !== undefined && <span className="photo-rank num">{rank}</span>}
        {item?.reference && <span className="badge tone-accent photo-ref">Them</span>}
        {item?.pending && <span className="photo-pending" aria-hidden />}
      </button>
      <figcaption>
        <span className="photo-tags">
          {leftOut && <span className="badge tone-quiet">left out</span>}
          {item && !leftOut && <FlagBadge item={item} />}
        </span>
        <PhotoMenu label={`More for ${file}`} disabled={busy}>
          {leftOut ? (
            <button role="menuitem" onClick={() => onAction('restore', file)}>
              Keep
            </button>
          ) : (
            <button role="menuitem" onClick={() => onAction('leave_out', file)}>
              Leave out
            </button>
          )}
          {item?.decision === 'keep' ? (
            <button role="menuitem" onClick={() => onAction('unkeep', file)}>
              Undo keep anyway
            </button>
          ) : (
            hasWarn && (
              <button role="menuitem" onClick={() => onAction('keep', file)}>
                Keep anyway
              </button>
            )
          )}
          {item?.decision === 'intended' ? (
            <button role="menuitem" onClick={() => onAction('unintend', file)}>
              Not intended
            </button>
          ) : (
            canIntend && (
              <button role="menuitem" onClick={() => onAction('intended', file)}>
                This is intended
              </button>
            )
          )}
          {partner && (
            <button role="menuitem" onClick={() => onAction('compare', file)}>
              Compare with {partner}
            </button>
          )}
          {photo.stored && person && !item?.reference && (
            <button role="menuitem" onClick={() => onAction('reference', file)}>
              This is them
            </button>
          )}
          <span className="photo-menu-file mono" title={file}>
            {file}
          </span>
        </PhotoMenu>
      </figcaption>
    </figure>
  );
}

/**
 * "For you to decide": every similar or burst group as tiles. Each tile
 * keeps or leaves out its photo, and nothing is chosen beforehand: a pair is
 * a claim the user checks with their own eyes.
 */
export function DecidePairs(p: { review: PhotoReview; photos: GridPhoto[]; onToggle(file: string): void; onCompare(a: string, b: string): void; busy?: boolean }) {
  const byFile = new Map(p.photos.map(x => [x.file, x]));
  const groups = p.review.groups.filter(g => (g.kind === 'similar' || g.kind === 'burst') && g.files.filter(f => byFile.has(f)).length >= 2);
  if (!groups.length) return null;
  return (
    <section className="photo-decide">
      <header>
        <h3>For you to decide</h3>
        <p className="quiet-note">Very similar photos, or the same moment. Keep the best of each, or both if they differ.</p>
      </header>
      <div className="photo-decide-list">
        {groups.map(g => {
          const files = g.files.filter(f => byFile.has(f));
          return (
            <div key={groupKey(g)} className="photo-pair">
              <div className="photo-pair-tiles">
                {files.map(f => {
                  const x = byFile.get(f)!;
                  return (
                    <button
                      key={f}
                      type="button"
                      className={`photo-tile${x.leftOut ? ' is-out' : ''}`}
                      onClick={() => p.onToggle(f)}
                      disabled={p.busy}
                      aria-pressed={!x.leftOut}
                      title={x.leftOut ? `${f} · left out: click to keep it` : `${f} · click to leave it out`}
                    >
                      <PhotoThumb photo={x} />
                      <span className={`badge ${x.leftOut ? 'tone-quiet' : 'tone-ok'}`}>{x.leftOut ? 'Left out' : 'Keep'}</span>
                    </button>
                  );
                })}
              </div>
              <div className="photo-pair-foot">
                <span className="muted">{g.kind === 'burst' ? 'Same moment' : 'Very similar'}</span>
                <button type="button" className="btn btn-sm btn-ghost" onClick={() => p.onCompare(files[0], files[1])}>
                  Compare
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

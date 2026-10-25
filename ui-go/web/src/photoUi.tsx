// Photo checks on the dataset page (docs/headless.md, Datasets, "Photo
// checks"): GET .../photos/check, POST .../photos/fix, PUT .../photos/settings,
// DELETE .../photos/analysis and POST .../photos/faces. The server judges;
// this file only shows the review and sends the user's choices.

import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { ArrowRight, Check, Columns2, Ellipsis, EyeOff, RotateCcw, ScanFace, Square, Undo2, UserCheck } from 'lucide-react';
import { api, ApiError, imgUrl, joinServerPath, plural } from './api';
import { href } from './router';
import type { Job, PhotoDecision, PhotoFix, PhotoFixOp, PhotoFlag, PhotoItem, PhotoKind, PhotoReview, PhotoSettingsPatch } from './types';
import { Bar, Callout, Modal, Spinner, useConfirm, useToast } from './ui';

// ---- flags (labels and default levels as internal/photocheck/flags.go Catalog) ----

type Level = 'warn' | 'info';
const FLAGS: Record<PhotoFlag, [string, Level]> = {
  unreadable: ['unreadable', 'warn'],
  unsupported: ['not read by the trainer', 'warn'],
  huge: ['too large', 'warn'],
  gray16: ['16-bit grey', 'warn'],
  transparent: ['transparent', 'warn'],
  animated: ['animated', 'info'],
  cmyk: ['CMYK', 'info'],
  same_stem: ['sharing a caption', 'warn'],
  rotated: ['stored sideways', 'info'],
  generated: ['AI-generated', 'info'],
  derived: ['face crop', 'info'],
  duplicate: ['duplicates', 'warn'],
  similar: ['very similar', 'info'],
  burst: ['same moment', 'info'],
  tiny: ['too small', 'warn'],
  small: ['low resolution', 'warn'],
  upscaled: ['maybe enlarged', 'info'],
  blurry: ['blurry', 'warn'],
  dark: ['too dark', 'warn'],
  bright: ['blown out', 'warn'],
  flat: ['flat', 'info'],
  grayscale: ['black and white', 'info'],
  strong_color: ['strong colour', 'info'],
  noisy: ['grainy', 'info'],
  compressed: ['compressed', 'info'],
  screenshot: ['screenshots', 'warn'],
  from_screen: ['saved from a screen', 'info'],
  bars: ['black bars', 'warn'],
  aspect: ['very wide or tall', 'info'],
  mentions_covered: ['sunglasses or hat (caption)', 'info'],
  mentions_people: ['other people (caption)', 'info'],
  mentions_text: ['text (caption)', 'info'],
  mentions_tone: ['black and white (caption)', 'info'],
  mentions_screen: ['screen (caption)', 'info'],
  no_face: ['no face', 'warn'],
  others: ['other people', 'warn'],
  small_face: ['small face', 'warn'],
  face_cut: ['face cut off', 'info'],
  face_blurry: ['face blurry', 'warn'],
  backlit: ['face in shadow', 'info'],
  not_them: ['someone else?', 'warn'],
  identity_check: ['check the face', 'info'],
  smoothed: ['smoothed skin', 'info'],
  text: ['text', 'info'],
  watermark: ['watermark', 'warn'],
  eyes_hidden: ['eyes hidden', 'info'],
  covered: ['face covered', 'info'],
  filtered: ['filtered', 'info'],
  collage: ['collage', 'info'],
  back_view: ['back view', 'info'],
  expression_extreme: ['strong expression', 'info'],
  lookalike: ['look-alike', 'info'],
};
const ORDER = Object.keys(FLAGS) as PhotoFlag[];
// flags Keep anyway never waives: the trainer cannot read the photo at all
const TRAINER_UNREADABLE = new Set<PhotoFlag>(['unreadable', 'unsupported', 'huge']);
// whether a flag warns on this photo; the judge sets three levels at run time
// (aspect above its warn ratio, bars on object sets, no_face when few photos lack one)
const flagWarns = (f: PhotoFlag, it: PhotoItem, kind: PhotoKind) => {
  if (f === 'bars') return kind !== 'object';
  if (f === 'aspect' || f === 'no_face') return it.level === 'warn';
  return FLAGS[f]?.[1] === 'warn';
};
export const flagLabel = (f: PhotoFlag) => FLAGS[f]?.[0] ?? f.replace(/_/g, ' ');
// "1 duplicate", "4 duplicates": the plural labels read inside counts
const countLabel = (f: PhotoFlag, n: number) => (n === 1 && (f === 'duplicate' || f === 'screenshot') ? f : flagLabel(f));

// flags "This is intended" waives (section 4.7)
const INTENDED: PhotoFlag[] = ['no_face', 'back_view', 'covered', 'eyes_hidden', 'grayscale', 'mentions_covered', 'mentions_tone'];
const ACTIVE = new Set(['queued', 'starting', 'running', 'stopping']);
const jobActive = (j: Job | null | undefined) => !!j && ACTIVE.has(j.state);
const KINDS: { id: PhotoKind; label: string }[] = [
  { id: 'person', label: 'Person' },
  { id: 'style', label: 'Style' },
  { id: 'object', label: 'Object' },
];

const photoBase = (dataset: string) => `/v1/datasets/${encodeURIComponent(dataset)}/photos`;

// ---- the review ----

/**
 * The dataset's photo review, fetched again whenever version changes and
 * polled every 2 s while photos are still being analysed. review stays null
 * when the server has no photo checks (an older server).
 */
export function usePhotoReview(dataset: string, version: string) {
  const [review, setReview] = useState<PhotoReview | null>(null);
  const [tick, setTick] = useState(0);
  const last = useRef<PhotoReview | null>(null);
  last.current = review;
  useEffect(() => {
    const ctrl = new AbortController();
    let timer = 0;
    const run = async () => {
      try {
        const r = await api.get<PhotoReview>(`${photoBase(dataset)}/check?wait_ms=3000`, ctrl.signal);
        if (ctrl.signal.aborted) return;
        setReview(r);
        if (r.pending > 0) timer = window.setTimeout(run, 2000);
      } catch (e) {
        if (ctrl.signal.aborted) return;
        // a 404 is a server without photo checks; anything else retries while photos are still pending
        if (e instanceof ApiError && e.status === 404) setReview(null);
        else if (!last.current || last.current.pending > 0) timer = window.setTimeout(run, 5000);
      }
    };
    timer = window.setTimeout(run, 250);
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [dataset, version, tick]);
  return { review, setReview, refresh: () => setTick(t => t + 1) };
}

/** The photo calls the bar, the cards and the Left out list share. */
export function usePhotoActions(dataset: string, setReview: (r: PhotoReview) => void, onChanged: () => void) {
  const toast = useToast();
  const base = photoBase(dataset);
  return useMemo(() => {
    const fix = async (ops: PhotoFixOp[], done?: (n: number) => string) => {
      try {
        const r = await api.post<PhotoFix>(`${base}/fix`, { ops, apply: true });
        setReview(r.review);
        if (done) toast.success(done(r.changes.length));
        onChanged();
        return true;
      } catch (e) {
        toast.error(e);
        return false;
      }
    };
    const settings = async (patch: PhotoSettingsPatch) => {
      try {
        setReview(await api.put<PhotoReview>(`${base}/settings`, patch));
      } catch (e) {
        toast.error(e);
      }
    };
    return {
      fix,
      settings,
      leaveOut: (files: string[]) => fix([{ op: 'set_aside', files }], n => (n ? `Left out ${plural(n, 'photo')}` : 'Nothing to leave out')),
      restore: (files: string[] | 'all') =>
        fix([files === 'all' ? { op: 'restore', all: true } : { op: 'restore', files }], n => `Restored ${plural(n, 'photo')}`),
      decide: (file: string, d: PhotoDecision | null) => settings({ decisions: { [file]: d } }),
      reference: (file: string | null) => settings({ reference: file }),
    };
  }, [base, setReview, onChanged, toast]);
}
export type PhotoActions = ReturnType<typeof usePhotoActions>;

// ---- the bar above the grid ----

// a finding's fix (findings[].op), as the studio's Readiness list offers it
const OP_LABEL: Partial<Record<PhotoFixOp['op'], string>> = {
  set_aside: 'Leave them out',
  set_aside_flagged: 'Leave them out',
  keep_best: 'Pick the best',
  bake_orientation: 'Turn upright',
  convert: 'Convert',
  rename_same_stem: 'Rename',
};
const opDone = (op: PhotoFixOp['op'], n: number): string => {
  if (n === 0) return 'Nothing to change';
  switch (op) {
    case 'bake_orientation':
      return `Turned ${plural(n, 'photo')} upright`;
    case 'convert':
      return `Converted ${plural(n, 'file')}`;
    case 'rename_same_stem':
      return `Renamed ${plural(n, 'file')}`;
    default:
      return `Left out ${plural(n, 'photo')}. Restore brings them back.`;
  }
};

type Filter = PhotoFlag | 'flagged' | null;
const gradeTone = (g: PhotoReview['grade']) => (g === 'A' || g === 'B' ? 'tone-ok' : g === 'F' ? 'tone-danger' : 'tone-warn');

/**
 * Counts, grade and the bulk photo actions, beside the caption check.
 * onFilter narrows the grid to the files it names (null: show all).
 */
export function PhotoCheckBar({
  dataset,
  review,
  actions,
  photoJob,
  onFilter,
  onFaceJob,
  onChanged,
  onRefresh,
}: {
  dataset: string;
  review: PhotoReview;
  actions: PhotoActions;
  photoJob: Job | null | undefined;
  onFilter: (files: Set<string> | null) => void;
  onFaceJob: (j: Job) => void;
  onChanged: () => void;
  onRefresh: () => void;
}) {
  const toast = useToast();
  const confirm = useConfirm();
  const [filter, setFilter] = useState<Filter>(null);
  const [picking, setPicking] = useState(false);
  const [details, setDetails] = useState(false);
  const [busy, setBusy] = useState('');

  // flagged: a photo with a flag it still carries (not waived)
  const flagged = useMemo(() => new Set(review.items.filter(i => !!i.level && i.flags.some(f => !i.waived?.includes(f))).map(i => i.file)), [review]);
  const files = useMemo(() => {
    if (!filter) return null;
    return filter === 'flagged' ? flagged : new Set(review.items.filter(i => i.flags.includes(filter)).map(i => i.file));
  }, [filter, flagged, review]);
  useEffect(() => onFilter(files), [files]); // eslint-disable-line react-hooks/exhaustive-deps
  // a filter whose photos were all left out or waived goes back to showing all
  useEffect(() => {
    if (filter && files && files.size === 0) setFilter(null);
  }, [filter, files]);
  useEffect(() => () => onFilter(null), []); // eslint-disable-line react-hooks/exhaustive-deps

  const counts = ORDER.filter(f => (review.counts[f] ?? 0) > 0).map(f => [f, review.counts[f]!] as const);
  // a count reads as a warning when a photo carrying the flag warns for it
  const warnCounts = useMemo(() => new Set(ORDER.filter(f => review.items.some(i => i.flags.includes(f) && flagWarns(f, i, review.kind)))), [review]);
  const dups = review.findings.find(f => f.id === 'duplicates');
  const faces = review.faces;
  const facing = jobActive(photoJob);
  const showFaces = faces.available && review.kind === 'person' && !facing && (faces.checked === 0 || faces.stale || faces.state === 'failed');

  const run = async (what: string, fn: () => Promise<unknown>) => {
    setBusy(what);
    try {
      await fn();
    } finally {
      setBusy('');
    }
  };

  const leaveDups = () =>
    run('dups', () => actions.fix([dups?.op ?? { op: 'set_aside_flagged', flags: ['duplicate'] }], n => `Left out ${plural(n, 'duplicate')}. Restore brings them back.`));

  const checkFaces = async () => {
    const ok = await confirm({
      title: 'Check faces',
      body: (
        <div className="stack" style={{ gap: 8 }}>
          <span>Finds other people, look-alikes, angles and framing. Runs on this server's CPU beside any training, about a minute for 30 photos (39 MB model download the first time).</span>
          <span>Only check photos of yourself or of someone who agreed. Faces are compared on your server; face templates are not stored.</span>
        </div>
      ),
      confirm: 'Check faces',
    });
    if (!ok) return;
    await run('faces', async () => {
      try {
        const j = await api.post<Job & { warnings?: string[] }>(`${photoBase(dataset)}/faces`, {});
        onFaceJob(j);
        j.warnings?.forEach(w => toast.info(w));
      } catch (e) {
        toast.error(e);
      }
    });
  };

  const forget = async () => {
    const ok = await confirm({
      title: 'Forget analysis',
      body: 'Forget the photo scores, face results, face similarities, the reference photo and every Keep anyway and This is intended choice for this dataset. The photos stay.',
      confirm: 'Forget',
      danger: true,
    });
    if (!ok) return;
    await run('forget', async () => {
      try {
        await api.del(`${photoBase(dataset)}/analysis`);
        toast.success('Photo analysis forgotten');
        onRefresh();
      } catch (e) {
        toast.error(e);
      }
    });
  };

  const usable = review.total > 0 ? `${review.usable} usable of ${review.total}` : 'No photos';
  return (
    <div className="photo-check">
      <div className="photo-check-row">
        <span className={`badge ${gradeTone(review.grade)}`} title="Readiness grade">
          {review.grade}
        </span>
        <span className="small num">{usable}</span>
        <span className="small muted ellipsis photo-check-line">
          {review.grade_line}
          {review.provisional && <span className="dim"> · before the face check</span>}
        </span>
        <span className="grow" />
        {review.pending > 0 && (
          <span className="small muted row num" style={{ gap: 6 }}>
            <Spinner size={12} /> Checking {review.total - review.pending} of {review.total}
          </span>
        )}
        <PhotoMenu label="More photo actions">
          {KINDS.map(k => (
            <button key={k.id} role="menuitemradio" aria-checked={review.kind === k.id} onClick={() => k.id !== review.kind && actions.settings({ kind: k.id })}>
              {review.kind === k.id ? <Check /> : <span className="photo-menu-gap" />} {k.label} photos
            </button>
          ))}
          <hr />
          <button role="menuitem" className="is-danger" onClick={forget} disabled={busy === 'forget'}>
            Forget analysis
          </button>
        </PhotoMenu>
      </div>

      {(counts.length > 0 || review.total > 0) && (
        <div className="photo-check-row">
          {counts.map(([f, n]) => (
            <button
              key={f}
              type="button"
              className={`badge photo-count ${warnCounts.has(f) ? 'tone-warn' : ''}${filter === f ? ' on' : ''}`}
              aria-pressed={filter === f}
              onClick={() => setFilter(x => (x === f ? null : f))}
              title={filter === f ? 'Show all' : `Show only these`}
            >
              {n} {countLabel(f, n)}
            </button>
          ))}
          <span className="grow" />
          {flagged.size > 0 && (
            <button className="btn btn-sm" onClick={() => setFilter(x => (x === 'flagged' ? null : 'flagged'))}>
              {filter === 'flagged' ? 'Show all' : `Show ${flagged.size} flagged`}
            </button>
          )}
          {(review.counts.duplicate ?? 0) > 0 && (
            <button className="btn btn-sm" onClick={leaveDups} disabled={!!busy}>
              {busy === 'dups' ? <Spinner /> : <EyeOff size={14} />} Leave out {plural(review.counts.duplicate!, 'duplicate')}
            </button>
          )}
          {review.total > 1 && (
            <button className="btn btn-sm" onClick={() => setPicking(true)} disabled={!!busy}>
              Pick the best…
            </button>
          )}
          {showFaces && (
            <button className="btn btn-sm" onClick={checkFaces} disabled={!!busy}>
              {busy === 'faces' ? <Spinner /> : <ScanFace size={14} />} {faces.checked > 0 ? 'Check faces again' : 'Check faces'}
            </button>
          )}
          {(review.findings.length > 0 || review.hints.length > 0) && (
            <button className="btn btn-sm btn-ghost" onClick={() => setDetails(d => !d)} aria-expanded={details}>
              {details ? 'Hide details' : 'Details'}
            </button>
          )}
        </div>
      )}

      {faces.stale && faces.added > 0 && !facing && (
        <div className="small muted">{plural(faces.added, 'photo')} {faces.added === 1 ? 'was' : 'were'} added after the face check.</div>
      )}

      {details && (
        <div className="card photo-details">
          {review.findings.length > 0 && (
            <ul className="photo-findings">
              {review.findings.map(f => (
                <li key={f.id}>
                  <span className={`dot ${f.level === 'info' ? '' : f.level === 'block' ? 'lv-danger' : 'lv-warn'}`} />
                  <span className="grow">
                    {f.message}
                    {f.fix && <span className="muted"> {f.fix}</span>}
                  </span>
                  {f.op && OP_LABEL[f.op.op] && (
                    <button className="btn btn-sm" disabled={!!busy} onClick={() => run(f.id, () => actions.fix([f.op!], n => opDone(f.op!.op, n)))}>
                      {busy === f.id ? <Spinner /> : null} {OP_LABEL[f.op.op]}
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
          {review.hints.length > 0 && (
            <div className="small">
              <span className="muted">To add: </span>
              {review.hints.join(' · ')}
            </div>
          )}
        </div>
      )}

      {photoJob && <FaceRun job={photoJob} onChange={onChanged} />}
      {picking && <PickDialog dataset={dataset} review={review} actions={actions} onClose={() => setPicking(false)} />}
    </div>
  );
}

// ---- pick the best N ----

function PickDialog({ dataset, review, actions, onClose }: { dataset: string; review: PhotoReview; actions: PhotoActions; onClose: () => void }) {
  const toast = useToast();
  const [n, setN] = useState(String(Math.min(25, Math.max(1, review.total))));
  const [pick, setPick] = useState<PhotoReview['pick'] | null>(null);
  const [busy, setBusy] = useState(false);
  const num = Math.floor(Number(n));
  const valid = num >= 1 && num <= 2000;

  useEffect(() => {
    if (!valid) return;
    const ctrl = new AbortController();
    setPick(null);
    const t = setTimeout(
      () =>
        api.get<PhotoReview>(`${photoBase(dataset)}/check?pick=${num}&wait_ms=3000`, ctrl.signal).then(
          r => setPick(r.pick ?? { n: num, kept: [], left_out: [] }),
          e => !ctrl.signal.aborted && toast.error(e),
        ),
      300,
    );
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [dataset, num, valid]); // eslint-disable-line react-hooks/exhaustive-deps

  const apply = async () => {
    setBusy(true);
    const ok = await actions.fix([{ op: 'keep_best', n: num }], k => (k ? `Left out ${plural(k, 'photo')}. Restore brings them back.` : 'Nothing to leave out'));
    setBusy(false);
    if (ok) onClose();
  };

  return (
    <Modal title="Pick the best photos" onClose={onClose}>
      <div className="modal-body stack" style={{ gap: 12 }}>
        <label className="row" style={{ gap: 8 }}>
          <span>Keep the best</span>
          <input type="number" min={1} max={2000} value={n} onChange={e => setN(e.target.value)} style={{ width: 88 }} aria-label="How many to keep" />
          <span>photos</span>
        </label>
        <span className="small">Picked for sharpness first, then for variety: angles, framing, outfits, places. The rest move to Left out, and Restore brings them back.</span>
        {!pick && valid && (
          <span className="row small">
            <Spinner /> Picking…
          </span>
        )}
        {pick && (
          <>
            <span className="small num">
              Keeps {pick.kept.length} · leaves out {pick.left_out.length}
            </span>
            {pick.note && <Callout>{pick.note}</Callout>}
            {pick.left_out.length > 0 && (
              <ul className="photo-pick-list">
                {pick.left_out.map(l => (
                  <li key={l.file}>
                    <span className="mono ellipsis" title={l.file}>
                      {l.file}
                    </span>
                    <span className="muted ellipsis" title={l.reason}>
                      {l.reason}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
      <div className="modal-actions">
        <button className="btn" onClick={onClose}>
          Cancel
        </button>
        <button className="btn btn-primary" onClick={apply} disabled={!valid || !pick || pick.left_out.length === 0 || busy}>
          {busy ? <Spinner /> : null} Leave out {pick ? pick.left_out.length : ''}
        </button>
      </div>
    </Modal>
  );
}

// ---- the face check job ----

/** The dataset's face check while it runs, or how the last one failed. */
function FaceRun({ job, onChange }: { job: Job; onChange: () => void }) {
  const toast = useToast();
  const [stopping, setStopping] = useState(false);
  const active = jobActive(job);
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
      <Callout tone="danger">
        The face check failed{job.info ? `: ${job.info}` : ''}.{' '}
        <a className="link" href={href('jobs', job.id)}>
          Open the log
        </a>
      </Callout>
    );
  return (
    <div className="card caption-run photo-run">
      <div className="caption-run-head">
        <Spinner />
        <span className="cell-title">{job.state === 'queued' ? 'Waiting for the CPU queue' : 'Checking faces'}</span>
        <span className="grow" />
        {total > 0 && (
          <span className="small muted num">
            {job.step} of {total}
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

// ---- per photo ----

/** The photo's flags in one soft badge; the notes are its tooltip. Silence is the good state. */
export function PhotoFlags({ item }: { item: PhotoItem | undefined }) {
  // the judge moves waived flags out of flags, so a photo can carry only waived ones
  if (!item || (item.flags.length === 0 && !item.waived?.length)) return null;
  const waived = new Set(item.waived ?? []);
  const live = item.flags.filter(f => !waived.has(f));
  const title = [...item.flags, ...waived].map(f => item.notes[f] ?? flagLabel(f)).join('\n');
  return (
    <div className="photo-flags" title={title}>
      {live.length > 0 && <span className={`badge ${item.level === 'warn' ? 'tone-warn' : ''}`}>{live.map(flagLabel).join(', ')}</span>}
      {[...waived].map(f => (
        <span key={f} className="photo-waived">
          {flagLabel(f)}
        </span>
      ))}
    </div>
  );
}

/** The status dot in a thumbnail's corner: warn or info, nothing when the photo is fine. */
export function PhotoDot({ item }: { item: PhotoItem | undefined }) {
  if (!item?.level) return null;
  return <span className={`photo-dot ${item.level === 'warn' ? 'lv-warn' : ''}`} aria-label={item.level === 'warn' ? 'Has warnings' : 'Has notes'} />;
}

/** The other photo of a duplicate, similar or burst pair. */
export function partnerOf(item: PhotoItem | undefined, review: PhotoReview | null): string | undefined {
  if (!item || !review) return undefined;
  if (item.same_as) return item.same_as;
  const g = item.group ? review.groups.find(x => x.id === item.group) : undefined;
  return g?.files.find(f => f !== item.file);
}

/** The card's photo menu: leave out, keep anyway, this is intended, this is them, compare. */
export function PhotoItemMenu({
  item,
  review,
  actions,
  onCompare,
}: {
  item: PhotoItem;
  review: PhotoReview;
  actions: PhotoActions;
  onCompare: (a: string, b: string) => void;
}) {
  const hasWarn = (item.level === 'warn' && item.flags.some(f => !TRAINER_UNREADABLE.has(f) && flagWarns(f, item, review.kind))) || item.decision === 'keep';
  const canIntend = item.flags.some(f => INTENDED.includes(f)) || item.decision === 'intended';
  const isRef = review.reference === item.file || !!item.reference;
  const partner = partnerOf(item, review);
  return (
    <PhotoMenu label={`Photo actions for ${item.file}`} small>
      <button role="menuitem" onClick={() => actions.leaveOut([item.file])}>
        <EyeOff /> Leave out
      </button>
      {hasWarn && (
        <button role="menuitem" onClick={() => actions.decide(item.file, item.decision === 'keep' ? null : 'keep')}>
          {item.decision === 'keep' ? <Undo2 /> : <Check />} {item.decision === 'keep' ? 'Undo keep anyway' : 'Keep anyway'}
        </button>
      )}
      {canIntend && (
        <button role="menuitem" onClick={() => actions.decide(item.file, item.decision === 'intended' ? null : 'intended')}>
          {item.decision === 'intended' ? <Undo2 /> : <Check />} {item.decision === 'intended' ? 'Undo this is intended' : 'This is intended'}
        </button>
      )}
      {review.kind === 'person' && (
        <button role="menuitem" onClick={() => actions.reference(isRef ? null : item.file)}>
          {isRef ? <Undo2 /> : <UserCheck />} {isRef ? 'Clear this is them' : 'This is them'}
        </button>
      )}
      {partner && (
        <button role="menuitem" onClick={() => onCompare(item.file, partner)}>
          <Columns2 /> Compare with {partner.split('/').pop()}
        </button>
      )}
    </PhotoMenu>
  );
}

/** Two photos side by side, to check a duplicate, similar or burst pair by eye. */
export function CompareDialog({
  folder,
  review,
  a,
  b,
  actions,
  onClose,
}: {
  folder: string;
  review: PhotoReview;
  a: string;
  b: string;
  actions: PhotoActions;
  onClose: () => void;
}) {
  const side = (file: string) => {
    const it = review.items.find(i => i.file === file);
    const notes = it ? it.flags.map(f => it.notes[f]).filter(Boolean) : [];
    return (
      <div className="photo-compare-side" key={file}>
        <img className="photo-compare-img" src={imgUrl(joinServerPath(folder, file))} alt={file} />
        <div className="mono ellipsis small" title={file}>
          {file}
        </div>
        {it?.width && it.height ? (
          <div className="item-meta">
            {it.width} × {it.height}
            {it.px ? ` · ${it.px} px` : ''}
          </div>
        ) : null}
        {notes.length > 0 && <div className="small muted">{notes.join(' ')}</div>}
        <div>
          <button className="btn btn-sm" onClick={async () => (await actions.leaveOut([file])) && onClose()}>
            <EyeOff size={14} /> Leave out
          </button>
        </div>
      </div>
    );
  };
  return (
    <Modal title="Compare" onClose={onClose} wide>
      <div className="modal-body photo-compare">
        {side(a)}
        {side(b)}
      </div>
    </Modal>
  );
}

/**
 * The photos left out into .set_aside/, folded away, each with Restore, and
 * apart from them the originals a convert or a turn upright kept there: a
 * backup, which Restore all leaves; Put back swaps one in for the photo made
 * from it.
 */
export function LeftOut({ folder, review, actions }: { folder: string; review: PhotoReview; actions: PhotoActions }) {
  const [open, setOpen] = useState(false);
  const [openOriginals, setOpenOriginals] = useState(false);
  const n = review.set_aside.length;
  const originals = review.originals ?? []; // an older server lists them under set_aside
  if (n === 0 && originals.length === 0) return null;
  return (
    <div className="photo-left-out">
      <div className="row" style={{ gap: 8 }}>
        {n > 0 && (
          <button className="btn btn-sm btn-ghost" onClick={() => setOpen(o => !o)} aria-expanded={open}>
            {open ? 'Hide left out' : `Left out (${n})`}
          </button>
        )}
        {open && n > 1 && (
          <button className="btn btn-sm" onClick={() => actions.restore('all')}>
            <RotateCcw size={14} /> Restore all
          </button>
        )}
        {originals.length > 0 && (
          <button className="btn btn-sm btn-ghost" onClick={() => setOpenOriginals(o => !o)} aria-expanded={openOriginals}>
            {openOriginals ? 'Hide originals' : `Originals kept (${originals.length})`}
          </button>
        )}
      </div>
      {open && (
        <div className="grid photo-left-grid">
          {review.set_aside.map(s => (
            <div className="card item photo-left-item" key={s.file}>
              <img className="item-media" src={imgUrl(joinServerPath(folder, s.file), true)} alt={s.from} loading="lazy" />
              <div className="mono ellipsis" style={{ fontSize: 12 }} title={s.from}>
                {s.from.split('/').pop()}
              </div>
              {s.flags.length > 0 && <div className="item-meta ellipsis">{s.flags.map(flagLabel).join(', ')}</div>}
              <button className="btn btn-sm" onClick={() => actions.restore([s.file])}>
                <RotateCcw size={14} /> Restore
              </button>
            </div>
          ))}
        </div>
      )}
      {openOriginals && originals.length > 0 && (
        <>
          <p className="muted" style={{ fontSize: 12, margin: 0 }}>
            The files a convert or a turn upright replaced, kept in .set_aside as a backup. They are not left out, so Restore all leaves them. Put back swaps one in for the
            photo made from it.
          </p>
          <div className="grid photo-left-grid">
            {originals.map(o => (
              <div className="card item photo-left-item" key={o.file}>
                <img className="item-media" src={imgUrl(joinServerPath(folder, o.file), true)} alt={o.from} loading="lazy" />
                <div className="mono ellipsis" style={{ fontSize: 12 }} title={o.from}>
                  {o.from.split('/').pop()}
                </div>
                <div className="item-meta ellipsis" title={o.of}>
                  original of {o.of.split('/').pop()}
                </div>
                <button className="btn btn-sm" onClick={() => actions.restore([o.file])}>
                  <RotateCcw size={14} /> Put back
                </button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ---- a small "…" menu; Escape or a click outside closes it ----

function PhotoMenu({ label, small, children }: { label: string; small?: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
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
    <div className="photo-menu" ref={box}>
      <button className={small ? 'icon-btn sm' : 'icon-btn'} aria-label={label} title="More" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(o => !o)}>
        <Ellipsis size={14} />
      </button>
      {open && (
        <div className="photo-menu-pop" role="menu" onClick={() => setOpen(false)}>
          {children}
        </div>
      )}
    </div>
  );
}

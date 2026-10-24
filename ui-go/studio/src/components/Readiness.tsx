import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import type { PhotoFinding, PhotoFix, PhotoFixOp, PhotoReview } from '../lib/photoTypes';

// How ready a photo set is to train (PHOTO-PLAN 7.3, 13.1 item 5): the
// grade, the usable count, the coverage the grade rests on and at most three
// hints. Everything here is the server's judge; nothing is decided in the UI.

const gradeTone = (g: string) => (g === 'A' || g === 'B' ? 'tone-ok' : g === 'F' ? 'tone-danger' : 'tone-warn');

/** "Add 2 side profiles." becomes "add 2 side profiles", for the compact line. */
const inline = (s: string) => (s ? s[0].toLowerCase() + s.slice(1) : s).replace(/\.$/, '');

const shotLabels: Record<string, string> = { close: 'Close-up', bust: 'Head and shoulders', half: 'Half-length', full: 'Full-length', back: 'From behind' };
const yawLabels: Record<string, string> = { front: 'Facing the camera', three_quarter: 'Turned partly away', profile: 'Side profile' };

/** Share bars, largest first, told apart by label. */
function Bars({ counts, labels, top = 5 }: { counts: Record<string, number>; labels?: Record<string, string>; top?: number }) {
  const rows = Object.entries(counts)
    .filter(([, n]) => n > 0)
    .sort((a, b) => (labels ? Object.keys(labels).indexOf(a[0]) - Object.keys(labels).indexOf(b[0]) : b[1] - a[1]))
    .slice(0, top);
  const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  if (!rows.length) return <span className="muted">None found</span>;
  return (
    <ul className="readiness-bars">
      {rows.map(([k, n]) => (
        <li key={k}>
          <span className="readiness-bar-label">{labels?.[k] ?? k}</span>
          <span className="readiness-bar" aria-hidden>
            <span style={{ width: `${Math.round((n / total) * 100)}%` }} />
          </span>
          <span className="num muted">{n}</span>
        </li>
      ))}
    </ul>
  );
}

/**
 * The readiness block. compact is the one-line form for the Captions step
 * header. onOp, when given, offers the ready-made fix of a finding.
 */
export function Readiness(p: { review: PhotoReview; compact?: boolean; onOp?(op: PhotoFixOp): void; busy?: boolean }) {
  const { review: r } = p;
  const [open, setOpen] = useState(false);
  const faces = r.coverage.basis.includes('faces');
  const captions = r.coverage.basis.includes('captions');
  const hints = r.hints.slice(0, 3);
  const grade = (
    <span className={`badge ${gradeTone(r.grade)}`} title={r.grade_line}>
      {r.grade}
    </span>
  );
  if (p.compact)
    return (
      <span className="readiness-line">
        Readiness {grade}
        {hints[0] ? <> · {inline(hints[0])}</> : null}
        {r.provisional && <span className="muted"> · faces not checked</span>}
      </span>
    );

  const c = r.coverage;
  // the face crop offer has its own opt-in below (FaceCropOffer: a preview first, D-12), never a one-click fix here
  const findings = r.findings.filter(f => f.id !== 'face_crops_offer' && (f.level !== 'info' || f.op));
  return (
    <section className="readiness card" aria-label="Readiness">
      <header className="readiness-head">
        <p>
          <span className="readiness-title">Readiness</span>
          {grade}
          <span>
            <strong className="num">{r.usable}</strong> usable
          </span>
          {r.provisional && <span className="muted">before the face check</span>}
        </p>
        <button type="button" className="btn btn-sm btn-ghost" aria-expanded={open} onClick={() => setOpen(v => !v)}>
          {open ? <ChevronDown /> : <ChevronRight />}
          Details
        </button>
      </header>
      <p className="readiness-grade-line">{r.grade_line}</p>
      <dl className="readiness-rows">
        {c.dated > 0 && (
          <div>
            <dt>Sessions</dt>
            <dd>
              {c.largest_session > 1 ? (
                <>
                  <span className="num">
                    {c.largest_session} of {c.dated}
                  </span>{' '}
                  taken within one hour
                </>
              ) : (
                <>
                  <span className="num">{c.sessions}</span> different times
                </>
              )}
              {c.span_days > 0 && <span className="muted num"> · over {c.span_days} days</span>}
            </dd>
          </div>
        )}
        {hints.length > 0 && (
          <div>
            <dt>To add</dt>
            <dd>
              <ul className="readiness-hints">
                {hints.map(h => (
                  <li key={h}>{h}</li>
                ))}
              </ul>
            </dd>
          </div>
        )}
      </dl>
      {open && (
        <div className="readiness-details">
          <dl className="readiness-rows">
            <div>
              <dt>Shot types</dt>
              <dd>{faces && c.shot ? <Bars counts={c.shot} labels={shotLabels} /> : <span className="muted">Appears after the face check</span>}</dd>
            </div>
            <div>
              <dt>Angles</dt>
              <dd>{faces && c.yaw ? <Bars counts={c.yaw} labels={yawLabels} /> : <span className="muted">Appears after the face check</span>}</dd>
            </div>
            <div>
              <dt>Sessions</dt>
              <dd>
                {c.dated > 0 ? (
                  <span className="num">
                    {c.sessions} {c.sessions === 1 ? 'session' : 'sessions'} · {c.dated} dated photos
                  </span>
                ) : (
                  <span className="muted">No capture dates</span>
                )}
              </dd>
            </div>
            {(['outfits', 'places', 'light'] as const).map(key => (
              <div key={key}>
                <dt>{key === 'outfits' ? 'Outfits' : key === 'places' ? 'Places' : 'Light'}</dt>
                <dd>{captions && c[key] ? <Bars counts={c[key]!} /> : <span className="muted">Appears after captions</span>}</dd>
              </div>
            ))}
          </dl>
          {findings.length > 0 && <Findings findings={findings} onOp={p.onOp} busy={p.busy} />}
        </div>
      )}
    </section>
  );
}

function Findings({ findings, onOp, busy }: { findings: PhotoFinding[]; onOp?(op: PhotoFixOp): void; busy?: boolean }) {
  return (
    <ul className="readiness-findings">
      {findings.map(f => (
        <li key={f.id}>
          <span className={`dot ${f.level === 'block' ? 'tone-danger' : f.level === 'warn' ? 'tone-warn' : 'tone-quiet'}`} aria-hidden />
          <span>
            {f.message}
            {f.fix && <span className="muted"> {f.fix}</span>}
          </span>
          {f.op && f.op.op !== 'face_crops' && onOp && (
            <button type="button" className="btn btn-sm" disabled={busy} onClick={() => onOp(f.op!)}>
              {opLabel[f.op.op] ?? 'Fix'}
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}

const opLabel: Partial<Record<PhotoFixOp['op'], string>> = {
  set_aside: 'Leave them out',
  set_aside_flagged: 'Leave them out',
  keep_best: 'Pick the best',
  bake_orientation: 'Turn upright',
  convert: 'Convert',
  rename_same_stem: 'Rename',
};

/**
 * "Add close-up crops of the face" (PHOTO-PLAN section 9), shown when the
 * judge offers it (face_crops_offer) or crops exist. It starts unticked:
 * ticking previews the crops, and nothing is written until "Add".
 */
export function FaceCropOffer(p: { review: PhotoReview; onPreview(): Promise<PhotoFix>; onApply(): Promise<void>; onRemove?(): Promise<void> }) {
  const offer = p.review.findings.find(f => f.id === 'face_crops_offer');
  const crops = p.review.items.filter(i => i.derived_from).length;
  const [preview, setPreview] = useState<PhotoFix | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  if (!offer && !crops) return null;
  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError('');
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const k = preview ? preview.changes.length : crops;
  const pct = p.review.usable ? Math.round((k / p.review.usable) * 100) : 0;
  const ticked = crops > 0 || preview !== null;
  return (
    <div className="face-crops">
      <label className="check">
        <input
          type="checkbox"
          checked={ticked}
          disabled={busy}
          onChange={e =>
            run(async () => {
              if (e.target.checked) setPreview(await p.onPreview());
              else if (crops && p.onRemove) await p.onRemove();
              else setPreview(null);
            })
          }
        />
        <span>Add close-up crops of the face</span>
        {ticked && (
          <span className="muted num">
            {k} extra {k === 1 ? 'photo' : 'photos'}
          </span>
        )}
      </label>
      <p className="quiet-note">
        For half- and full-length photos where the face is small. Each crop is a new photo next to the original, without a caption (Auto caption fills it).
        {ticked && k > 0 && <> About {pct}% more steps per epoch.</>}
      </p>
      {preview && !crops && (
        <div className="photo-actions">
          <button
            type="button"
            className="btn btn-sm btn-primary"
            disabled={busy || !preview.changes.length}
            onClick={() =>
              run(async () => {
                await p.onApply();
                setPreview(null);
              })
            }
          >
            Add {preview.changes.length} {preview.changes.length === 1 ? 'crop' : 'crops'}
          </button>
          <button type="button" className="btn btn-sm btn-ghost" disabled={busy} onClick={() => setPreview(null)}>
            Cancel
          </button>
        </div>
      )}
      {error && <p className="photo-error">{error}</p>}
    </div>
  );
}

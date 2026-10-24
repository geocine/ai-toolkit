import { useEffect, useRef, useState } from 'react';
import type { Server } from '../lib/api';
import { checkImage, DEFAULT_REPO, imageRepo, imageTag, imageTags, type ImageCheckRequest } from '../lib/serverImage';
import type { ImageCheck, ImageChoice, ImageRecommendation, ImageTag } from '../lib/serverImageTypes';
import { Dialog } from './Dialog';
import { LevelBadge } from './FitBadge';
import { CheckList } from './ImageRow';
import { ErrorNote } from './ui';

const builtOn = (iso: string) => {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
};

/** "0.13.34 · build f53a407" from "AI Toolkit 0.13.34 · build f53a407". */
const build = (label: string) => label.replace(/^AI Toolkit\s*/, '') || label;

function channelTitle(c: ImageChoice): string {
  switch (c.channel) {
    case 'same':
      return 'Same as this server';
    case 'latest':
      return 'Newest build';
    case 'flavor':
      return c.flavor ? `Other flavor (${c.flavor})` : 'Other flavor';
    case 'template':
      return 'From the template';
    default:
      return 'Custom image';
  }
}

function channelLine(c: ImageChoice, tags: ImageTag[] | null): string {
  switch (c.channel) {
    case 'same':
      return c.revision ? `Build ${c.revision}, the one this server runs. Same API, same bundle format.` : 'The build this server runs. Same API, same bundle format.';
    case 'latest': {
      const t = tags?.find(x => x.digest === c.digest && x.created);
      const when = t ? builtOn(t.created) : '';
      return `latest is ${build(c.label)}${when ? ` (built ${when})` : ''}.`;
    }
    default:
      return c.label;
  }
}

// The first why line when it only repeats the card's title; anything else
// (a newer build than this server, a flavor for this driver) is shown.
const plainWhy = ['Same build as this server', 'Newest build', 'Custom image', 'The image of the QuickPod template'];

type Pick = { image: string | null; source: 'auto' | 'channel' | 'tag' | 'custom' };

/**
 * Change the server image (IMAGE-PLAN §6.3): the candidates of
 * images.recommend, any tag of the repository, or a custom image checked
 * as it is typed. onPick(null) goes back to the recommended image, which
 * follows the chosen offer.
 */
export function ImageDialog({
  server: s,
  rec,
  recError,
  current,
  auto,
  context,
  onPick,
  onClose,
}: {
  server: Server;
  rec: ImageRecommendation | null;
  recError: unknown;
  current: string;
  auto: boolean;
  /** The host and launch the custom image is checked against. */
  context: Omit<ImageCheckRequest, 'image'>;
  onPick: (image: string | null) => void;
  onClose: () => void;
}) {
  const cands = rec ? [...(rec.choice ? [rec.choice] : []), ...rec.alternatives.filter(a => a.channel !== 'template')] : [];
  const known = cands.some(c => c.image === current);
  // tags: for the "built on" date and the Other tags table
  const repo = imageRepo(rec?.choice?.image ?? (current || DEFAULT_REPO)) || DEFAULT_REPO;
  // a picked tag of this repository opens as that tag until the tags show it is not listed
  const [startTag] = useState(!auto && !known && imageRepo(current) === repo && !!imageTag(current));
  const [pick, setPick] = useState<Pick>(
    auto ? { image: null, source: 'auto' } : known ? { image: current, source: 'channel' } : { image: current, source: startTag ? 'tag' : 'custom' },
  );

  // this dialog sits on top of the create dialog: Escape closes only this one
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      e.stopPropagation();
      close.current();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, []);

  const [tags, setTags] = useState<ImageTag[] | null>(null);
  const [tagError, setTagError] = useState<unknown>(null);
  useEffect(() => {
    let live = true;
    imageTags(s, repo, 8).then(
      t => live && setTags(t),
      e => live && setTagError(e),
    );
    return () => {
      live = false;
    };
  }, [s.id, repo]); // eslint-disable-line react-hooks/exhaustive-deps

  // custom image, checked 500 ms after typing stops, and at once on blur or Enter
  const [custom, setCustom] = useState(!auto && !known && !startTag ? current : '');
  // the picked tag is not among the listed ones (or they failed): it is a custom image
  useEffect(() => {
    if (!startTag || (!tags && !tagError)) return;
    if (tags?.some(t => `${repo}:${t.tag}` === current)) return;
    setPick(x => (x.source === 'tag' && x.image === current ? { image: current, source: 'custom' } : x));
    setCustom(c => c || current);
  }, [tags, tagError]); // eslint-disable-line react-hooks/exhaustive-deps
  const [customCheck, setCustomCheck] = useState<{ key: string; ref: string; check: ImageCheck | null; error: unknown } | null>(null);
  const [customBusy, setCustomBusy] = useState(false);
  const seq = useRef(0);
  const asked = useRef(''); // the ref and context of the newest check, so a blur after the debounce asks nothing new
  const ctxKey = JSON.stringify(context);
  const runCheck = (ref: string) => {
    const r = ref.trim();
    const key = `${r}|${ctxKey}`;
    if (!r || asked.current === key) return;
    asked.current = key;
    const n = ++seq.current;
    setCustomBusy(true);
    checkImage(s, { ...context, image: r })
      .then(
        c => n === seq.current && setCustomCheck({ key, ref: r, check: c, error: null }),
        e => {
          if (n !== seq.current) return;
          asked.current = ''; // a failed check may be asked again
          setCustomCheck({ key, ref: r, check: null, error: e });
        },
      )
      .finally(() => n === seq.current && setCustomBusy(false));
  };
  useEffect(() => {
    if (!custom.trim()) {
      seq.current++;
      asked.current = '';
      setCustomCheck(null);
      setCustomBusy(false);
      return;
    }
    const t = setTimeout(() => runCheck(custom), 500);
    return () => clearTimeout(t);
  }, [custom, ctxKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const customRef = custom.trim();
  const customDone = customCheck?.ref === customRef && !customBusy;
  const customBlocked = customDone && customCheck?.check != null && !customCheck.check.ok;
  const canUse = pick.source !== 'custom' || (!!customRef && customDone && !customBlocked);

  const use = () => {
    if (!canUse) return;
    onPick(pick.source === 'auto' ? null : pick.source === 'custom' ? customRef : pick.image);
    onClose();
  };

  const card = (c: ImageChoice, recommended: boolean) => {
    // a picked image that is the choice selects the choice card, and stays a pick
    const on = (recommended && pick.source === 'auto') || (pick.source === 'channel' && pick.image === c.image);
    const why = (plainWhy.includes(c.why[0]) ? c.why.slice(1) : c.why).join(' · ');
    return (
      <label key={`${c.channel}:${c.image}`} className={on ? 'channel-card is-selected' : 'channel-card'}>
        <input
          type="radio"
          name="image-channel"
          className="visually-hidden"
          checked={on}
          onChange={() => setPick(recommended ? { image: null, source: 'auto' } : { image: c.image, source: 'channel' })}
        />
        <span className="channel-card-head">
          <span className="channel-card-name">{channelTitle(c)}</span>
          {recommended && <span className="badge tone-accent">Recommended</span>}
          <LevelBadge level={c.level} />
        </span>
        <span className="channel-card-line">{channelLine(c, tags)}</span>
        {why && <span className="channel-card-why">{why}</span>}
      </label>
    );
  };

  return (
    <Dialog
      wide
      title="Server image"
      description="What the new GPU server runs. The server checks it again when you create the server."
      onClose={onClose}
      footer={
        <>
          <button className="btn btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={use} disabled={!canUse}>
            Use this image
          </button>
        </>
      }
    >
      <div className="dialog-form image-dialog">
        <ErrorNote error={recError} />
        {!rec && !recError && <p className="quiet-note">Checking the images…</p>}
        {rec && (
          <div className="channel-list" role="radiogroup" aria-label="Image">
            {rec.choice && card(rec.choice, true)}
            {cands.slice(rec.choice ? 1 : 0).map(c => card(c, false))}
          </div>
        )}
        {rec && !rec.choice && <CheckList checks={rec.checks} />}

        <details className="image-tags" open={startTag || undefined}>
          <summary>Other tags</summary>
          <ErrorNote error={tagError} />
          {!tags && !tagError && <p className="quiet-note">Loading tags…</p>}
          {tags && tags.length === 0 && <p className="quiet-note">No tags found.</p>}
          {tags && tags.length > 0 && (
            <div className="table-wrap">
              <table className="table image-tag-table">
                <thead>
                  <tr>
                    <th>Tag</th>
                    <th>Built</th>
                    <th>CUDA</th>
                    <th>Build</th>
                    <th aria-label="Use" />
                  </tr>
                </thead>
                <tbody>
                  {tags.map(t => {
                    const ref = `${repo}:${t.tag}`;
                    const on = pick.source === 'tag' && pick.image === ref;
                    return (
                      <tr key={t.tag} className={on ? 'is-selected' : undefined}>
                        <td className="mono">{t.tag}</td>
                        <td className="num">{builtOn(t.created) || '–'}</td>
                        <td className="num">{t.cuda_min || '–'}</td>
                        <td className="mono">{t.revision || '–'}</td>
                        <td>
                          <button type="button" className="btn btn-ghost btn-sm" aria-pressed={on} onClick={() => setPick({ image: ref, source: 'tag' })}>
                            {on ? 'Picked' : 'Pick'}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </details>

        <label className="field image-custom">
          <span>Custom image</span>
          <input
            className="mono"
            value={custom}
            onChange={e => {
              setCustom(e.target.value);
              setPick({ image: null, source: 'custom' });
            }}
            onFocus={() => custom.trim() && setPick({ image: null, source: 'custom' })}
            onBlur={() => runCheck(custom)}
            onKeyDown={e => {
              if (e.key !== 'Enter') return;
              e.preventDefault();
              runCheck(custom);
            }}
            placeholder="ghcr.io/you/aitoolkit-go:tag"
            spellCheck={false}
            autoCapitalize="off"
            aria-invalid={customBlocked}
          />
          {pick.source === 'custom' && customRef && customBusy && <small>Checking…</small>}
          {customCheck?.ref === customRef && customCheck.error != null && <ErrorNote error={customCheck.error} />}
        </label>
        {customCheck?.ref === customRef && customCheck.check && <CheckList checks={customCheck.check.checks} />}
      </div>
    </Dialog>
  );
}

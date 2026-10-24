import type { CheckLevel, OfferFit, VramFit } from '../lib/serverImageTypes';

type Level = 'ok' | CheckLevel;

/**
 * Whether an offer's host runs the image (IMAGE-PLAN §6.2): a dot and a short
 * label from fit.summary. A blocked host is a danger badge, never a stripe.
 * No fit (the image could not be judged) shows nothing.
 */
export function FitBadge({ fit }: { fit: OfferFit | null | undefined }) {
  if (!fit) return null;
  if (fit.level === 'block')
    return (
      <span className="badge tone-danger fit-badge" title={fit.summary}>
        Won't run here
      </span>
    );
  const warn = fit.level === 'warn';
  return (
    <span className={`fit-dot ${warn ? 'tone-warn' : 'tone-ok'}`} title={warn ? fit.summary : undefined}>
      <span className="dot" aria-hidden />
      {warn ? fit.summary : 'Fits'}
    </span>
  );
}

const levelLabels: Record<Level, string> = { ok: 'Fits', info: 'Fits', warn: 'Fits, with warnings', block: "Won't run here" };
const levelTones: Record<Level, string> = { ok: 'tone-ok', info: 'tone-ok', warn: 'tone-warn', block: 'tone-danger' };

/** A tinted badge for one image candidate's verdict on the host. */
export function LevelBadge({ level }: { level: Level }) {
  return <span className={`badge fit-badge ${levelTones[level]}`}>{levelLabels[level]}</span>;
}

const vramTones: Record<VramFit['verdict'], string> = { fits: 'tone-ok', tight: 'tone-warn', no: 'tone-danger', unknown: 'tone-quiet' };
const vramWords: Record<VramFit['verdict'], string> = { fits: 'fits', tight: 'tight', no: 'too small', unknown: '' };

/** The GPU's VRAM, tinted by whether the chosen model fits in it (soft fill, no border). */
export function VramBadge({ gb, verdict, model }: { gb: number; verdict?: VramFit['verdict']; model?: string }) {
  const v = model ? (verdict ?? 'unknown') : 'unknown';
  const title = model && v !== 'unknown' ? `${model}: ${vramWords[v]} in ${gb} GB` : undefined;
  return (
    <span className={`badge fit-vram ${vramTones[v]}`} title={title}>
      {gb} GB
    </span>
  );
}

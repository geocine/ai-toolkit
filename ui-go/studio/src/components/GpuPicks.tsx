import { useState } from 'react';
import { ApiError } from '../lib/api';
import { perHour, providerNames } from '../lib/machines';
import type { GpuPick, GpuRecommendation, Offer } from '../lib/provisionTypes';
import { Callout, ErrorNote } from './ui';

/** The words internal/gpupick uses for its reject codes (RejectLabels). */
export const rejectLabels: Record<string, string> = {
  inet: 'slow internet',
  disk_bw: 'slow disk',
  disk: 'too little disk',
  cuda: 'an old driver',
  cuda_unknown: 'an unknown driver',
  gpu_arch: 'an unsupported GPU',
  reliability: 'low reliability',
  price: 'price above your limit',
  cuda_capacity: 'no CUDA 13 hosts free',
  speed_unknown: 'an unrated GPU',
  gpu_count: 'several GPUs',
  fit_block: 'not runnable',
};

/**
 * The image note that leads rec.message: an image the registry says is
 * missing or private (MASTER §3.10.3). The picks are then judged against the
 * default image's built-in requirements.
 */
export function imageNote(rec: GpuRecommendation): GpuRecommendation['notes'][number] | null {
  return rec.notes.find(n => !n.provider && (n.code === 'image_denied' || n.code === 'image_not_found')) ?? null;
}

/** "RTX 4090 on Vast.ai · $0.34/h": the cross-provider line's offer. */
export const pickLine = (o: Offer) => `${o.gpu} on ${providerNames[o.provider] ?? o.provider} · ${perHour(o.price_per_hour)}`;

const speedTip = 'Training speed compared with an RTX 4090 (an estimate until measured)';

/** The pick an offer stands for in the offer list, for its badge. */
export function pickBadge(rec: GpuRecommendation | null, id: string): string {
  if (rec?.value?.offer.id === id) return 'Recommended';
  if (rec?.speed?.offer.id === id) return 'Best speed';
  return '';
}

/**
 * The recommender's two picks above the offer list (MASTER §5.1.5): best
 * value (most speed per $/h) and best speed, from the smallest VRAM tier the
 * model fits. It only preselects; any offer below stays a click away.
 */
export function GpuPicks({
  rec,
  loading,
  error,
  selectedOfferId,
  onPick,
}: {
  rec: GpuRecommendation | null;
  loading: boolean;
  error: unknown;
  selectedOfferId: string | null;
  onPick(offer: Offer, kind: 'value' | 'speed'): void;
}) {
  const [why, setWhy] = useState(false);
  // a server older than the recommender: the offer list alone, as before
  if (error instanceof ApiError && error.status === 404) return null;
  if (error) {
    const msg =
      error instanceof ApiError && error.code === 'rate_limited'
        ? `Couldn't suggest a GPU: the provider is busy.${error.retryAfter ? ` Try again in ${error.retryAfter} s.` : ''}`
        : `Couldn't suggest a GPU: ${error instanceof Error ? error.message : String(error)}`;
    return (
      <div className="picks">
        <ErrorNote error={new Error(msg)} />
      </div>
    );
  }
  if (!rec)
    return loading ? (
      <div className="pick-list" aria-busy="true">
        {[0, 1].map(i => (
          <div key={i} className="pick is-skeleton" aria-hidden />
        ))}
      </div>
    ) : null;

  const img = imageNote(rec);
  // the image sentence shows on its own, above the picks
  const rest = (img && rec.message.startsWith(img.message) ? rec.message.slice(img.message.length) : rec.message).trim();
  const { value, speed } = rec;
  const same = !!value && !!speed && value.offer.id === speed.offer.id;
  const cards: { pick: GpuPick; title: string; badge: boolean }[] = [];
  if (value) cards.push({ pick: value, title: same ? 'Best value and fastest' : 'Best value', badge: true });
  if (speed && !same) cards.push({ pick: speed, title: 'Best speed', badge: false });

  return (
    <div className="picks" aria-busy={loading || undefined}>
      {img && <Callout kind="warn">{img.message}</Callout>}
      {cards.length > 0 ? (
        <>
          <div className="pick-list" role="radiogroup" aria-label="Suggested GPUs">
            {cards.map(({ pick, title, badge }) => {
              const on = pick.offer.id === selectedOfferId;
              return (
                <button
                  key={pick.kind}
                  type="button"
                  role="radio"
                  aria-checked={on}
                  className={on ? 'pick is-selected' : 'pick'}
                  onClick={() => onPick(pick.offer, pick.kind)}
                >
                  <span className="pick-title">
                    {title}
                    {badge && <span className="badge tone-accent">Recommended</span>}
                  </span>
                  <span className="pick-reason">{pick.reason}</span>
                  <span className="pick-meta" title={speedTip}>
                    {pick.speed.toFixed(2)}× an RTX 4090 · {pick.per_dollar.toFixed(2)} speed per dollar
                  </span>
                </button>
              );
            })}
          </div>
          {rest && <p className="pick-note">{rest}</p>}
        </>
      ) : (
        rest && (
          <>
            <Callout kind="info">
              {rest}{' '}
              {rec.tiers.length > 0 && (
                <button type="button" className="link" aria-expanded={why} onClick={() => setWhy(w => !w)}>
                  Why?
                </button>
              )}
            </Callout>
            {why && rec.tiers.length > 0 && <TierTable tiers={rec.tiers} />}
          </>
        )
      )}
    </div>
  );
}

/** Each VRAM tier the recommender walked, with the two reasons that left out most offers. */
function TierTable({ tiers }: { tiers: GpuRecommendation['tiers'] }) {
  return (
    <div className="table-wrap pick-tiers">
      <table className="table">
        <thead>
          <tr>
            <th>Tier</th>
            <th className="num">Offers</th>
            <th className="num">Qualified</th>
            <th>Left out for</th>
          </tr>
        </thead>
        <tbody>
          {tiers.map(t => {
            const top = Object.entries(t.rejected)
              .filter(([, n]) => n > 0)
              .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
              .slice(0, 2)
              .map(([code, n]) => `${rejectLabels[code] ?? code} (${n})`);
            return (
              <tr key={t.tier_gb}>
                <td>{t.tier_gb} GB</td>
                <td className="num">{t.offers}</td>
                <td className="num">{t.qualified}</td>
                <td className="pick-tiers-why">{top.join(', ') || '–'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

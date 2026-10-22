import { ChevronRight, Clock, TriangleAlert } from 'lucide-react';
import { clock, modeVerb, needsAttention, useLifecycle, useNow } from '../lib/lifecycle';
import { href } from '../lib/route';

/** A countdown or a held server, on every page; tap to open "When done". */
export function LifeBanner() {
  const { life } = useLifecycle();
  const now = useNow();
  if (!life || !needsAttention(life)) return null;
  const counting = life.state === 'countdown' && life.deadline;
  const warn = life.state === 'held' || life.state === 'failed';
  const text = counting
    ? `${modeVerb(life.mode)} this server in ${clock((new Date(life.deadline!).getTime() - now) / 1000)}`
    : life.state === 'held'
      ? 'Held: results only on this server'
      : life.state === 'failed'
        ? 'When done failed'
        : life.state === 'verifying'
          ? 'Checking results before acting'
          : `${modeVerb(life.mode)} this server now`;
  return (
    <a className={warn ? 'life-banner tone-warn' : 'life-banner tone-accent'} href={href({ page: 'lifecycle' })}>
      {warn ? <TriangleAlert aria-hidden /> : <Clock aria-hidden />}
      <span className="life-banner-text">
        <strong className="num">{text}</strong>
        {life.dry_run && <span className="life-banner-sub"> · dry run</span>}
      </span>
      <span className="life-banner-go">
        {counting ? 'Keep or extend' : 'Details'}
        <ChevronRight aria-hidden />
      </span>
    </a>
  );
}

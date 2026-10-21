// Loss over steps as an SVG: the raw points faint, an EMA-smoothed line on
// top, checkpoint steps marked along the bottom.
import { useMemo } from 'react';

interface Props {
  points: { step: number; value: number }[];
  checkpoints?: number[];
  totalSteps?: number | null;
}

const W = 720;
const H = 220;
const PAD = { l: 44, r: 12, t: 12, b: 28 };

export function LossChart({ points, checkpoints = [], totalSteps }: Props) {
  const view = useMemo(() => {
    const pts = points.filter(p => Number.isFinite(p.value));
    if (pts.length < 2) return null;
    const vals = pts.map(p => p.value).sort((a, b) => a - b);
    // clip the top 2% so one spike does not flatten the curve
    const hi = vals[Math.floor(vals.length * 0.98)] ?? vals[vals.length - 1];
    const lo = vals[0];
    // scale to the data (with a little headroom), not to the full run: early
    // in a long run the curve would otherwise be a sliver on the left
    const lastStep = pts[pts.length - 1].step;
    const maxStep = Math.min(totalSteps ?? Infinity, Math.ceil(lastStep * 1.05)) || lastStep;
    const x = (s: number) => PAD.l + (Math.min(s, maxStep) / Math.max(1, maxStep)) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (Math.min(v, hi) - lo) / Math.max(1e-9, hi - lo)) * (H - PAD.t - PAD.b);
    let ema = pts[0].value;
    const alpha = Math.min(0.3, 10 / pts.length);
    const smooth = pts.map(p => {
      ema = alpha * p.value + (1 - alpha) * ema;
      return [x(p.step), y(ema)] as const;
    });
    const ticks = [lo, (lo + hi) / 2, hi];
    return { pts, x, y, smooth, ticks, lastStep };
  }, [points, totalSteps]);

  if (!view) return <div className="chart-empty dim">Loss shows up after the first steps.</div>;
  const { pts, x, y, smooth, ticks, lastStep } = view;
  return (
    <svg className="loss-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="training loss">
      {ticks.map(t => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className="grid-line" />
          <text x={PAD.l - 6} y={y(t) + 4} textAnchor="end" className="axis">
            {t.toPrecision(2)}
          </text>
        </g>
      ))}
      {checkpoints.map(s => (
        <g key={s}>
          <line x1={x(s)} x2={x(s)} y1={PAD.t} y2={H - PAD.b} className="ckpt-line" />
          {/* keep clear of the step counter at the right end */}
          {x(s) < W - PAD.r - 90 && (
            <text x={x(s)} y={H - 8} textAnchor="middle" className="axis axis-ckpt">
              {s}
            </text>
          )}
        </g>
      ))}
      <text x={W - PAD.r} y={H - 8} textAnchor="end" className="axis">
        {lastStep}
        {totalSteps ? ` / ${totalSteps}` : ''}
      </text>
      {pts.map(p => (
        <circle key={p.step} cx={x(p.step)} cy={y(p.value)} r={1.4} className="raw-dot" />
      ))}
      <polyline points={smooth.map(([a, b]) => `${a},${b}`).join(' ')} className="smooth-line" />
    </svg>
  );
}

// Loss over steps as an SVG: the raw points faint, an EMA-smoothed line on
// top, checkpoint steps marked along the bottom. Drawn at the container's
// pixel width so the axis labels stay 11 px.
import { useLayoutEffect, useMemo, useRef, useState } from 'react';

interface Props {
  points: { step: number; value: number }[];
  checkpoints?: number[];
  totalSteps?: number | null;
}

const H = 240;
const PAD = { l: 56, r: 12, t: 12, b: 28 };

export function LossChart({ points, checkpoints = [], totalSteps }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(720);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const measure = () => setW(Math.max(280, Math.round(el.clientWidth)));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

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
    const ticks = [lo, lo + (hi - lo) / 3, lo + ((hi - lo) * 2) / 3, hi];
    return { pts, x, y, smooth, ticks, lastStep };
  }, [points, totalSteps, W]);

  return (
    <div className="chart" ref={box}>
      {!view ? (
        <p className="chart-empty">Loss shows up after the first steps.</p>
      ) : (
        <svg className="loss-chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="training loss">
          {view.ticks.map(t => (
            <g key={t}>
              <line x1={PAD.l} x2={W - PAD.r} y1={view.y(t)} y2={view.y(t)} className="grid-line" />
              <text x={PAD.l - 8} y={view.y(t) + 4} textAnchor="end" className="axis">
                {t.toPrecision(2)}
              </text>
            </g>
          ))}
          {checkpoints.map(s => (
            <g key={s}>
              <line x1={view.x(s)} x2={view.x(s)} y1={PAD.t} y2={H - PAD.b} className="ckpt-line" />
              {/* keep clear of the step counter at the right end */}
              {view.x(s) < W - PAD.r - 90 && (
                <text x={view.x(s)} y={H - 8} textAnchor="middle" className="axis">
                  {s}
                </text>
              )}
            </g>
          ))}
          <text x={W - PAD.r} y={H - 8} textAnchor="end" className="axis">
            {view.lastStep}
            {totalSteps ? ` / ${totalSteps}` : ''}
          </text>
          {view.pts.map(p => (
            <circle key={p.step} cx={view.x(p.step)} cy={view.y(p.value)} r={1.5} className="raw-dot" />
          ))}
          <polyline points={view.smooth.map(([a, b]) => `${a},${b}`).join(' ')} className="smooth-line" />
        </svg>
      )}
    </div>
  );
}

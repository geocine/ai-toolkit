import { useMemo } from 'react';

type Point = { step: number; value: number };

/** Exponential moving average, like TensorBoard's smoothing slider. */
const ema = (pts: Point[], alpha: number) => {
  let last: number | null = null;
  return pts.map(p => {
    last = last === null ? p.value : last * alpha + p.value * (1 - alpha);
    return { step: p.step, value: last };
  });
};

const niceTicks = (min: number, max: number, count: number) => {
  if (max <= min) return [min];
  const raw = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(v);
  return out;
};

const fmt = (v: number) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(1) : Number(v.toPrecision(3)).toString());

export function LossChart({ points, smoothing, log }: { points: Point[]; smoothing: number; log: boolean }) {
  const W = 800;
  const H = 280;
  const pad = { l: 52, r: 12, t: 12, b: 26 };
  const tf = (v: number) => (log ? Math.log10(Math.max(v, 1e-12)) : v);

  const view = useMemo(() => {
    const pts = points.filter(p => Number.isFinite(p.value));
    if (pts.length === 0) return null;
    const smooth = ema(pts, smoothing);
    const xs = pts.map(p => p.step);
    const ys = pts.map(p => tf(p.value));
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs, x0 + 1);
    // clip the y range to the 1st..99th percentile so a spike does not flatten the curve
    const sorted = [...ys].sort((a, b) => a - b);
    let y0 = sorted[Math.floor(sorted.length * 0.01)];
    let y1 = sorted[Math.ceil(sorted.length * 0.99) - 1];
    if (y1 <= y0) {
      y0 -= 1;
      y1 += 1;
    }
    const margin = (y1 - y0) * 0.08;
    y0 -= margin;
    y1 += margin;
    const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
    const sy = (y: number) => pad.t + (1 - (Math.min(Math.max(y, y0), y1) - y0) / (y1 - y0)) * (H - pad.t - pad.b);
    const path = (arr: Point[]) => arr.map((p, i) => `${i ? 'L' : 'M'}${sx(p.step).toFixed(1)},${sy(tf(p.value)).toFixed(1)}`).join('');
    return {
      raw: path(pts),
      smooth: path(smooth),
      xTicks: niceTicks(x0, x1, 6).map(v => ({ v, x: sx(v) })),
      yTicks: niceTicks(y0, y1, 5).map(v => ({ v, y: sy(v) })),
      last: smooth[smooth.length - 1],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, smoothing, log]);

  if (!view) return null;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Loss over steps">
      {view.yTicks.map(t => (
        <g key={'y' + t.v}>
          <line className="grid-line" x1={pad.l} x2={W - pad.r} y1={t.y} y2={t.y} />
          <text className="tick" x={pad.l - 6} y={t.y + 3} textAnchor="end">
            {log ? fmt(Math.pow(10, t.v)) : fmt(t.v)}
          </text>
        </g>
      ))}
      {view.xTicks.map(t => (
        <text key={'x' + t.v} className="tick" x={t.x} y={H - 8} textAnchor="middle">
          {Math.round(t.v)}
        </text>
      ))}
      <line className="axis" x1={pad.l} x2={pad.l} y1={pad.t} y2={H - pad.b} />
      <line className="axis" x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} />
      <path className="raw" d={view.raw} vectorEffect="non-scaling-stroke" />
      <path className="smooth" d={view.smooth} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

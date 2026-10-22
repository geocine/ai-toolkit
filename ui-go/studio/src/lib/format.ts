export function bytes(n: number): string {
  if (!Number.isFinite(n)) return '–';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(n >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}

export function ago(iso: string | null | undefined): string {
  if (!iso) return '–';
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 45) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

/** Steps per second from ai-toolkit's speed string ("1.52it/s", "3.1s/it", "2.07 iter/sec"). */
export function stepsPerSecond(speed: string): number | null {
  const m = /([\d.]+)\s*(it\/s|s\/it|iter\/sec|sec\/iter)/.exec(speed || '');
  if (!m) return null;
  const v = parseFloat(m[1]);
  if (!v) return null;
  return m[2] === 'it/s' || m[2] === 'iter/sec' ? v : 1 / v;
}

export function duration(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '–';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (h) return `${h} h ${m} min`;
  if (m) return `${m} min`;
  return `${Math.max(1, Math.round(sec))} s`;
}

export const pad = (n: number, w = 4) => String(n).padStart(w, '0');

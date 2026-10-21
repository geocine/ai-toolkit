// Minimal client for the Go API (same contract as upstream's Next.js API).

const TOKEN_KEY = 'AI_TOOLKIT_AUTH';

export const getToken = (): string => {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
};

export const setToken = (token: string) => {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // private mode: the token just won't persist
  }
};

export class Unauthorized extends Error {}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const res = await fetch(path, {
    ...init,
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...init?.headers },
  });
  if (res.status === 401) throw new Unauthorized();
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export interface Job {
  id: string;
  name: string;
  gpu_ids: string;
  status: string;
  step: number;
  info: string;
  job_type: string;
  updated_at: string;
}

export interface GpuInfo {
  hasNvidiaSmi: boolean;
  gpus: { index: number; name: string; utilization: { gpu: number }; memory: { total: number; used: number } }[];
}

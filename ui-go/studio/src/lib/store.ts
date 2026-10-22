// localStorage with failures tolerated (private windows, blocked storage).
import { useCallback, useEffect, useState } from 'react';

export function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? ((JSON.parse(raw) as T | null) ?? fallback) : fallback;
  } catch {
    return fallback;
  }
}

export function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* not persisted; the app still works for this session */
  }
}

export function useStored<T>(key: string, fallback: T): [T, (v: T | ((p: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => load(key, fallback));
  useEffect(() => save(key, value), [key, value]);
  const set = useCallback((v: T | ((p: T) => T)) => setValue(v as any), []);
  return [value, set];
}

export const uid = () => Math.random().toString(36).slice(2, 10);

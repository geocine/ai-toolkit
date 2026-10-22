// Light, dark, or follow the system. Applied as html[data-theme]; "system"
// removes the attribute so the prefers-color-scheme tokens take over.
import { useCallback, useEffect, useState } from 'react';
import { load, save } from './store';

export type Theme = 'system' | 'light' | 'dark';

export const THEME_KEY = 'aitk.studio.theme';

export function applyTheme(t: Theme) {
  const el = document.documentElement;
  if (t === 'system') el.removeAttribute('data-theme');
  else el.setAttribute('data-theme', t);
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    const t = load<Theme>(THEME_KEY, 'system');
    return t === 'light' || t === 'dark' ? t : 'system';
  });
  useEffect(() => applyTheme(theme), [theme]);
  const set = useCallback((t: Theme) => {
    setTheme(t);
    save(THEME_KEY, t);
  }, []);
  return [theme, set];
}

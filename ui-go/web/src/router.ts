import { useEffect, useState } from 'react';

// Hash routes: #/jobs, #/jobs/<id>, #/datasets/<name>, ... (no server routes
// needed, and links keep working when the console sits behind a proxy path).

const current = () => {
  const h = window.location.hash.replace(/^#/, '');
  return h.startsWith('/') ? h : '/';
};

export function useRoute(): string[] {
  const [path, setPath] = useState(current);
  useEffect(() => {
    const on = () => setPath(current());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return path
    .split('?')[0]
    .split('/')
    .filter(Boolean)
    .map(s => decodeURIComponent(s));
}

export const href = (...parts: string[]) => '#/' + parts.map(encodeURIComponent).join('/');
export const go = (...parts: string[]) => {
  window.location.hash = href(...parts);
};

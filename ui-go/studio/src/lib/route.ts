// Hash routes, so dist/ can be hosted as plain files anywhere.
import { useEffect, useState } from 'react';

export type Route =
  | { page: 'runs' }
  | { page: 'run'; id: string }
  | { page: 'new' }
  | { page: 'presets' }
  | { page: 'servers' };

export function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  switch (parts[0]) {
    case 'runs':
      return parts[1] ? { page: 'run', id: parts[1] } : { page: 'runs' };
    case 'new':
      return { page: 'new' };
    case 'presets':
      return { page: 'presets' };
    case 'servers':
      return { page: 'servers' };
  }
  return { page: 'runs' };
}

export function href(r: Route): string {
  return r.page === 'run' ? `#/runs/${encodeURIComponent(r.id)}` : `#/${r.page}`;
}

export function go(r: Route) {
  window.location.hash = href(r);
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash));
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

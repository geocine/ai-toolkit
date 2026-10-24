// Hash routes, so dist/ can be hosted as plain files anywhere.
import { useEffect, useState } from 'react';

export type Route =
  | { page: 'runs' }
  | { page: 'run'; id: string }
  | { page: 'new' }
  | { page: 'presets'; tab?: 'data' }
  | { page: 'lifecycle' }
  | { page: 'servers'; create?: boolean };

export function parse(hash: string): Route {
  const [path, query = ''] = hash.replace(/^#\/?/, '').split('?', 2);
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  switch (parts[0]) {
    case 'runs':
      return parts[1] ? { page: 'run', id: parts[1] } : { page: 'runs' };
    case 'new':
      return { page: 'new' };
    case 'presets':
      return parts[1] === 'data' ? { page: 'presets', tab: 'data' } : { page: 'presets' };
    case 'lifecycle':
      return { page: 'lifecycle' };
    case 'servers':
      // ?create=1 opens the create dialog once (Servers drops it from the hash)
      return new URLSearchParams(query).get('create') === '1' ? { page: 'servers', create: true } : { page: 'servers' };
  }
  return { page: 'runs' };
}

export function href(r: Route): string {
  if (r.page === 'run') return `#/runs/${encodeURIComponent(r.id)}`;
  if (r.page === 'presets' && r.tab) return `#/presets/${r.tab}`;
  if (r.page === 'servers' && r.create) return '#/servers?create=1';
  return `#/${r.page}`;
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

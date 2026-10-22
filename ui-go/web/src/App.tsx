// Management console for the headless server: an operator's view of what the
// server is doing (jobs, datasets, remotes, settings, keys). Daily training
// happens in the studio app; this is for stepping in.
import { useEffect, useState, type ReactNode } from 'react';
import { Boxes, Cloud, Database, Gauge, KeyRound, ListChecks, LogOut, Menu, Terminal, X } from 'lucide-react';
import { api, onUnauthorized, setToken } from './api';
import { href, useRoute } from './router';
import type { Health } from './types';
import { ConfirmProvider, Spinner, ThemeToggle, ToastProvider } from './ui';
import { Login } from './pages/Login';
import { Overview } from './pages/Overview';
import { Jobs } from './pages/Jobs';
import { JobDetail } from './pages/JobDetail';
import { Datasets } from './pages/Datasets';
import { DatasetDetail } from './pages/DatasetDetail';
import { Remotes } from './pages/Remotes';
import { SettingsPage } from './pages/Settings';

type Auth = 'checking' | 'in' | 'out';

export function App() {
  const [auth, setAuth] = useState<Auth>('checking');
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.get<Health>('/v1/health').then(setHealth, () => setHealth(null));
    // an open server (no token configured) accepts us without one
    api.get('/v1/jobs?state=running').then(
      () => setAuth('in'),
      () => setAuth('out'),
    );
    return onUnauthorized(() => setAuth('out'));
  }, []);

  return (
    <ToastProvider>
      <ConfirmProvider>
        {auth === 'checking' ? (
          <div className="loading center">
            <Spinner size={18} />
          </div>
        ) : auth === 'out' ? (
          <Login health={health} onIn={() => setAuth('in')} />
        ) : (
          <Shell
            health={health}
            onLogout={() => {
              setToken('');
              setAuth('out');
            }}
          />
        )}
      </ConfirmProvider>
    </ToastProvider>
  );
}

const NAV: { to: string; label: string; icon: ReactNode }[] = [
  { to: '', label: 'Overview', icon: <Gauge size={16} /> },
  { to: 'jobs', label: 'Jobs', icon: <ListChecks size={16} /> },
  { to: 'datasets', label: 'Datasets', icon: <Database size={16} /> },
  { to: 'remotes', label: 'Remotes', icon: <Cloud size={16} /> },
  { to: 'settings', label: 'Settings', icon: <KeyRound size={16} /> },
];

function Brand({ health }: { health: Health | null }) {
  return (
    <div className="brand">
      <div className="brand-mark">
        <Boxes size={14} />
      </div>
      <div className="brand-text">
        <div className="brand-name">AI Toolkit</div>
        <div className="brand-sub">Console{health?.upstream_version ? ` · ${health.upstream_version}` : ''}</div>
      </div>
    </div>
  );
}

function Shell({ health, onLogout }: { health: Health | null; onLogout: () => void }) {
  const route = useRoute();
  const top = route[0] ?? '';
  const [menu, setMenu] = useState(false);
  const routeKey = route.join('/');
  useEffect(() => setMenu(false), [routeKey]);
  useEffect(() => {
    if (!menu) return;
    const on = (e: KeyboardEvent) => e.key === 'Escape' && setMenu(false);
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [menu]);

  const nav = (
    <nav className="nav">
      {NAV.map(n => (
        <a key={n.label} href={href(...(n.to ? [n.to] : []))} className={top === n.to ? 'nav-item active' : 'nav-item'} aria-current={top === n.to ? 'page' : undefined}>
          {n.icon}
          {n.label}
        </a>
      ))}
    </nav>
  );
  const foot = (
    <div className="sidebar-foot">
      <a className="nav-item" href="/v1/openapi.json" target="_blank" rel="noreferrer">
        <Terminal size={16} /> API reference
      </a>
      <button type="button" className="nav-item btn-reset" onClick={onLogout}>
        <LogOut size={16} /> Sign out
      </button>
      <div className="foot-row">
        <span>Theme</span>
        <ThemeToggle />
      </div>
      <div className="foot-row">
        <span className="server-status" title={health ? `Server ${health.version}, started ${new Date(health.started_at).toLocaleString()}` : 'Server not reachable'}>
          <span className="dot" style={{ ['--c' as string]: health?.ok ? 'var(--ok)' : 'var(--text-3)' }} />
          <span className="ellipsis">{health ? <>Connected · <span className="mono xs">{health.version}</span></> : 'Server not reachable'}</span>
        </span>
      </div>
    </div>
  );
  return (
    <div className="shell">
      <aside className="sidebar">
        <Brand health={health} />
        {nav}
        {foot}
      </aside>
      <div className="main">
        <header className="topbar">
          <Brand health={health} />
          <button className="icon-btn" onClick={() => setMenu(m => !m)} aria-label={menu ? 'Close menu' : 'Open menu'} aria-expanded={menu}>
            {menu ? <X size={18} /> : <Menu size={18} />}
          </button>
          {menu && (
            <>
              <div className="menu-scrim" onClick={() => setMenu(false)} />
              <div className="menu">
                {nav}
                {foot}
              </div>
            </>
          )}
        </header>
        <main className="page">
          <Page route={route} />
        </main>
      </div>
    </div>
  );
}

function Page({ route }: { route: string[] }) {
  switch (route[0]) {
    case 'jobs':
      return route[1] ? <JobDetail id={route[1]} /> : <Jobs />;
    case 'datasets':
      return route[1] ? <DatasetDetail name={route[1]} /> : <Datasets />;
    case 'remotes':
      return <Remotes />;
    case 'settings':
      return <SettingsPage />;
    default:
      return <Overview />;
  }
}

// Management console for the headless server: an operator's view of what the
// server is doing (jobs, datasets, remotes, settings, keys). Daily training
// happens in the studio app; this is for stepping in.
import { useEffect, useState, type ReactNode } from 'react';
import { Boxes, Cloud, Database, Gauge, KeyRound, ListChecks, LogOut, Terminal } from 'lucide-react';
import { api, onUnauthorized, setToken } from './api';
import { href, useRoute } from './router';
import type { Health } from './types';
import { ConfirmProvider, Spinner, ToastProvider } from './ui';
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
          <div className="login">
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
  { to: '', label: 'Overview', icon: <Gauge size={14} /> },
  { to: 'jobs', label: 'Jobs', icon: <ListChecks size={14} /> },
  { to: 'datasets', label: 'Datasets', icon: <Database size={14} /> },
  { to: 'remotes', label: 'Remotes', icon: <Cloud size={14} /> },
  { to: 'settings', label: 'Settings', icon: <KeyRound size={14} /> },
];

function Brand({ health }: { health: Health | null }) {
  return (
    <div className="row">
      <div className="brand-mark">
        <Boxes size={14} />
      </div>
      <div>
        <div className="brand-name">AI Toolkit</div>
        <div className="brand-sub">Console{health?.upstream_version ? ` · ${health.upstream_version}` : ''}</div>
      </div>
    </div>
  );
}

function Shell({ health, onLogout }: { health: Health | null; onLogout: () => void }) {
  const route = useRoute();
  const top = route[0] ?? '';
  const nav = (
    <nav className="nav">
      {NAV.map(n => (
        <a key={n.label} href={href(...(n.to ? [n.to] : []))} className={top === n.to ? 'active' : ''}>
          {n.icon}
          {n.label}
        </a>
      ))}
    </nav>
  );
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Brand health={health} />
        </div>
        {nav}
        <div className="sidebar-foot">
          <a className="nav-link small muted row" href="/v1/openapi.json" target="_blank" rel="noreferrer">
            <Terminal size={12} /> API reference
          </a>
          <button className="btn btn-sm" onClick={onLogout}>
            <LogOut size={12} /> Sign out
          </button>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <div className="brand">
            <Brand health={health} />
            <button className="icon-btn" onClick={onLogout} aria-label="Sign out" title="Sign out">
              <LogOut size={14} />
            </button>
          </div>
          {nav}
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

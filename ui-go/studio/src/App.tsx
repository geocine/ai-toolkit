import { Layers, List, Menu, Monitor, Moon, Plus, Server as ServerIcon, Sun, X } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';
import { useLoad } from './components/ui';
import { get, type Health, type Server } from './lib/api';
import { StudioProvider, useStudio } from './lib/context';
import { href, useRoute, type Route } from './lib/route';
import { useTheme, type Theme } from './lib/theme';
import { NewRun } from './pages/NewRun';
import { Presets } from './pages/Presets';
import { RunView } from './pages/RunView';
import { Runs } from './pages/Runs';
import { Servers } from './pages/Servers';

const nav: { label: string; icon: ReactNode; route: Route; match: Route['page'][] }[] = [
  { label: 'Runs', icon: <List />, route: { page: 'runs' }, match: ['runs', 'run'] },
  { label: 'New run', icon: <Plus />, route: { page: 'new' }, match: ['new'] },
  { label: 'Presets', icon: <Layers />, route: { page: 'presets' }, match: ['presets'] },
  { label: 'Servers', icon: <ServerIcon />, route: { page: 'servers' }, match: ['servers'] },
];

function Mark() {
  return (
    <svg className="brand-mark" viewBox="0 0 20 20" aria-hidden>
      <rect width="20" height="20" rx="6" fill="var(--accent)" />
      <path d="M10 4.5a5.5 5.5 0 1 0 0 11z" fill="var(--accent-fg)" />
      <circle cx="10" cy="10" r="5.5" fill="none" stroke="var(--accent-fg)" strokeWidth="1.5" />
    </svg>
  );
}

/** Reachability of the active server, for the sidebar footer. */
function ServerStatus({ server }: { server: Server }) {
  const health = useLoad(server, s => get<Health>(s, '/v1/health'), [], 30000);
  const state = health.error ? 'down' : health.data ? 'up' : 'checking';
  const label = state === 'up' ? 'Connected' : state === 'down' ? 'Unreachable' : 'Checking';
  return (
    <p className={`conn conn-${state}`} title={health.error instanceof Error ? health.error.message : undefined}>
      <span className="dot" aria-hidden />
      {label}
      {state === 'up' && health.data?.upstream_version && <span className="conn-version">ai-toolkit {health.data.upstream_version}</span>}
    </p>
  );
}

const themes: { value: Theme; label: string; icon: ReactNode }[] = [
  { value: 'system', label: 'System theme', icon: <Monitor /> },
  { value: 'light', label: 'Light theme', icon: <Sun /> },
  { value: 'dark', label: 'Dark theme', icon: <Moon /> },
];

function ThemeToggle() {
  const [theme, setTheme] = useTheme();
  return (
    <div className="theme-toggle" role="radiogroup" aria-label="Theme">
      {themes.map(t => (
        <button
          key={t.value}
          type="button"
          role="radio"
          aria-checked={theme === t.value}
          aria-label={t.label}
          title={t.label}
          className={theme === t.value ? 'is-on' : ''}
          onClick={() => setTheme(t.value)}
        >
          {t.icon}
        </button>
      ))}
    </div>
  );
}

function Shell() {
  const route = useRoute();
  const { servers, active, setActiveId } = useStudio();
  const [menu, setMenu] = useState(false);
  useEffect(() => setMenu(false), [route]);
  return (
    <div className="app">
      <aside className={menu ? 'sidebar is-open' : 'sidebar'}>
        <div className="sidebar-top">
          <a className="brand" href={href({ page: 'runs' })}>
            <Mark />
            <span className="brand-name">Studio</span>
            <span className="brand-sub">ai-toolkit</span>
          </a>
          <button
            className="btn btn-ghost btn-icon menu-btn"
            aria-label={menu ? 'Close menu' : 'Open menu'}
            aria-expanded={menu}
            onClick={() => setMenu(m => !m)}
          >
            {menu ? <X /> : <Menu />}
          </button>
        </div>
        <div className="sidebar-body">
          <nav className="nav">
            {nav.map(t => {
              const on = t.match.includes(route.page);
              return (
                <a key={t.label} href={href(t.route)} className={on ? 'nav-item is-active' : 'nav-item'} aria-current={on ? 'page' : undefined}>
                  {t.icon}
                  {t.label}
                </a>
              );
            })}
          </nav>
          <div className="sidebar-foot">
            <div className="server-pick">
              <label htmlFor="server-pick">Server</label>
              {servers.length ? (
                <>
                  <select id="server-pick" value={active?.id ?? ''} onChange={e => setActiveId(e.target.value)}>
                    {servers.map(s => (
                      <option key={s.id} value={s.id}>
                        {s.name}
                      </option>
                    ))}
                  </select>
                  {active && <ServerStatus key={active.id} server={active} />}
                </>
              ) : (
                <a className="conn conn-none" href={href({ page: 'servers' })}>
                  <span className="dot" aria-hidden />
                  None yet, add one
                </a>
              )}
            </div>
            <div className="theme-row">
              <span>Theme</span>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </aside>
      {menu && <div className="scrim" onClick={() => setMenu(false)} aria-hidden />}
      <main className="main">
        <div className="page" key={route.page + (route.page === 'run' ? route.id : '')}>
          {route.page === 'runs' && <Runs />}
          {route.page === 'run' && <RunView id={route.id} />}
          {route.page === 'new' && <NewRun />}
          {route.page === 'presets' && <Presets />}
          {route.page === 'servers' && <Servers />}
        </div>
      </main>
    </div>
  );
}

export function App() {
  return (
    <StudioProvider>
      <Shell />
    </StudioProvider>
  );
}

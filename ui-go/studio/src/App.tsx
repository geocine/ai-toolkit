import { Layers, List, Menu, Monitor, Moon, Plus, Server as ServerIcon, Sun, Timer, X } from 'lucide-react';
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { StudioApiNote } from './components/StudioApiNote';
import { LifeBanner } from './components/LifeBanner';
import { PageHeader, useLoad } from './components/ui';
import { get, type Health } from './lib/api';
import { StudioProvider, useStudio } from './lib/context';
import { LifecycleProvider } from './lib/lifecycle';
import { href, useRoute, type Route } from './lib/route';
import { useTheme, type Theme } from './lib/theme';
import { Lifecycle } from './pages/Lifecycle';
import { NewRun } from './pages/NewRun';
import { Presets } from './pages/Presets';
import { RunView } from './pages/RunView';
import { Runs } from './pages/Runs';
import { Servers } from './pages/Servers';

const nav: { label: string; icon: ReactNode; route: Route; match: Route['page'][] }[] = [
  { label: 'Runs', icon: <List />, route: { page: 'runs' }, match: ['runs', 'run'] },
  { label: 'New run', icon: <Plus />, route: { page: 'new' }, match: ['new'] },
  { label: 'Presets', icon: <Layers />, route: { page: 'presets' }, match: ['presets'] },
  { label: 'When done', icon: <Timer />, route: { page: 'lifecycle' }, match: ['lifecycle'] },
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

/**
 * What the active server's last health check said it is. mode is null when it
 * has not answered (unreachable, or no server); pending holds until the first
 * check for the active server comes back either way.
 */
type Mode = NonNullable<Health['mode']>;
const ActiveMode = createContext<{ mode: Mode | null; pending: boolean }>({ mode: null, pending: false });

/**
 * A page that needs the trainer: the studio API shows the note instead, under the
 * same nav. Until the first check answers only the header shows, so the page
 * neither flashes nor starts its own loads against the studio API.
 */
function Trainer({ title, children }: { title: string; children: ReactNode }) {
  const { mode, pending } = useContext(ActiveMode);
  if (pending) return <PageHeader title={title} />;
  return mode === 'studio-api' ? <StudioApiNote title={title} /> : <>{children}</>;
}

/** Reachability of the active server, for the sidebar footer. */
function ServerStatus({ health }: { health: { data: Health | null; error: unknown } }) {
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
  // each answer carries its server id: right after a switch the previous
  // server's answer is still held for a render and must not count for this one.
  // A failed poll keeps that server's last health, as useLoad would.
  const lastOk = useRef<{ id: string; data: Health } | null>(null);
  const probe = useLoad(
    active,
    async s => {
      try {
        const data = await get<Health>(s, '/v1/health');
        lastOk.current = { id: s.id, data };
        return { id: s.id, data: data as Health | null, error: null as unknown };
      } catch (error) {
        return { id: s.id, data: lastOk.current?.id === s.id ? lastOk.current.data : null, error };
      }
    },
    [],
    30000,
  );
  const health = probe.data && probe.data.id === active?.id ? probe.data : null;
  // older servers send no mode: they are ordinary servers
  const activeMode: Mode | null = health?.data ? (health.data.mode ?? 'server') : null;
  const pending = !!active && !health;
  return (
    <ActiveMode.Provider value={{ mode: activeMode, pending }}>
    <LifecycleProvider server={active}>
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
                  {active && <ServerStatus health={health ?? { data: null, error: null }} />}
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
          {route.page !== 'lifecycle' && <LifeBanner />}
          {route.page === 'runs' && (
            <Trainer title="Runs">
              <Runs />
            </Trainer>
          )}
          {route.page === 'run' && (
            <Trainer title="Run">
              <RunView id={route.id} />
            </Trainer>
          )}
          {route.page === 'new' && (
            <Trainer title="New run">
              <NewRun />
            </Trainer>
          )}
          {route.page === 'presets' &&
            (route.tab === 'data' ? (
              <Trainer title="Presets">
                <Presets tab="data" />
              </Trainer>
            ) : (
              <Presets tab="settings" />
            ))}
          {route.page === 'lifecycle' && (
            <Trainer title="When done">
              <Lifecycle />
            </Trainer>
          )}
          {route.page === 'servers' && <Servers create={route.create} activeMode={activeMode} />}
        </div>
      </main>
    </div>
    </LifecycleProvider>
    </ActiveMode.Provider>
  );
}

export function App() {
  return (
    <StudioProvider>
      <Shell />
    </StudioProvider>
  );
}

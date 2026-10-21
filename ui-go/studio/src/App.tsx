import { StudioProvider, useStudio } from './lib/context';
import { href, useRoute, type Route } from './lib/route';
import { NewRun } from './pages/NewRun';
import { Presets } from './pages/Presets';
import { RunView } from './pages/RunView';
import { Runs } from './pages/Runs';
import { Servers } from './pages/Servers';

const tabs: { label: string; route: Route; match: Route['page'][] }[] = [
  { label: 'Runs', route: { page: 'runs' }, match: ['runs', 'run'] },
  { label: 'New run', route: { page: 'new' }, match: ['new'] },
  { label: 'Presets', route: { page: 'presets' }, match: ['presets'] },
  { label: 'Servers', route: { page: 'servers' }, match: ['servers'] },
];

function Shell() {
  const route = useRoute();
  const { servers, active, setActiveId } = useStudio();
  return (
    <div className="shell">
      <header className="masthead">
        <a className="brand" href={href({ page: 'runs' })}>
          <span className="brand-mark">◐</span>
          <span className="brand-name">
            Studio<em>for ai-toolkit</em>
          </span>
        </a>
        <nav className="tabs">
          {tabs.map(t => (
            <a key={t.label} href={href(t.route)} className={t.match.includes(route.page) ? 'tab tab-on' : 'tab'}>
              {t.label}
            </a>
          ))}
        </nav>
        <label className="server-pick">
          <span className="kicker">Server</span>
          {servers.length ? (
            <select value={active?.id ?? ''} onChange={e => setActiveId(e.target.value)}>
              {servers.map(s => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          ) : (
            <a href={href({ page: 'servers' })}>none — add one</a>
          )}
        </label>
      </header>
      <main className="page" key={route.page + (route.page === 'run' ? route.id : '')}>
        {route.page === 'runs' && <Runs />}
        {route.page === 'run' && <RunView id={route.id} />}
        {route.page === 'new' && <NewRun />}
        {route.page === 'presets' && <Presets />}
        {route.page === 'servers' && <Servers />}
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

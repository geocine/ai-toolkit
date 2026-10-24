import { BookOpen, Cpu, Plus, Search } from 'lucide-react';
import { useEffect, useState } from 'react';
import { CreateGpuServer } from '../components/CreateGpuServer';
import { FindMachines } from '../components/FindMachines';
import { MachineRow } from '../components/MachineRow';
import { ProviderGuide } from '../components/ProviderGuide';
import { ErrorNote, PageHeader, Section } from '../components/ui';
import { get, type Health, type Server, type SystemInfo } from '../lib/api';
import { useStudio } from '../lib/context';
import { autoLink, canAutoLink, clearLinked, mayBill, providerNames, relinked, useMachines, type Tracked } from '../lib/machines';
import { providers } from '../lib/provision';
import type { ProviderKind } from '../lib/provisionTypes';
import { href } from '../lib/route';
import { uid } from '../lib/store';

type Check = { ok: true; health: Health; system: SystemInfo } | { ok: false; error: unknown };

async function check(s: Server): Promise<Check> {
  try {
    const health = await get<Health>(s, '/v1/health');
    const system = await get<SystemInfo>(s, '/v1/system');
    return { ok: true, health, system };
  } catch (error) {
    return { ok: false, error };
  }
}

// served by the server itself (/studio/): that server is the natural first one
const sameOrigin = () => (window.location.pathname.startsWith('/studio') ? window.location.origin : '');
const blank = (token = ''): Server => ({ id: uid(), name: '', base: sameOrigin(), token });

const scrollTo = (id: string) => setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 60);

/**
 * create comes from #/servers?create=1 (StudioApiNote); activeMode is what the
 * active server's health said it is, for servers this page has not tested yet.
 */
export function Servers({ create, activeMode }: { create?: boolean; activeMode: Health['mode'] | null }) {
  const { servers, setServers, active, setActiveId } = useStudio();
  const [draft, setDraft] = useState<Server | null>(servers.length ? null : blank());
  const [checks, setChecks] = useState<Record<string, Check | 'pending'>>({});
  const [creating, setCreating] = useState(false);
  const [guide, setGuide] = useState<{ kind?: ProviderKind } | null>(null);
  const [finding, setFinding] = useState(false);
  // the active server has a provider key: only then can it find GPU servers
  const [canFind, setCanFind] = useState(false);
  const [toast, setToast] = useState<Server | null>(null);
  const tracked = useMachines(servers);

  useEffect(() => {
    let live = true;
    setCanFind(false);
    if (active)
      providers(active).then(
        list => live && setCanFind(list.some(p => p.configured)),
        () => {},
      );
    return () => {
      live = false;
    };
  }, [active?.id, active?.base, active?.token]); // eslint-disable-line react-hooks/exhaustive-deps

  // open the dialog once, then drop ?create=1 so a reload does not reopen it
  useEffect(() => {
    if (!create) return;
    setCreating(true);
    window.location.replace(href({ page: 'servers' }));
  }, [create]);

  const modeOf = (s: Server): Health['mode'] | null => {
    const c = checks[s.id];
    if (c && c !== 'pending' && c.ok) return c.health.mode ?? 'server';
    return s.id === active?.id ? activeMode : null;
  };
  const onlyStudioApi = servers.length === 1 && modeOf(servers[0]) === 'studio-api';

  const test = async (s: Server) => {
    setChecks(c => ({ ...c, [s.id]: 'pending' }));
    const r = await check(s);
    setChecks(c => ({ ...c, [s.id]: r }));
  };

  const saveDraft = () => {
    if (!draft) return;
    const s = { ...draft, name: draft.name.trim() || draft.base, base: draft.base.trim().replace(/\/+$/, '') };
    setServers(list => (list.some(x => x.id === s.id) ? list.map(x => (x.id === s.id ? s : x)) : [...list, s]));
    if (!active) setActiveId(s.id);
    setDraft(null);
    test(s);
  };

  const addLinked = ({ machine, owner }: Tracked) =>
    autoLink(owner, machine).then(s => {
      if (!s) return;
      setServers(list => (list.some(x => x.machine?.id === machine.id) ? list : [...list, s]));
      test(s);
      setToast(s);
    });

  // a ready machine becomes an ordinary server, once per browser; a linked one
  // that came back at a new address (Vast after Stop and Start) follows it
  useEffect(() => {
    for (const t of tracked.items) {
      const cur = servers.find(x => x.machine?.id === t.machine.id);
      if (!cur) {
        if (canAutoLink(t.machine)) addLinked(t);
        continue;
      }
      const next = relinked(cur, t.machine, t.owner);
      if (!next) continue;
      setServers(list => list.map(x => (x.id === next.id ? next : x)));
      test(next);
    }
  }, [tracked.items]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 10_000);
    return () => clearTimeout(t);
  }, [toast]);

  const openForm = (token = '') => {
    setGuide(null);
    setDraft(blank(token));
    scrollTo('server-form');
  };

  return (
    <>
      <PageHeader
        title="Servers"
        description={
          'Each server is the ai-toolkit Go image running somewhere. Add one you already run, or create a GPU server on RunPod, Vast.ai or QuickPod.' +
          (onlyStudioApi ? ' Create your first GPU server through it.' : '')
        }
        actions={
          <>
            <button className="btn btn-ghost" onClick={() => setGuide({})}>
              <BookOpen />
              Set up a template yourself
            </button>
            {!draft && (
              <button className="btn" onClick={() => openForm()}>
                <Plus />
                Add server
              </button>
            )}
            <button className="btn btn-primary" onClick={() => setCreating(true)}>
              <Cpu />
              Create a GPU server
            </button>
          </>
        }
      />
      {servers.length > 0 && (
        <ul className="server-list">
          {servers.map(s => {
            const c = checks[s.id];
            const inUse = s.id === active?.id;
            const via = s.machine?.via === 'relay' ? servers.find(x => x.id === s.machine?.control) : undefined;
            return (
              <li key={s.id} className="card server">
                <div className="server-main">
                  <p className="server-name">
                    {s.name}
                    {inUse && <span className="badge tone-accent">In use</span>}
                    {modeOf(s) === 'studio-api' && <span className="badge tone-quiet">Studio API</span>}
                    {s.machine && <span className="badge tone-quiet">{providerNames[s.machine.provider] ?? s.machine.provider}</span>}
                  </p>
                  <p className="mono server-base">{s.base}</p>
                  {s.machine?.via === 'relay' && <p className="quiet-note">Through the relay on {via?.name ?? 'the server that created it'}</p>}
                  {c === 'pending' && (
                    <p className="server-check conn conn-checking">
                      <span className="dot" aria-hidden />
                      Testing
                    </p>
                  )}
                  {c && c !== 'pending' && c.ok && (
                    <p className="server-check conn conn-up">
                      <span className="dot" aria-hidden />
                      Reachable{c.health.upstream_version ? ` · ai-toolkit ${c.health.upstream_version}` : ''} ·{' '}
                      {c.system.gpus.length ? c.system.gpus.map(g => g.name).join(', ') : 'no GPU reported'}
                    </p>
                  )}
                  {c && c !== 'pending' && !c.ok && <ErrorNote error={c.error} />}
                </div>
                <div className="server-actions">
                  {!inUse && (
                    <button className="btn" onClick={() => setActiveId(s.id)}>
                      Use
                    </button>
                  )}
                  <button className="btn" onClick={() => test(s)}>
                    Test
                  </button>
                  <button className="btn" onClick={() => setDraft({ ...s })}>
                    Edit
                  </button>
                  <button
                    className="btn btn-ghost btn-danger"
                    onClick={() => {
                      const billing = mayBill(tracked.items, s.id);
                      const msg =
                        `Remove ${s.name} from the studio? The server itself is untouched.` +
                        (billing
                          ? ` ${billing} GPU server${billing > 1 ? 's' : ''} it tracks keep${billing > 1 ? '' : 's'} billing at the provider and will no longer show here until you add ${s.name} again.`
                          : '');
                      if (confirm(msg)) setServers(list => list.filter(x => x.id !== s.id));
                    }}
                  >
                    Remove
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {(tracked.items.length > 0 || (active && canFind)) && (
        <Section
          title="GPU servers"
          description={
            tracked.items.length > 0
              ? 'Rented through your servers. They bill at the provider until you terminate them.'
              : 'None tracked yet. If a server lost track of one, find it on your provider account.'
          }
          actions={
            active &&
            canFind && (
              <button className="btn btn-ghost" aria-expanded={finding} onClick={() => setFinding(f => !f)}>
                <Search />
                Find GPU servers on your account
              </button>
            )
          }
        >
          <div className="server-list">
            {finding && active && canFind && (
              <FindMachines
                key={active.id}
                owner={active}
                onAdopted={(m, owner) => {
                  tracked.put(owner.id, m);
                  scrollTo(`machine-${m.id}`);
                }}
              />
            )}
            {tracked.items.length > 0 && (
              <ul className="server-list">
                {tracked.items.map(t => {
                  const linked = servers.find(x => x.machine?.id === t.machine.id) ?? null;
                  return (
                    <MachineRow
                      key={t.machine.id}
                      t={t}
                      linked={linked}
                      inUse={!!linked && linked.id === active?.id}
                      onUse={() => linked && setActiveId(linked.id)}
                      onChanged={m => tracked.put(t.owner.id, m)}
                      onTerminated={m => {
                        tracked.put(t.owner.id, m);
                        // the server entry goes with the machine
                        setServers(list => list.filter(x => x.machine?.id !== m.id));
                      }}
                      onForgotten={() => tracked.drop(t.machine.id)}
                      onLink={() => {
                        clearLinked(t.machine.id);
                        addLinked(t);
                      }}
                    />
                  );
                })}
              </ul>
            )}
          </div>
        </Section>
      )}

      {draft && (
        <Section title={servers.some(s => s.id === draft.id) ? 'Edit server' : 'New server'}>
          <form
            id="server-form"
            className="card form-grid server-form"
            onSubmit={e => {
              e.preventDefault();
              saveDraft();
            }}
          >
            <label className="field">
              <span>Name</span>
              <input value={draft.name} placeholder="RunPod 4090" onChange={e => setDraft({ ...draft, name: e.target.value })} />
            </label>
            <label className="field field-wide">
              <span>Base URL</span>
              <input
                required
                value={draft.base}
                placeholder="https://<pod>-8675.proxy.runpod.net"
                onChange={e => setDraft({ ...draft, base: e.target.value })}
              />
              <small>
                Cross-origin servers need this page's origin in <code>AITK_CORS_ORIGINS</code>. In <code>npm run dev</code>, use{' '}
                <button type="button" className="link" onClick={() => setDraft({ ...draft, base: '/aitk' })}>
                  /aitk
                </button>{' '}
                to go through the dev proxy instead.
              </small>
            </label>
            <label className="field field-wide">
              <span>Token</span>
              <input
                type="password"
                autoComplete="off"
                value={draft.token}
                placeholder="AI_TOOLKIT_AUTH or an aitk_… API key"
                onChange={e => setDraft({ ...draft, token: e.target.value })}
              />
            </label>
            <div className="form-actions">
              {servers.length > 0 && (
                <button className="btn btn-ghost" type="button" onClick={() => setDraft(null)}>
                  Cancel
                </button>
              )}
              <button className="btn btn-primary" type="submit">
                Save and test
              </button>
            </div>
          </form>
        </Section>
      )}

      {creating && (
        <CreateGpuServer
          owner={active}
          onClose={() => setCreating(false)}
          onCreated={(m, owner) => {
            tracked.put(owner.id, m);
            scrollTo(`machine-${m.id}`);
          }}
          onGuide={kind => {
            setCreating(false);
            setGuide({ kind });
          }}
          onAddServer={() => {
            setCreating(false);
            openForm();
          }}
        />
      )}
      {guide && <ProviderGuide owner={active} initial={guide.kind} onClose={() => setGuide(null)} onAddServer={openForm} />}
      {toast && (
        <div className="toast" role="status">
          <span>
            Added <strong>{toast.name}</strong> to your servers
          </span>
          <button
            className="btn btn-sm"
            onClick={() => {
              setActiveId(toast.id);
              setToast(null);
            }}
          >
            Use it
          </button>
        </div>
      )}
    </>
  );
}

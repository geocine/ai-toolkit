import { Plus } from 'lucide-react';
import { useState } from 'react';
import { ErrorNote, PageHeader, Section } from '../components/ui';
import { get, type Health, type Server, type SystemInfo } from '../lib/api';
import { useStudio } from '../lib/context';
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

const blank = (): Server => ({ id: uid(), name: '', base: '', token: '' });

export function Servers() {
  const { servers, setServers, active, setActiveId } = useStudio();
  const [draft, setDraft] = useState<Server | null>(servers.length ? null : blank());
  const [checks, setChecks] = useState<Record<string, Check | 'pending'>>({});

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

  return (
    <>
      <PageHeader
        title="Servers"
        description="Each server is the ai-toolkit Go image running somewhere: RunPod, Vast, your own box. The URL and token stay in this browser."
        actions={
          !draft && (
            <button className="btn btn-primary" onClick={() => setDraft(blank())}>
              <Plus />
              Add server
            </button>
          )
        }
      />
      {servers.length > 0 && (
        <ul className="server-list">
          {servers.map(s => {
            const c = checks[s.id];
            const inUse = s.id === active?.id;
            return (
              <li key={s.id} className="card server">
                <div className="server-main">
                  <p className="server-name">
                    {s.name}
                    {inUse && <span className="badge tone-accent">In use</span>}
                  </p>
                  <p className="mono server-base">{s.base}</p>
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
                      if (confirm(`Remove ${s.name} from the studio? The server itself is untouched.`))
                        setServers(list => list.filter(x => x.id !== s.id));
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

      {draft && (
        <Section title={servers.some(s => s.id === draft.id) ? 'Edit server' : 'New server'}>
          <form
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
    </>
  );
}

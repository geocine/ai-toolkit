// Placeholder UI: proves the Go server serves a frontend and its API answers.
// Replace this folder with the real UI; keep `npm run build` emitting dist/.
import { useCallback, useEffect, useState } from 'react';
import { api, getToken, setToken, Unauthorized, type GpuInfo, type Job } from './api';

type State =
  | { kind: 'loading' }
  | { kind: 'auth' }
  | { kind: 'error'; message: string }
  | { kind: 'ready'; jobs: Job[]; gpu: GpuInfo | null };

export function App() {
  const [state, setState] = useState<State>({ kind: 'loading' });
  const [token, setTokenInput] = useState(getToken());

  const load = useCallback(async () => {
    try {
      await api('/api/auth');
      const [{ jobs }, gpu] = await Promise.all([
        api<{ jobs: Job[] }>('/api/jobs'),
        api<GpuInfo>('/api/gpu').catch(() => null),
      ]);
      setState({ kind: 'ready', jobs, gpu });
    } catch (e) {
      if (e instanceof Unauthorized) setState({ kind: 'auth' });
      else setState({ kind: 'error', message: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [load]);

  return (
    <main>
      <header>
        <h1>AI Toolkit</h1>
        <span className="badge">placeholder UI</span>
      </header>

      {state.kind === 'loading' && <p className="muted">Connecting to the API…</p>}

      {state.kind === 'error' && <p className="error">API error: {state.message}</p>}

      {state.kind === 'auth' && (
        <form
          onSubmit={e => {
            e.preventDefault();
            setToken(token);
            load();
          }}
        >
          <label>
            Access token (AI_TOOLKIT_AUTH)
            <input type="password" value={token} onChange={e => setTokenInput(e.target.value)} autoFocus />
          </label>
          <button type="submit">Sign in</button>
        </form>
      )}

      {state.kind === 'ready' && (
        <>
          <section>
            <h2>GPUs</h2>
            {state.gpu?.gpus?.length ? (
              <ul>
                {state.gpu.gpus.map(g => (
                  <li key={g.index}>
                    #{g.index} {g.name} · {g.utilization.gpu}% · {Math.round(g.memory.used / 1024)} /{' '}
                    {Math.round(g.memory.total / 1024)} GB
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted">No NVIDIA GPUs reported.</p>
            )}
          </section>
          <section>
            <h2>Jobs</h2>
            {state.jobs.length ? (
              <table>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Status</th>
                    <th>Step</th>
                    <th>GPU</th>
                    <th>Info</th>
                  </tr>
                </thead>
                <tbody>
                  {state.jobs.map(j => (
                    <tr key={j.id}>
                      <td>{j.name}</td>
                      <td>
                        <span className={`status ${j.status}`}>{j.status}</span>
                      </td>
                      <td>{j.step}</td>
                      <td>{j.gpu_ids}</td>
                      <td className="info">{j.info}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">No jobs yet.</p>
            )}
          </section>
        </>
      )}
    </main>
  );
}

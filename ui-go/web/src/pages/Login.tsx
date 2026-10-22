import { useState } from 'react';
import { Boxes, LogIn } from 'lucide-react';
import { ApiError, api, setToken } from '../api';
import type { Health } from '../types';
import { Callout, Field, Spinner, ThemeToggle } from '../ui';

export function Login({ health, onIn }: { health: Health | null; onIn: () => void }) {
  const [token, setTok] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    setToken(token.trim());
    try {
      await api.get('/v1/jobs?state=running');
      onIn();
    } catch (err) {
      setToken('');
      setError(err instanceof ApiError && err.status === 401 ? 'That token was not accepted.' : `Cannot reach the server: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <form className="login-card" onSubmit={submit}>
        <div className="brand-mark">
          <Boxes size={14} />
        </div>
        <h1>Sign in to the console</h1>
        <div className="login-sub">Use the server token or an API key.</div>
        <Field label="Access token" hint={<>AI_TOOLKIT_AUTH or an API key (<span className="mono">aitk_…</span>)</>}>
          <input type="password" value={token} onChange={e => setTok(e.target.value)} autoFocus autoComplete="current-password" placeholder="••••••••" />
        </Field>
        {error && (
          <div style={{ marginTop: 12 }}>
            <Callout tone="danger">{error}</Callout>
          </div>
        )}
        <button className="btn btn-primary btn-block" style={{ marginTop: 20 }} disabled={busy || !token.trim()}>
          {busy ? <Spinner size={14} /> : <LogIn size={14} />} Sign in
        </button>
        <div className="login-foot">
          <span className="server-status">
            <span className="dot" style={{ ['--c' as string]: health?.ok ? 'var(--ok)' : 'var(--text-3)' }} />
            {health ? (
              <span>
                Server <span className="mono xs">{health.version}</span>
                {health.upstream_version && <> · toolkit {health.upstream_version}</>}
              </span>
            ) : (
              'Server not reachable'
            )}
          </span>
          <ThemeToggle />
        </div>
      </form>
    </div>
  );
}

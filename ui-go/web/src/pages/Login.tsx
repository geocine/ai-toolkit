import { useState } from 'react';
import { Boxes, LogIn } from 'lucide-react';
import { ApiError, api, setToken } from '../api';
import type { Health } from '../types';
import { Field, Spinner } from '../ui';

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
        <h1>AI Toolkit Console</h1>
        <div className="muted small" style={{ marginBottom: 20 }}>
          {health ? (
            <>
              Server <span className="mono">{health.version}</span>{health.upstream_version && <> · toolkit <span className="mono">{health.upstream_version}</span></>}
            </>
          ) : (
            'Sign in with the server token or an API key.'
          )}
        </div>
        <Field label="Access token" hint="AI_TOOLKIT_AUTH or an API key (aitk_…)">
          <input type="password" value={token} onChange={e => setTok(e.target.value)} autoFocus autoComplete="current-password" placeholder="••••••••" />
        </Field>
        {error && (
          <div className="danger-text small" style={{ marginTop: 12 }}>
            {error}
          </div>
        )}
        <button className="btn btn-primary" style={{ width: '100%', marginTop: 20 }} disabled={busy || !token.trim()}>
          {busy ? <Spinner size={12} /> : <LogIn size={12} />} Sign in
        </button>
      </form>
    </div>
  );
}

import { KeyRound, RefreshCw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { ApiError, type Server } from '../lib/api';
import { perHour, pillState, providerNames, stateLabel } from '../lib/machines';
import { adopt, instances, providers } from '../lib/provision';
import type { Machine, Provider, ProviderInstance, ProviderKind } from '../lib/provisionTypes';
import { ErrorNote, StatePill } from './ui';

/** "RTX 4090 × 2 · $0.44/h · created Oct 24" */
function meta(i: ProviderInstance): string {
  const parts = [i.gpu ? `${i.gpu}${i.gpu_count > 1 ? ` × ${i.gpu_count}` : ''}` : ''];
  if (i.price_per_hour != null && i.status !== 'stopped') parts.push(perHour(i.price_per_hour));
  if (i.created_at) parts.push(`created ${new Date(i.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`);
  return parts.filter(Boolean).join(' · ');
}

/**
 * The aitk-labelled servers on a provider account, as owner sees them, and
 * tracking one again that no server tracks (PLAN §12): after the server that
 * created it was lost, or it was forgotten while still running.
 */
export function FindMachines({ owner, onAdopted }: { owner: Server; onAdopted: (m: Machine, owner: Server) => void }) {
  const [provs, setProvs] = useState<Provider[] | null>(null);
  const [kind, setKind] = useState<ProviderKind | null>(null);
  const [items, setItems] = useState<ProviderInstance[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let live = true;
    setProvs(null);
    setError(null);
    providers(owner).then(
      list => {
        if (!live) return;
        const ready = list.filter(p => p.configured);
        setProvs(ready);
        setKind(k => (k && ready.some(p => p.kind === k) ? k : (ready[0]?.kind ?? null)));
      },
      e => live && setError(e),
    );
    return () => {
      live = false;
    };
  }, [owner.id, owner.base, owner.token]); // eslint-disable-line react-hooks/exhaustive-deps

  // only the latest search may fill the list: switching providers quickly
  // must not show the previous provider's servers under the new tab
  const seq = useRef(0);
  const search = async (k: ProviderKind) => {
    const n = ++seq.current;
    setLoading(true);
    setError(null);
    try {
      const list = await instances(owner, k);
      if (n === seq.current) setItems(list);
    } catch (e) {
      if (n !== seq.current) return;
      setItems(null);
      setError(e);
    } finally {
      if (n === seq.current) setLoading(false);
    }
  };
  useEffect(() => {
    setItems(null);
    if (kind) search(kind);
  }, [kind]); // eslint-disable-line react-hooks/exhaustive-deps

  const label = (provs?.find(p => p.kind === kind)?.label ?? (kind && providerNames[kind])) || '';

  if (provs && !provs.length)
    return <p className="quiet-note">{owner.name} has no provider key yet. Add one under Create a GPU server, then search here.</p>;

  return (
    <div className="server-list">
      <div className="row">
        {provs && provs.length > 1 && (
          <div className="seg" role="radiogroup" aria-label="Provider">
            {provs.map(p => (
              <button key={p.kind} type="button" role="radio" aria-checked={p.kind === kind} className={p.kind === kind ? 'is-on' : ''} onClick={() => setKind(p.kind)}>
                {p.label}
              </button>
            ))}
          </div>
        )}
        <button className="btn btn-sm" onClick={() => kind && search(kind)} disabled={!kind || loading}>
          <RefreshCw />
          {loading ? 'Searching' : 'Search again'}
        </button>
      </div>
      {kind && (
        <p className="quiet-note">
          Servers with an <span className="mono">aitk-</span> label on your {label} account, as {owner.name} sees them. One that is not tracked may still be
          billing: track it here, or terminate it in the provider console.
        </p>
      )}
      <ErrorNote error={error} />
      {items && !items.length && <p className="muted">No aitk servers on your {label} account.</p>}
      {items && items.length > 0 && kind && (
        <ul className="server-list">
          {items.map(i => (
            <InstanceRow
              key={i.provider_id}
              item={i}
              kind={kind}
              owner={owner}
              onAdopted={m => {
                setItems(list => list?.map(x => (x.provider_id === i.provider_id ? { ...x, machine_id: m.id } : x)) ?? null);
                onAdopted(m, owner);
              }}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function InstanceRow({
  item: i,
  kind,
  owner,
  onAdopted,
}: {
  item: ProviderInstance;
  kind: ProviderKind;
  owner: Server;
  onAdopted: (m: Machine) => void;
}) {
  // QuickPod never echoes the env, and an echo can be a RunPod secret reference
  const [asking, setAsking] = useState(false);
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const track = async (withToken?: string) => {
    setBusy(true);
    setError(null);
    try {
      onAdopted(await adopt(owner, kind, { provider_id: i.provider_id, ...(withToken ? { token: withToken } : {}) }));
      setAsking(false);
      setToken('');
    } catch (e) {
      // the token was not where the list said: ask for it instead of showing the refusal
      if (!withToken && e instanceof ApiError && e.code === 'invalid_request') setAsking(true);
      else setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <li className="card server">
      <div className="server-main">
        <p className="server-name">
          <span className="mono machine-name">{i.label}</span>
          <StatePill state={pillState(i.status)} label={i.status === 'running' ? 'Running' : stateLabel(i.status)} />
        </p>
        {meta(i) && <p className="machine-meta">{meta(i)}</p>}
        <p className="quiet-note machine-via">
          {i.machine_id ? (
            <>
              Tracked on {owner.name} as <span className="mono">{i.machine_id}</span>
            </>
          ) : (
            'Not tracked by this server'
          )}
        </p>
        {asking && (
          <form
            className="machine-token"
            onSubmit={e => {
              e.preventDefault();
              if (token.trim()) track(token.trim());
            }}
          >
            <label className="field">
              <span>Its token (AI_TOOLKIT_AUTH)</span>
              <input type="password" autoComplete="off" value={token} onChange={e => setToken(e.target.value)} />
            </label>
            <button className="btn btn-sm" type="submit" disabled={busy || !token.trim()}>
              <KeyRound />
              Track with token
            </button>
          </form>
        )}
        <ErrorNote error={error} />
      </div>
      {!i.machine_id && !asking && (
        <div className="server-actions">
          <button className="btn btn-sm btn-primary" disabled={busy} onClick={() => (i.token_known ? track() : setAsking(true))}>
            Track here
          </button>
        </div>
      )}
    </li>
  );
}

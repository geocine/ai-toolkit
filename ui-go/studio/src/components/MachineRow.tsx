import { Ellipsis, ExternalLink, KeyRound, Link2, Play, ScrollText, Square, Trash2 } from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { Server } from '../lib/api';
import { copyText, perHour, pillState, providerNames, stateLabel, type Tracked } from '../lib/machines';
import { forget, link, logs, patchMachine, start, stop, terminate } from '../lib/provision';
import type { Machine, MachineLogs } from '../lib/provisionTypes';
import { Confirm, Dialog } from './Dialog';
import { Callout, ErrorNote, StatePill } from './ui';

/** "RTX 4090 · 24 GB · $0.47/h", or what a stopped machine still costs. */
function meta(t: Tracked): string {
  const m = t.machine;
  const gpu = [m.gpu ? `${m.gpu}${m.gpu_count > 1 ? ` × ${m.gpu_count}` : ''}` : '', m.vram_gb ? `${m.vram_gb} GB` : ''].filter(Boolean);
  if (m.status === 'stopped') {
    if (m.stopped_price_per_hour) return `Stopped · still ${perHour(m.stopped_price_per_hour)} for disk`;
    const wipes = t.provider ? !t.provider.capabilities.stop_keeps_disk : m.provider === 'runpod';
    return wipes ? 'Stopped · nothing bills (disk was wiped)' : 'Stopped';
  }
  if (m.price_per_hour != null && !['terminated', 'failed'].includes(m.status)) gpu.push(perHour(m.price_per_hour));
  return gpu.join(' · ');
}

/** One GPU server tracked by the server that holds its provider key: state, problems and actions (PLAN §11). */
export function MachineRow({
  t,
  linked,
  inUse,
  onUse,
  onChanged,
  onTerminated,
  onForgotten,
  onLink,
}: {
  t: Tracked;
  linked: Server | null;
  inUse: boolean;
  onUse: () => void;
  onChanged: (m: Machine) => void;
  onTerminated: (m: Machine) => void;
  onForgotten: () => void;
  /** Adds this ready machine to the saved servers again, after its server was removed by hand. */
  onLink: () => void;
}) {
  const m = t.machine;
  const s = t.owner;
  const label = t.provider?.label ?? providerNames[m.provider];
  const consoleUrl = t.provider?.console_url ?? '';
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [ask, setAsk] = useState<'terminate' | 'forget' | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [token, setToken] = useState('');
  const [copied, setCopied] = useState<string | null>(null);

  const act = async (fn: () => Promise<Machine | void>, after?: (m: Machine) => void) => {
    setBusy(true);
    setError(null);
    try {
      const r = await fn();
      if (r) (after ?? onChanged)(r);
      return true;
    } catch (e) {
      setError(e);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const copyToken = async () => {
    setError(null);
    try {
      const l = await link(s, m.id);
      if (await copyText(l.token)) setCopied('Token copied');
      else window.prompt('Token for this server', l.token);
    } catch (e) {
      setError(e);
    }
  };
  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(null), 2000);
    return () => clearTimeout(t);
  }, [copied]);

  const can = (a: Machine['actions'][number]) => m.actions.includes(a);
  // the server's forget rule: refused unless ended, or refused before anything was rented
  // (a never_listed create may have been made, so it needs force)
  const live = !m.ended_at && !(m.provider_id === '' && m.status === 'failed' && m.problem !== 'never_listed');
  const plainHttp = !m.simulated && !!m.base_url && m.base_url.startsWith('http:');
  const terminateBtn = can('terminate') && (
    <button className="btn btn-sm btn-danger" onClick={() => setAsk('terminate')} disabled={busy}>
      {m.status === 'terminating' ? 'Terminate again' : 'Terminate'}
    </button>
  );
  const consoleBtn = consoleUrl && (
    <a className="btn btn-sm" href={consoleUrl} target="_blank" rel="noreferrer">
      <ExternalLink />
      Open provider console
    </a>
  );

  let problem = null;
  switch (m.problem) {
    case null:
    case 'not_refreshed':
      break;
    case 'open_server':
      problem = (
        <Callout kind="danger">
          <p>This server answers without a token. Terminate it or fix the template.</p>
          {terminateBtn && <div className="machine-callout-actions">{terminateBtn}</div>}
        </Callout>
      );
      break;
    case 'token_mismatch':
      problem = (
        <Callout kind="warn">
          <p>{m.detail || 'The template set its own AI_TOOLKIT_AUTH. Paste that token to link it.'}</p>
          <form
            className="machine-token"
            onSubmit={async e => {
              e.preventDefault();
              if (token.trim() && (await act(() => patchMachine(s, m.id, { token: token.trim() })))) setToken('');
            }}
          >
            <label className="field">
              <span>Token from your template</span>
              <input type="password" autoComplete="off" value={token} onChange={e => setToken(e.target.value)} />
            </label>
            <button className="btn btn-sm" type="submit" disabled={busy || !token.trim()}>
              <KeyRound />
              Save token
            </button>
          </form>
        </Callout>
      );
      break;
    case 'no_port':
    case 'not_ready_timeout':
    case 'never_listed':
      problem = (
        <Callout kind="warn">
          <p>{m.detail}</p>
          <div className="machine-callout-actions">
            {terminateBtn}
            {consoleBtn}
          </div>
        </Callout>
      );
      break;
    case 'provider_error':
      problem = <Callout kind="danger">{m.detail}</Callout>;
      break;
    default:
      problem = m.detail ? <Callout kind="warn">{m.detail}</Callout> : null;
  }

  return (
    <li className="card server machine" id={`machine-${m.id}`}>
      <div className="server-main">
        <p className="server-name">
          <span className="machine-name">{m.name}</span>
          <span className="badge tone-quiet">{label}</span>
          {t.provider?.stage === 'beta' && <span className="badge tone-quiet">Beta</span>}
          {m.simulated && <span className="badge tone-quiet">Simulated</span>}
          {inUse && <span className="badge tone-accent">In use</span>}
        </p>
        <p className="machine-state">
          <StatePill state={pillState(m.status)} label={stateLabel(m.status)} />
          {m.status === 'failed' && m.detail && !m.problem && <span className="muted">{m.detail}</span>}
        </p>
        {meta(t) && <p className="machine-meta">{meta(t)}</p>}
        {m.base_url && (
          <p className="server-base">
            <span className="mono">{m.base_url}</span> <span className="quiet-note">via {s.name}</span>
          </p>
        )}
        {!m.base_url && <p className="quiet-note machine-via">Via {s.name}</p>}
        {plainHttp && <p className="quiet-note machine-via">Plain http: the token is not encrypted on the way.</p>}
        {m.warnings.map(w => (
          <p key={w} className="quiet-note machine-via">
            {w}
          </p>
        ))}
        {problem}
        <ErrorNote error={error} />
      </div>
      <div className="server-actions">
        {copied && (
          <span className="quiet-note machine-copied" role="status">
            {copied}
          </span>
        )}
        {linked && !inUse && (
          <button className="btn btn-sm" onClick={onUse}>
            Use
          </button>
        )}
        {can('stop') && (
          <button className="btn btn-sm" onClick={() => act(() => stop(s, m.id))} disabled={busy}>
            <Square />
            Stop
          </button>
        )}
        {can('start') && (
          <button className="btn btn-sm" onClick={() => act(() => start(s, m.id))} disabled={busy}>
            <Play />
            Start
          </button>
        )}
        {m.problem !== 'open_server' && !['no_port', 'not_ready_timeout', 'never_listed'].includes(m.problem ?? '') && terminateBtn}
        <Menu label={`More for ${m.name}`}>
          {!linked && m.status === 'ready' && m.problem !== 'open_server' && (
            <button role="menuitem" onClick={onLink}>
              <Link2 />
              Add to your servers
            </button>
          )}
          {m.status === 'ready' && m.problem !== 'open_server' && (
            <button role="menuitem" onClick={copyToken}>
              <KeyRound />
              Copy token
            </button>
          )}
          <button role="menuitem" onClick={() => setShowLogs(true)}>
            <ScrollText />
            Logs
          </button>
          {consoleUrl && (
            <a role="menuitem" href={consoleUrl} target="_blank" rel="noreferrer">
              <ExternalLink />
              Open provider console
            </a>
          )}
          {can('forget') && (
            <button role="menuitem" className="is-danger" onClick={() => setAsk('forget')}>
              <Trash2 />
              Forget
            </button>
          )}
        </Menu>
      </div>

      {ask === 'terminate' && (
        <Confirm
          title="Terminate GPU server"
          confirm="Terminate"
          busy={busy}
          onClose={() => setAsk(null)}
          onConfirm={async () => {
            if (await act(() => terminate(s, m.id, m.self), onTerminated)) setAsk(null);
          }}
        >
          <p>
            Terminate <strong>{m.name}</strong>? The machine and its disk are deleted at {label}. Results not synced to a remote are lost.
          </p>
          {m.self && <p className="machine-self">This is the server you are using right now.</p>}
          <ErrorNote error={error} />
        </Confirm>
      )}
      {ask === 'forget' && (
        <Confirm
          title={live ? 'Forget a live GPU server' : 'Forget GPU server'}
          confirm={live ? 'Forget anyway' : 'Forget'}
          busy={busy}
          onClose={() => setAsk(null)}
          onConfirm={async () => {
            if (await act(() => forget(s, m.id, live))) {
              setAsk(null);
              onForgotten();
            }
          }}
        >
          {live && m.problem === 'never_listed' ? (
            <p>
              <strong>{m.name}</strong> is removed from this list. If {label} made it after all, it keeps running and billing there and this server stops
              watching for it. Check the provider console first.
            </p>
          ) : live ? (
            <p>
              <strong>{m.name}</strong> is removed from this list, but it keeps running and billing at {label}. Terminate it first unless you will manage it in
              the provider console.
            </p>
          ) : (
            <p>
              Remove <strong>{m.name}</strong> from this list?
            </p>
          )}
          <ErrorNote error={error} />
        </Confirm>
      )}
      {showLogs && <LogsDialog server={s} machine={m} onClose={() => setShowLogs(false)} />}
    </li>
  );
}

function LogsDialog({ server, machine, onClose }: { server: Server; machine: Machine; onClose: () => void }) {
  const [data, setData] = useState<MachineLogs | null>(null);
  const [error, setError] = useState<unknown>(null);
  const read = () => {
    setError(null);
    logs(server, machine.id, 500).then(setData, setError);
  };
  useEffect(read, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Dialog
      wide
      title={`Logs of ${machine.name}`}
      description={data ? `From ${providerNames[data.source as keyof typeof providerNames] ?? data.source}${data.truncated ? ', the last part only' : ''}` : undefined}
      onClose={onClose}
      footer={
        <button className="btn" onClick={read}>
          Refresh
        </button>
      }
    >
      <ErrorNote error={error} />
      {!error && <pre className="log">{data ? data.text || 'Nothing logged yet.' : 'Loading…'}</pre>}
    </Dialog>
  );
}

/** A small "…" menu; Escape or a click outside closes it. */
function Menu({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);
  return (
    <div className="menu" ref={box}>
      <button className="btn btn-sm btn-ghost btn-icon" aria-label={label} title="More" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(o => !o)}>
        <Ellipsis />
      </button>
      {open && (
        <div className="menu-pop" role="menu" onClick={() => setOpen(false)}>
          {children}
        </div>
      )}
    </div>
  );
}

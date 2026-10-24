import { Check, Copy, ExternalLink, RefreshCw } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { Server } from '../lib/api';
import { fillGuide, guideFor, newToken, validateGuideOptions } from '../lib/guides';
import { copyText, providerNames } from '../lib/machines';
import { setup } from '../lib/provision';
import type { ProviderKind, SetupGuide } from '../lib/provisionTypes';
import { Dialog } from './Dialog';

const kinds: ProviderKind[] = ['runpod', 'vast', 'quickpod'];

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  useEffect(() => {
    if (!done) return;
    const t = setTimeout(() => setDone(false), 1500);
    return () => clearTimeout(t);
  }, [done]);
  return (
    <button
      type="button"
      className="btn btn-sm"
      onClick={async () => {
        if (await copyText(text)) setDone(true);
        else window.prompt('Copy this', text);
      }}
    >
      {done ? <Check /> : <Copy />}
      {done ? 'Copied' : label}
    </button>
  );
}

/**
 * "Set up a template yourself": the manual steps per provider. It works with
 * no server at all (studio/src/lib/guides.json); with a controller the same
 * guide comes from providers.setup, the source MCP agents read.
 */
export function ProviderGuide({
  controller,
  initial = 'runpod',
  onClose,
  onAddServer,
}: {
  controller: Server | null;
  initial?: ProviderKind;
  onClose: () => void;
  onAddServer: (token: string) => void;
}) {
  const [kind, setKind] = useState<ProviderKind>(initial);
  const [token, setToken] = useState(newToken);
  const [origin, setOrigin] = useState(() => window.location.origin);
  const [served, setServed] = useState<Partial<Record<ProviderKind, SetupGuide | 'none'>>>({});

  useEffect(() => {
    if (!controller || served[kind]) return;
    let cancelled = false;
    setup(controller, kind).then(
      g => !cancelled && setServed(v => ({ ...v, [kind]: g })),
      () => !cancelled && setServed(v => ({ ...v, [kind]: 'none' })), // older server: the bundled copy
    );
    return () => {
      cancelled = true;
    };
  }, [controller?.id, kind]); // eslint-disable-line react-hooks/exhaustive-deps

  const fromServer = served[kind];
  // fillGuide checks the token and origin the same way the server does, so a
  // bad value keeps the placeholder instead of reaching the copied text
  const g: SetupGuide = fromServer && fromServer !== 'none' ? fillGuide(fromServer, { token, origin }) : guideFor(kind, { token, origin });
  const errs = validateGuideOptions({ token, origin });
  const block = g.env_block || g.docker_options;

  return (
    <Dialog
      wide
      title="Set up a template yourself"
      description="Start the aitk image at a provider by hand, then add it here as a server."
      onClose={onClose}
      footer={
        <>
          <a className="btn btn-ghost" href={g.console_url} target="_blank" rel="noreferrer">
            <ExternalLink />
            Open {providerNames[kind]}
          </a>
          <button className="btn btn-primary" disabled={!!errs.token} onClick={() => onAddServer(token.trim())}>
            Add this server
          </button>
        </>
      }
    >
      <div className="dialog-form guide">
        <div className="tabs sink-tabs" role="tablist" aria-label="Provider">
          {kinds.map(k => (
            <button key={k} type="button" role="tab" aria-selected={kind === k} className={kind === k ? 'tab is-active' : 'tab'} onClick={() => setKind(k)}>
              {providerNames[k]}
            </button>
          ))}
        </div>

        <div className="form-grid guide-inputs">
          <label className="field field-wide">
            <span>Token</span>
            <div className="guide-row">
              <input className="mono" value={token} onChange={e => setToken(e.target.value)} aria-invalid={!!errs.token} autoComplete="off" spellCheck={false} />
              <button type="button" className="btn btn-sm" onClick={() => setToken(newToken())}>
                <RefreshCw />
                Generate
              </button>
              <CopyButton text={token} />
            </div>
            {errs.token ? <small className="field-error">{errs.token}</small> : <small>The server asks for it on every request. It stays in this page until you add the server.</small>}
          </label>
          <label className="field field-wide">
            <span>Studio origin</span>
            <input className="mono" value={origin} onChange={e => setOrigin(e.target.value)} aria-invalid={!!errs.origin} inputMode="url" autoCapitalize="off" spellCheck={false} />
            {errs.origin ? <small className="field-error">{errs.origin}</small> : <small>The server lets this page call it (AITK_CORS_ORIGINS).</small>}
          </label>
        </div>

        {g.fields.length > 0 && (
          <dl className="guide-fields">
            {g.fields.map(f => (
              <div key={f.label} className="guide-field">
                <dt>{f.label}</dt>
                <dd>
                  <span className="mono">{f.value}</span>
                  {f.note && <small>{f.note}</small>}
                </dd>
                <CopyButton text={f.value} />
              </div>
            ))}
          </dl>
        )}

        {block && (
          <div className="guide-block">
            <div className="guide-block-head">
              <span className="label">{g.env_block ? 'Environment variables' : 'Docker options'}</span>
              <CopyButton text={block} />
            </div>
            <pre className="env-block">{block}</pre>
          </div>
        )}

        <ol className="sink-steps">
          {g.steps.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ol>

        {g.url_pattern && (
          <p className="quiet-note">
            Its URL looks like <span className="mono">{g.url_pattern}</span>
          </p>
        )}
      </div>
    </Dialog>
  );
}

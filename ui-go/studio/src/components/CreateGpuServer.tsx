import { Cpu, RefreshCw } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, get, type Items, type Remote, type Server, type Settings } from '../lib/api';
import type { NotifyConfig } from '../lib/lifecycle';
import { perHour, providerNames } from '../lib/machines';
import { launch, offers as listOffers, providers, removeKey, setKey, templates as listTemplates, testKey } from '../lib/provision';
import type { Machine, Offer, OfferQuery, Provider, ProviderKind, ProviderTemplate, WhenDone } from '../lib/provisionTypes';
import { Dialog } from './Dialog';
import { Callout, Empty, ErrorNote, useLoad } from './ui';

const DEFAULT_IMAGE = 'ghcr.io/geocine/aitoolkit-go:latest';
const OUR_REPO = 'ghcr.io/geocine/aitoolkit-go';
const vramSteps = [0, 16, 24, 48, 80];
const countSteps = [1, 2, 4, 8];

const round2 = (n: number) => Math.round(n * 100) / 100;

/** The repository of an image reference, without @digest or :tag (a registry port stays). */
export function imageRepo(ref: string): string {
  const r = ref.trim().split('@')[0];
  const colon = r.lastIndexOf(':');
  return colon > r.lastIndexOf('/') ? r.slice(0, colon) : r;
}

/** The first "$N.NN/h" in a message, as a number. */
function priceIn(msg: string): number | null {
  const m = /\$(\d+(?:\.\d+)?)\/h/.exec(msg);
  return m ? Number(m[1]) : null;
}

/** 16 random bytes as hex: a request_id the server accepts (8..64 of [A-Za-z0-9_-]). */
function requestId(): string {
  const b = new Uint8Array(16);
  crypto.getRandomValues(b);
  return Array.from(b, x => x.toString(16).padStart(2, '0')).join('');
}

// The passthrough rules of internal/provision/env.go, checked as the user types.
const allowed = /^(AITK_[A-Z0-9_]{1,60}|HF_[A-Z0-9_]{1,60}|HUGGING_FACE_HUB_TOKEN|TZ)$/;
const reserved = (k: string) =>
  ['AI_TOOLKIT_AUTH', 'AITK_CORS_ORIGINS', 'AITK_STUDIO_URL', 'AITK_MACHINE_ID', 'PORT'].includes(k) ||
  k.startsWith('AITK_WHEN_DONE') ||
  k.startsWith('AITK_PROVISION_') ||
  /^AITK_.*_API_KEY$/.test(k) ||
  ['RUNPOD_', 'VAST_', 'CONTAINER_', 'QUICKPOD_', 'PUBLIC_IPADDR'].some(p => k.startsWith(p));

/** What the server adds to the environment itself (env.go BuildEnv), estimated so the limits hold for the whole launch. */
export type ServerEnv = { count: number; chars: number };

/**
 * Parses KEY=value lines. Errors name the variable and never echo its value.
 * The provider's count and length limits apply to these plus server.
 */
export function parseEnv(text: string, p: Provider | null, server: ServerEnv = { count: 0, chars: 0 }): { env: Record<string, string>; error: string | null } {
  const env: Record<string, string> = {};
  for (const [i, raw] of text.split('\n').entries()) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq < 1) return { env, error: `Line ${i + 1} is not KEY=value.` };
    const key = line.slice(0, eq).trim();
    const value = line.slice(eq + 1).trim();
    if (reserved(key)) return { env, error: `${key} is set by the studio and cannot be passed here.` };
    if (!allowed.test(key)) return { env, error: `${key} is not allowed. Use AITK_…, HF_…, HUGGING_FACE_HUB_TOKEN or TZ.` };
    const c = p?.capabilities;
    if (c?.env_value_max && value.length > c.env_value_max) return { env, error: `${key} is longer than ${p!.label} allows (${c.env_value_max} characters).` };
    if (c?.env_flag_string && /[\s"'`$\\;|&<>]/.test(value)) return { env, error: `${key} has a space or one of " ' \` $ \\ ; | & < >, which ${p!.label} cannot carry.` };
    env[key] = value;
  }
  const c = p?.capabilities;
  const n = Object.keys(env).length;
  if (c?.env_count_max && n + server.count > c.env_count_max) {
    const room = Math.max(0, c.env_count_max - server.count);
    return { env, error: `${p!.label} takes at most ${c.env_count_max} variables and the studio sets about ${server.count}, so ${room} fit here.` };
  }
  const total = Object.entries(env).reduce((t, [k, v]) => t + k.length + v.length + 1, server.chars);
  if (c?.env_total_max && total > c.env_total_max) return { env, error: `These variables are longer than ${p!.label} allows in total.` };
  return { env, error: null };
}

type LaunchProblem = { text: string; link?: string };

/**
 * Create a GPU server (PLAN §11): through the active server, pick a provider,
 * an offer and the settings, then rent it at a capped price.
 */
export function CreateGpuServer({
  controller,
  onClose,
  onCreated,
  onGuide,
  onAddServer,
}: {
  controller: Server | null;
  onClose: () => void;
  onCreated: (m: Machine, controller: Server) => void;
  onGuide: (kind?: ProviderKind) => void;
  onAddServer: () => void;
}) {
  if (!controller)
    return (
      <Dialog wide title="Create a GPU server" onClose={onClose}>
        <Empty
          icon={<Cpu />}
          action={
            <div className="row create-empty-actions">
              <button className="btn" onClick={() => onGuide()}>
                Set up a template yourself
              </button>
              <button className="btn btn-primary" onClick={onAddServer}>
                Add server
              </button>
            </div>
          }
        >
          GPU servers are created through one of your servers. Add one first, or set up your first GPU server by hand.
        </Empty>
      </Dialog>
    );
  return <CreateForm controller={controller} onClose={onClose} onCreated={onCreated} onGuide={onGuide} />;
}

function CreateForm({
  controller: s,
  onClose,
  onCreated,
  onGuide,
}: {
  controller: Server;
  onClose: () => void;
  onCreated: (m: Machine, controller: Server) => void;
  onGuide: (kind?: ProviderKind) => void;
}) {
  const provs = useLoad(s, providers, []);
  const extras = useLoad(
    s,
    async x => {
      const [remotes, settings, notify] = await Promise.all([
        get<Items<Remote>>(x, '/v1/remotes').then(r => r.items, () => [] as Remote[]),
        get<Settings & { hf_token_set?: boolean }>(x, '/v1/settings').catch(() => null),
        get<NotifyConfig>(x, '/v1/notify').catch(() => null),
      ]);
      return { remotes: remotes.filter(r => r.kind === 's3' || r.kind === 'hf'), hfToken: !!settings?.hf_token_set, sinks: notify?.sinks.length ?? 0 };
    },
    [],
  );

  const [kind, setKind] = useState<ProviderKind | null>(null);
  const list = provs.data ?? [];
  const p = list.find(x => x.kind === kind) ?? null;
  useEffect(() => {
    if (!kind && list.length) setKind((list.find(x => x.configured) ?? list[0]).kind);
  }, [list, kind]);
  const caps = p?.capabilities;

  // key
  const [keyText, setKeyText] = useState('');
  const [keyBusy, setKeyBusy] = useState(false);
  const [keyError, setKeyError] = useState<unknown>(null);
  const [changingKey, setChangingKey] = useState(false);
  const [credit, setCredit] = useState<Partial<Record<ProviderKind, number | null>>>({});
  const [keySaves, setKeySaves] = useState(0); // a replaced key reads its credit again
  const configured = !!p?.configured && !changingKey;
  useEffect(() => {
    if (!p?.configured || !p.capabilities.credit || p.kind in credit) return;
    const k = p.kind;
    testKey(s, k).then(
      t => setCredit(c => ({ ...c, [k]: t.ok ? t.credit : null })),
      () => setCredit(c => ({ ...c, [k]: null })),
    );
  }, [p?.kind, p?.configured, keySaves]); // eslint-disable-line react-hooks/exhaustive-deps

  const saveKey = async () => {
    if (!p || !keyText.trim()) return;
    setKeyBusy(true);
    setKeyError(null);
    try {
      const v = await setKey(s, p.kind, keyText.trim());
      provs.setData(list.map(x => (x.kind === v.kind ? v : x)));
      setChangingKey(false);
      setCredit(c => {
        const { [v.kind]: _, ...rest } = c;
        return rest;
      });
      setKeySaves(n => n + 1);
    } catch (e) {
      setKeyError(e instanceof ApiError && e.code === 'provider_auth' ? new Error(`${p.label} did not accept that key: ${e.message}`) : e);
    } finally {
      setKeyText(''); // never kept, never stored in the browser
      setKeyBusy(false);
    }
  };
  const dropKey = async () => {
    if (!p) return;
    setKeyError(null);
    try {
      await removeKey(s, p.kind);
      provs.reload();
    } catch (e) {
      setKeyError(e);
    }
  };

  // offers
  const [gpu, setGpu] = useState('');
  const [minVram, setMinVram] = useState(0);
  const [count, setCount] = useState(1);
  const [cloud, setCloud] = useState<'secure' | 'community'>('secure');
  const [disk, setDisk] = useState('100');
  const [offerList, setOfferList] = useState<Offer[] | null>(null);
  const [offerError, setOfferError] = useState<unknown>(null);
  const [offersLoading, setOffersLoading] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [offerId, setOfferId] = useState<string | null>(null);
  const diskGb = Number(disk) || 0;
  const q: OfferQuery = {
    gpu: gpu.trim() || undefined,
    min_vram_gb: minVram || undefined,
    gpu_count: caps?.count_choice ? count : undefined,
    cloud: caps?.cloud_choice ? cloud : undefined,
    disk_gb: diskGb >= 20 ? diskGb : undefined,
    limit: 50,
  };
  const qKey = JSON.stringify(q);
  const seq = useRef(0);
  useEffect(() => {
    if (!p || !configured) {
      setOfferList(null);
      return;
    }
    const n = ++seq.current;
    setOffersLoading(true);
    const t = setTimeout(async () => {
      try {
        const got = await listOffers(s, p.kind, q);
        if (n !== seq.current) return;
        setOfferList(got.slice(0, 50));
        setOfferError(null);
      } catch (e) {
        if (n === seq.current) setOfferError(e);
      } finally {
        if (n === seq.current) setOffersLoading(false);
      }
    }, 400);
    return () => clearTimeout(t);
  }, [s.id, p?.kind, configured, qKey, refresh]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    setOfferId(null);
    setOfferList(null); // another provider's offers never show under this one
  }, [p?.kind]);
  const offer = offerList?.find(o => o.id === offerId) ?? null;
  // a 409's new quote; it holds while the list (cached 30 s) still shows the price it replaced
  const [requote, setRequote] = useState<{ id: string; was: number; price: number } | null>(null);
  const offerPrice = !offer ? 0 : requote?.id === offer.id && requote.was === offer.price_per_hour ? requote.price : offer.price_per_hour;

  // QuickPod templates
  const [tpls, setTpls] = useState<ProviderTemplate[] | null>(null);
  const [tplError, setTplError] = useState<unknown>(null);
  const [templateId, setTemplateId] = useState('');
  useEffect(() => {
    setTpls(null);
    setTplError(null);
    if (!p || !configured || !p.capabilities.template_required) return;
    listTemplates(s, p.kind).then(
      t => {
        setTpls(t);
        setTemplateId((t.find(x => x.ours && x.publishes_port) ?? t[0])?.id ?? '');
      },
      setTplError,
    );
  }, [s.id, p?.kind, configured]); // eslint-disable-line react-hooks/exhaustive-deps
  const tpl = tpls?.find(t => t.id === templateId) ?? null;

  // settings
  const [name, setName] = useState('');
  const [image, setImage] = useState(DEFAULT_IMAGE);
  const [remote, setRemote] = useState('');
  const [copyNotify, setCopyNotify] = useState(false);
  const [copyHf, setCopyHf] = useState(false);
  const [selfKey, setSelfKey] = useState(false);
  const [chosen, setChosen] = useState<WhenDone>('keep');
  const [touched, setTouched] = useState(false);
  const [envText, setEnvText] = useState('');
  useEffect(() => setSelfKey(false), [p?.kind]);

  // When done follows D-6: Delete once a remote is picked (QuickPod: only with
  // the key), back to Keep when it is cleared, unless the user chose themselves.
  const keyDestroy = caps?.self_destroy === 'key';
  const canAct = !!remote && (kind !== 'quickpod' || selfKey);
  const auto: WhenDone = canAct ? 'destroy' : 'keep';
  const whenDone: WhenDone = !touched ? auto : chosen !== 'keep' && !canAct ? 'keep' : chosen;
  const whenHint = !remote
    ? 'Stop and Delete need a remote so results are kept.'
    : kind === 'quickpod' && !selfKey
      ? 'QuickPod can stop the pod only with your key: tick the box below.'
      : kind === 'runpod' && whenDone === 'destroy' && !selfKey
        ? 'Without your key it can only stop itself.'
        : '';

  // env.go BuildEnv: token, machine id, CORS origin and studio URL, then what the settings add
  const remoteKind = extras.data?.remotes.find(r => r.name === remote)?.kind;
  const serverCount =
    4 +
    (kind === 'vast' ? 1 : 0) +
    (remote ? (remoteKind === 'hf' ? 7 : 10) : 0) +
    (copyHf ? 1 : 0) +
    (copyNotify ? 2 : 0) +
    (whenDone !== 'keep' ? 2 : 0) +
    (keyDestroy && selfKey ? 1 : 0);
  const serverChars = 200 + 2 * window.location.origin.length + window.location.pathname.length + (remote ? 600 : 0) + (copyNotify ? 300 : 0);
  const envParsed = useMemo(() => parseEnv(envText, p, { count: serverCount, chars: serverChars }), [envText, p, serverCount, serverChars]);
  const maxDisk = offer?.max_disk_gb ?? 2000;
  const diskError = diskGb < 20 || diskGb > 2000 ? 'Between 20 and 2000 GB.' : diskGb > maxDisk ? `This offer has room for ${maxDisk} GB at most.` : null;

  // launch
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<LaunchProblem | null>(null);
  const rid = useRef(requestId()); // kept for the dialog's life, so a retried request rents once
  const label = p?.label ?? '';
  const defaultName = offer ? `${offer.gpu} on ${label}` : '';
  const price = offer ? perHour(offerPrice) : '';
  const ready = !!p && configured && !!offer && !diskError && !envParsed.error && (!caps?.template_required || !!templateId) && !!image.trim();

  const sending = useRef(false);
  const create = async () => {
    if (!p || !offer || !ready || sending.current) return;
    sending.current = true;
    setBusy(true);
    setProblem(null);
    try {
      const m = await launch(s, {
        provider: p.kind,
        offer_id: offer.id,
        max_price_per_hour: round2(offerPrice * 1.02),
        request_id: rid.current,
        name: name.trim() || undefined,
        image: image.trim(),
        disk_gb: diskGb,
        gpu_count: caps?.count_choice ? count : undefined,
        cloud: caps?.cloud_choice ? cloud : undefined,
        template_id: caps?.template_required ? templateId : undefined,
        cors_origins: [window.location.origin],
        studio_url: window.location.origin + window.location.pathname,
        copy_remote: remote || undefined,
        copy_hf_token: copyHf || undefined,
        copy_notify: copyNotify || undefined,
        when_done: whenDone,
        self_destroy_key: keyDestroy && selfKey ? true : undefined,
        env: Object.keys(envParsed.env).length ? envParsed.env : undefined,
      });
      onCreated(m, s);
      onClose();
    } catch (e) {
      // the server answered, so this request is settled: the next try is a new one
      if (e instanceof ApiError && e.status > 0) rid.current = requestId();
      const msg = e instanceof Error ? e.message : String(e);
      const code = e instanceof ApiError ? e.code : '';
      if (code === 'no_capacity') {
        setProblem({ text: 'That offer was just taken. Pick another.' });
        setOfferId(null);
        setRefresh(r => r + 1);
      } else if (code === 'conflict' && msg.includes('$')) {
        // the price moved: the button and the cap take the new quote
        const d = e instanceof ApiError ? (e.details as { price_per_hour?: unknown } | undefined) : undefined;
        const now = typeof d?.price_per_hour === 'number' ? d.price_per_hour : priceIn(msg);
        if (now != null) setRequote({ id: offer.id, was: offer.price_per_hour, price: now });
        setProblem({ text: msg });
        setRefresh(r => r + 1);
      } else if (code === 'insufficient_credit') {
        setProblem({ text: `Not enough credit at ${label}. Add some, then try again.`, link: p.console_url });
      } else if (code === 'provider_auth') {
        // the key step replaces the form, so the reason shows there
        setKeyError(new Error(`${label} refused the key: ${msg}`));
        setChangingKey(true);
      } else if (code === 'rate_limited') {
        const after = e instanceof ApiError ? e.retryAfter : null;
        setProblem({ text: after ? `${label} is busy. Try again in ${after} s.` : msg });
      } else {
        setProblem({ text: msg });
      }
    } finally {
      sending.current = false;
      setBusy(false);
    }
  };

  const old = provs.error instanceof ApiError && provs.error.status === 404;
  return (
    <Dialog
      wide
      title="Create a GPU server"
      description={
        <>
          Created through <strong>{s.name}</strong>, which tracks it and holds the provider keys.
        </>
      }
      onClose={onClose}
      footer={
        p &&
        configured && (
          <>
            {offer && (
              <p className="create-summary">
                {offer.gpu}
                {offer.gpu_count > 1 ? ` × ${offer.gpu_count}` : ''} on {label} · about {price} while running
                {offer.stopped_price_per_hour != null ? `, ${perHour(offer.stopped_price_per_hour)} stopped` : ''}
              </p>
            )}
            <button className="btn btn-ghost" onClick={onClose}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={create} disabled={!ready || busy}>
              {busy ? 'Creating…' : offer ? `Create for ${price}` : 'Create'}
            </button>
          </>
        )
      }
    >
      <div className="dialog-form create">
        {old && <Callout kind="info">This server's image is older than GPU provisioning. Update it, or switch to another server.</Callout>}
        {!old && <ErrorNote error={provs.error} />}
        {!provs.data && !provs.error && <p className="quiet-note">Loading providers…</p>}
        {provs.data && list.length === 0 && <Callout kind="info">No GPU providers are enabled on this server (AITK_PROVISION_PROVIDERS).</Callout>}

        {p && (
          <>
            <div className="field">
              <span>Provider</span>
              <div className="seg" role="radiogroup" aria-label="Provider">
                {list.map(x => (
                  <button key={x.kind} type="button" role="radio" aria-checked={x.kind === kind} className={x.kind === kind ? 'is-on' : ''} onClick={() => setKind(x.kind)}>
                    {x.label}
                    {x.stage === 'beta' && <span className="seg-badge">Beta</span>}
                  </button>
                ))}
              </div>
              {p.unconfirmed.length > 0 && (
                <details className="create-unverified">
                  <summary>What is not verified</summary>
                  <ul>
                    {p.unconfirmed.map(u => (
                      <li key={u}>{u}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
            {p.simulated && <Callout kind="info">Simulated: nothing is rented and no money is spent. The new server is this server under another address.</Callout>}

            <KeyStep
              p={p}
              configured={configured}
              credit={credit[p.kind]}
              rented={!!s.machine}
              keyText={keyText}
              setKeyText={setKeyText}
              busy={keyBusy}
              error={keyError}
              onSave={saveKey}
              onChange={() => setChangingKey(true)}
              onCancelChange={
                p.configured && changingKey
                  ? () => {
                      setChangingKey(false);
                      setKeyError(null);
                    }
                  : undefined
              }
              onRemove={p.key_source === 'api' ? dropKey : undefined}
            />
          </>
        )}

        {p && configured && (
          <>
            <div className="field">
              <span>GPU</span>
              <div className="offer-filters">
                <input className="offer-search" value={gpu} onChange={e => setGpu(e.target.value)} placeholder="Search, e.g. 4090" aria-label="Search GPUs" />
                <label className="field-inline">
                  <span>Min VRAM</span>
                  <select value={minVram} onChange={e => setMinVram(Number(e.target.value))}>
                    {vramSteps.map(v => (
                      <option key={v} value={v}>
                        {v ? `${v} GB` : 'Any'}
                      </option>
                    ))}
                  </select>
                </label>
                {caps?.count_choice && (
                  <label className="field-inline">
                    <span>GPUs</span>
                    <select value={count} onChange={e => setCount(Number(e.target.value))}>
                      {countSteps.map(v => (
                        <option key={v} value={v}>
                          {v}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {caps?.cloud_choice && (
                  <div className="seg" role="radiogroup" aria-label="Cloud">
                    {(['secure', 'community'] as const).map(c => (
                      <button key={c} type="button" role="radio" aria-checked={cloud === c} className={cloud === c ? 'is-on' : ''} onClick={() => setCloud(c)}>
                        {c === 'secure' ? 'Secure' : 'Community'}
                      </button>
                    ))}
                  </div>
                )}
                <label className="field-inline">
                  <span>Disk (GB)</span>
                  <input className="offer-disk" type="number" min={20} max={2000} inputMode="numeric" value={disk} onChange={e => setDisk(e.target.value)} aria-invalid={!!diskError} />
                </label>
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setRefresh(r => r + 1)} disabled={offersLoading}>
                  <RefreshCw />
                  Refresh
                </button>
              </div>
              {diskError && <small className="field-error">{diskError}</small>}
            </div>
            <OfferList kind={p.kind} offers={offerList} loading={offersLoading && !offerList} error={offerError} selected={offerId} onSelect={setOfferId} />

            {caps?.template_required && (
              <TemplateStep tpls={tpls} error={tplError} templateId={templateId} setTemplateId={setTemplateId} tpl={tpl} onGuide={() => onGuide(p.kind)} />
            )}

            <div className="form-grid create-settings">
              <label className="field">
                <span>Name</span>
                <input value={name} onChange={e => setName(e.target.value)} placeholder={defaultName || 'Named after the GPU'} maxLength={64} />
              </label>
              <label className="field">
                <span>Image</span>
                <input className="mono" value={image} onChange={e => setImage(e.target.value)} spellCheck={false} autoCapitalize="off" />
                {image.trim() && imageRepo(image) !== OUR_REPO && <small className="warn">Not the aitk image. The server must answer /v1 on port 8675 with AI_TOOLKIT_AUTH.</small>}
              </label>
              <label className="field field-wide">
                <span>Keep results on</span>
                <select value={remote} onChange={e => setRemote(e.target.value)}>
                  <option value="">Don't copy a remote</option>
                  {(extras.data?.remotes ?? []).map(r => (
                    <option key={r.id} value={r.name}>
                      {r.name} ({r.kind === 's3' ? 'S3' : 'Hugging Face'})
                    </option>
                  ))}
                </select>
                <small>Copies this remote, keys included, into the new server's environment. Anyone with access to your provider account can read them.</small>
              </label>
              {((extras.data?.sinks ?? 0) > 0 || extras.data?.hfToken) && (
                <div className="field field-wide checks">
                  {(extras.data?.sinks ?? 0) > 0 && (
                    <label className="check">
                      <input type="checkbox" checked={copyNotify} onChange={e => setCopyNotify(e.target.checked)} />
                      Send its notifications where this server sends them
                    </label>
                  )}
                  {extras.data?.hfToken && (
                    <label className="check">
                      <input type="checkbox" checked={copyHf} onChange={e => setCopyHf(e.target.checked)} />
                      Give it this server's Hugging Face token
                    </label>
                  )}
                </div>
              )}
              <div className="field field-wide">
                <span>When done</span>
                <div className="seg" role="radiogroup" aria-label="When done">
                  {(
                    [
                      ['keep', 'Keep running'],
                      ['stop', 'Stop'],
                      ['destroy', 'Delete'],
                    ] as const
                  ).map(([v, l]) => (
                    <button
                      key={v}
                      type="button"
                      role="radio"
                      aria-checked={whenDone === v}
                      className={whenDone === v ? 'is-on' : ''}
                      disabled={v !== 'keep' && !canAct}
                      onClick={() => {
                        if (v === whenDone) return; // only a change stops the seg following the remote
                        setChosen(v);
                        setTouched(true);
                      }}
                    >
                      {l}
                    </button>
                  ))}
                </div>
                <small>{whenHint || 'What the server does after its last run finishes.'}</small>
              </div>
              {keyDestroy && (
                <div className="field field-wide">
                  <label className="check">
                    <input type="checkbox" checked={selfKey} onChange={e => setSelfKey(e.target.checked)} />
                    Let it stop or delete itself with my key
                  </label>
                  {selfKey && <Callout kind="warn">Your provider key goes into the server's environment, visible in the provider console.</Callout>}
                </div>
              )}
              <details className="field field-wide create-env">
                <summary>More environment variables</summary>
                <textarea
                  className="mono"
                  rows={4}
                  value={envText}
                  onChange={e => setEnvText(e.target.value)}
                  placeholder={'AITK_EXAMPLE=value\nTZ=Europe/Berlin'}
                  spellCheck={false}
                  autoCapitalize="off"
                  aria-invalid={!!envParsed.error}
                />
                {envParsed.error ? <small className="field-error">{envParsed.error}</small> : <small>One KEY=value per line: AITK_…, HF_…, HUGGING_FACE_HUB_TOKEN or TZ.</small>}
              </details>
            </div>

            {problem && (
              <Callout kind="danger">
                {problem.text}
                {problem.link && (
                  <>
                    {' '}
                    <a href={problem.link} target="_blank" rel="noreferrer">
                      Open the {label} console
                    </a>
                  </>
                )}
              </Callout>
            )}
          </>
        )}
      </div>
    </Dialog>
  );
}

function KeyStep({
  p,
  configured,
  credit,
  rented,
  keyText,
  setKeyText,
  busy,
  error,
  onSave,
  onChange,
  onCancelChange,
  onRemove,
}: {
  p: Provider;
  configured: boolean;
  credit: number | null | undefined;
  rented: boolean;
  keyText: string;
  setKeyText: (v: string) => void;
  busy: boolean;
  error: unknown;
  onSave: () => void;
  onChange: () => void;
  onCancelChange?: () => void;
  onRemove?: () => void;
}) {
  if (configured)
    return (
      <div className="field">
        <span>API key</span>
        <p className="key-line">
          {p.key_source === 'sim' ? 'Simulated key' : p.key_hint ? `Key ending ${p.key_hint}` : 'Key set'}
          {p.key_source === 'env' && <span className="badge tone-quiet">From env</span>}
          {credit != null && <span className="num"> · ${credit.toFixed(2)} credit</span>}
          <span aria-hidden> · </span>
          <button type="button" className="link" onClick={onChange}>
            Change
          </button>
          {onRemove && (
            <>
              <span aria-hidden> · </span>
              <button type="button" className="link" onClick={onRemove}>
                Remove
              </button>
            </>
          )}
        </p>
        <ErrorNote error={error} />
      </div>
    );
  return (
    <form
      className="field"
      onSubmit={e => {
        e.preventDefault();
        onSave();
      }}
    >
      <label htmlFor="provider-key">{p.label} API key</label>
      {rented && (
        <Callout kind="warn">
          This is a rented GPU server. If it is terminated, its keys and the GPU servers it tracks go with it; machines it launched keep billing.
        </Callout>
      )}
      <div className="key-row">
        <input id="provider-key" type="password" autoComplete="new-password" value={keyText} onChange={e => setKeyText(e.target.value)} placeholder={`Paste your ${p.label} key`} />
        <button className="btn btn-primary" type="submit" disabled={busy || !keyText.trim()}>
          {busy ? 'Checking…' : 'Save key'}
        </button>
        {onCancelChange && (
          <button className="btn btn-ghost" type="button" onClick={onCancelChange}>
            Cancel
          </button>
        )}
      </div>
      <small>
        <a href={p.keys_url} target="_blank" rel="noreferrer">
          Where do I find it?
        </a>
        {p.key_scopes.length > 0 && <> Give it {p.key_scopes.join(', ')}.</>} It is checked with {p.label}, kept on this server and never stored in the browser.
      </small>
      <ErrorNote error={error} />
    </form>
  );
}

const availability: Record<Offer['availability'], string> = { high: 'High availability', medium: 'Some availability', low: 'Low availability', available: 'Available' };

function OfferList({
  kind,
  offers,
  loading,
  error,
  selected,
  onSelect,
}: {
  kind: ProviderKind;
  offers: Offer[] | null;
  loading: boolean;
  error: unknown;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  if (error) return <ErrorNote error={error} />;
  if (loading || !offers)
    return (
      <div className="offer-list" aria-busy="true">
        {[0, 1, 2].map(i => (
          <div key={i} className="offer is-skeleton" aria-hidden />
        ))}
      </div>
    );
  if (!offers.length) return <p className="quiet-note offer-empty">No offers match. Loosen the filters.</p>;
  return (
    <div className="offer-list" role="radiogroup" aria-label="Offers">
      {offers.map(o => {
        const on = o.id === selected;
        const facts = [
          `${o.vram_gb} GB`,
          o.location || (kind === 'runpod' ? `${providerNames.runpod} picks the data center` : ''),
          o.reliability != null ? `${Math.round(o.reliability * 100)}% reliable` : '',
          availability[o.availability] ?? '',
          o.https ? 'https' : 'plain http',
        ].filter(Boolean);
        return (
          <button key={o.id} type="button" role="radio" aria-checked={on} className={on ? 'offer is-selected' : 'offer'} onClick={() => onSelect(o.id)}>
            <span className="offer-main">
              <span className="offer-gpu">
                {o.gpu}
                {o.gpu_count > 1 ? ` × ${o.gpu_count}` : ''}
              </span>
              <span className="offer-facts">{facts.join(' · ')}</span>
            </span>
            <span className="offer-price">
              <span className="offer-rate">{perHour(o.price_per_hour)}</span>
              {o.stopped_price_per_hour != null && <span className="offer-stopped">{perHour(o.stopped_price_per_hour)} stopped</span>}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function TemplateStep({
  tpls,
  error,
  templateId,
  setTemplateId,
  tpl,
  onGuide,
}: {
  tpls: ProviderTemplate[] | null;
  error: unknown;
  templateId: string;
  setTemplateId: (id: string) => void;
  tpl: ProviderTemplate | null;
  onGuide: () => void;
}) {
  if (error) return <ErrorNote error={error} />;
  if (!tpls) return <p className="quiet-note">Loading templates…</p>;
  if (!tpls.length)
    return (
      <Callout kind="info">
        QuickPod starts pods from a template you create in its console.{' '}
        <button type="button" className="link" onClick={onGuide}>
          Show me how
        </button>
      </Callout>
    );
  return (
    <label className="field">
      <span>Template</span>
      <select value={templateId} onChange={e => setTemplateId(e.target.value)}>
        {tpls.map(t => (
          <option key={t.id} value={t.id}>
            {t.name}
            {t.ours ? ' (aitk)' : ''}
          </option>
        ))}
      </select>
      {tpl && !tpl.publishes_port && <Callout kind="warn">This template doesn't publish port 8675.</Callout>}
      {tpl && !tpl.has_token && <Callout kind="warn">Add -e AI_TOOLKIT_AUTH to the template so the server never starts open.</Callout>}
      <small>
        <button type="button" className="link" onClick={onGuide}>
          How to make one
        </button>
      </small>
    </label>
  );
}

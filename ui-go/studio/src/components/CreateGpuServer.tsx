import { Cpu, RefreshCw } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, get, type Items, type Remote, type Server, type Settings } from '../lib/api';
import type { NotifyConfig } from '../lib/lifecycle';
import { perHour, providerNames } from '../lib/machines';
import { launch, offers as listOffers, providers, recommend, recommendAll, removeKey, setKey, templates as listTemplates, testKey } from '../lib/provision';
import type { GpuRecommendation, Machine, Offer, OfferQuery, Provider, ProviderKind, ProviderTemplate, RecommendQuery, WhenDone } from '../lib/provisionTypes';
import { archName, checkImage, DEFAULT_IMAGE, gb, imageTag, minutes, modelArchs, recommendImage, type ImageCheckRequest, type ImageRecommendRequest } from '../lib/serverImage';
import type { CheckAction, ImageCheck, ImageCheckItem, ImageRecommendation, ImageRequirements } from '../lib/serverImageTypes';
import { Dialog } from './Dialog';
import { FitBadge, VramBadge } from './FitBadge';
import { GpuPicks, pickBadge, pickLine } from './GpuPicks';
import { ImageDialog } from './ImageDialog';
import { CheckList, diskBreakdown, diskParts, ImageRow, imageName } from './ImageRow';
import { Callout, Empty, ErrorNote, useLoad } from './ui';

const vramSteps = [0, 16, 24, 48, 80];
const countSteps = [1, 2, 4, 8];

const round2 = (n: number) => Math.round(n * 100) / 100;

/** "RTX 30-series or newer" from the image's SASS targets (sm_75 is hidden from offers by default). */
function gpuFloor(sm: number[]): string {
  const min = Math.min(...sm.filter(x => x > 75));
  if (!Number.isFinite(min)) return '';
  if (min <= 86) return 'RTX 30-series or newer';
  if (min <= 89) return 'RTX 40-series or newer';
  if (min >= 120) return 'RTX 50-series or newer';
  return `sm_${min} or newer`;
}

/** A request's answer, kept with the key it was asked for, so a stale one never shows for new inputs. */
type Answer<T> = { key: string; data: T | null; error: unknown };

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

type LaunchProblem = { text: string; link?: string; checks?: ImageCheckItem[] };

/**
 * Create a GPU server (PLAN §11): through the active server, pick a provider,
 * an offer and the settings, then rent it at a capped price.
 */
export function CreateGpuServer({
  controller,
  model,
  onClose,
  onCreated,
  onGuide,
  onAddServer,
}: {
  controller: Server | null;
  /** The model arch of the run the user came from, for "What will you train?". */
  model?: string;
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
  return <CreateForm controller={controller} model={model} onClose={onClose} onCreated={onCreated} onGuide={onGuide} />;
}

function CreateForm({
  controller: s,
  model,
  onClose,
  onCreated,
  onGuide,
}: {
  controller: Server;
  model?: string;
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
  const [diskTouched, setDiskTouched] = useState(false); // until then the disk follows what the image and model need
  const [arch, setArch] = useState(model && modelArchs.some(m => m.arch === model) ? model : '');
  const [fast, setFast] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [offerList, setOfferList] = useState<Offer[] | null>(null);
  const [offerError, setOfferError] = useState<unknown>(null);
  const [offersLoading, setOffersLoading] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [offerId, setOfferId] = useState<string | null>(null);
  const diskGb = Number(disk) || 0;
  const diskOk = diskGb >= 20 && diskGb <= 2000;

  // QuickPod templates (loaded below) decide the image there
  const [tpls, setTpls] = useState<ProviderTemplate[] | null>(null);
  const [tplError, setTplError] = useState<unknown>(null);
  const [templateId, setTemplateId] = useState('');
  const tpl = tpls?.find(t => t.id === templateId) ?? null;
  const fromTemplate = !!caps?.template_required;
  // the image the user picked in the Change dialog; null follows images.recommend
  const [picked, setPicked] = useState<string | null>(null);

  // The offers are judged against the image the Image row settles on: the
  // template's, the picked one, or what images.recommend picks with no host
  // (this server's build when it is published, else latest). The row's own
  // recommend follows the chosen offer, so it cannot feed the offers query.
  // It also settles the image for the cross-provider line, so that line asks
  // once with the image, not first without it, on QuickPod too.
  const configuredKinds = list.filter(x => x.configured).map(x => x.kind);
  const wantCross = configuredKinds.length >= 2;
  const [baseRec, setBaseRec] = useState<Answer<ImageRecommendation> | null>(null);
  const wantBase = (!!p && configured && !fromTemplate) || wantCross;
  useEffect(() => {
    if (!wantBase || baseRec?.key === s.id) return;
    let live = true;
    const key = s.id;
    // a slow registry never holds the offers for long: they then use the server's default
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), 8000);
    recommendImage(s, {}, ac.signal)
      .then(
        data => live && setBaseRec({ key, data, error: null }),
        error => live && setBaseRec({ key, data: null, error }),
      )
      .finally(() => clearTimeout(t));
    return () => {
      live = false;
      clearTimeout(t);
      ac.abort();
    };
  }, [s.id, wantBase]); // eslint-disable-line react-hooks/exhaustive-deps
  const baseNow = baseRec?.key === s.id ? baseRec : null;
  // a failed or empty answer leaves image unset: the server then judges its default
  const offersImage = fromTemplate ? tpl?.image : (picked ?? baseNow?.data?.choice?.image);
  const offersSettled = fromTemplate ? tpls != null || tplError != null : picked != null || !!baseNow;

  const q: OfferQuery = {
    gpu: gpu.trim() || undefined,
    min_vram_gb: minVram || undefined,
    gpu_count: caps?.count_choice ? count : undefined,
    cloud: caps?.cloud_choice ? cloud : undefined,
    disk_gb: diskGb >= 20 ? diskGb : undefined,
    limit: 50,
    image: offersImage || undefined,
    model: arch || undefined,
    fast: fast && caps?.reports_inet ? true : undefined,
    show_all: showAll || undefined,
  };
  const qKey = JSON.stringify(q);
  const seq = useRef(0);
  useEffect(() => {
    if (!p || !configured) {
      setOfferList(null);
      return;
    }
    if (!offersSettled) return; // asked once the image is known, not for the default first
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
  }, [s.id, p?.kind, configured, qKey, offersSettled, refresh]); // eslint-disable-line react-hooks/exhaustive-deps

  // GPU picks (MASTER §5.1.5): the recommender's best value and best speed,
  // judged against the same image as the offers, with the model and the Disk
  // and Cloud filters, not the search text or Min VRAM. They are for one GPU,
  // so a count above 1 leaves them out.
  const [userChose, setUserChose] = useState(false); // once set, refreshes never move the selection
  const wantPicks = !!p && configured && offersSettled && (!caps?.count_choice || count === 1);
  const pickQ: RecommendQuery = {
    model: arch || undefined,
    image: offersImage || undefined,
    disk_gb: diskOk ? diskGb : undefined,
    cloud: caps?.cloud_choice ? cloud : undefined,
  };
  const pickKey = JSON.stringify([s.id, p?.kind, pickQ]);
  const [picks, setPicks] = useState<(Answer<GpuRecommendation> & { kind: ProviderKind }) | null>(null);
  const pickSeq = useRef(0);
  useEffect(() => {
    if (!wantPicks || !p) return;
    const n = ++pickSeq.current;
    const k = p.kind;
    const t = setTimeout(() => {
      recommend(s, k, pickQ).then(
        data => n === pickSeq.current && setPicks({ key: pickKey, kind: k, data, error: null }),
        error => n === pickSeq.current && setPicks({ key: pickKey, kind: k, data: null, error }),
      );
    }, 400);
    return () => clearTimeout(t);
  }, [pickKey, wantPicks, refresh]); // eslint-disable-line react-hooks/exhaustive-deps
  const picksNow = picks?.key === pickKey ? picks : null;
  // while new picks load, the last ones for this provider stay on screen
  const picksShown = !wantPicks ? null : (picksNow ?? (picks?.kind === p?.kind ? picks : null));
  const gpuRec = picksShown?.data ?? null;

  // Best value across every configured provider, for the Provider step
  const crossSettled = picked != null || !!baseNow;
  const crossQ: RecommendQuery = {
    model: arch || undefined,
    // the Image row's image off QuickPod (a template's image is for QuickPod only)
    image: (picked ?? baseNow?.data?.choice?.image) || undefined,
    disk_gb: diskOk ? diskGb : undefined,
    cloud: cloud === 'community' ? cloud : undefined,
  };
  const crossKey = JSON.stringify([s.id, configuredKinds, crossQ]);
  const [cross, setCross] = useState<Answer<GpuRecommendation> | null>(null);
  const crossSeq = useRef(0);
  useEffect(() => {
    if (!wantCross || !crossSettled) return;
    const n = ++crossSeq.current;
    const t = setTimeout(() => {
      recommendAll(s, crossQ).then(
        data => n === crossSeq.current && setCross({ key: crossKey, data, error: null }),
        // a quiet line: an error leaves it out
        error => n === crossSeq.current && setCross({ key: crossKey, data: null, error }),
      );
    }, 400);
    return () => clearTimeout(t);
  }, [crossKey, wantCross, crossSettled]); // eslint-disable-line react-hooks/exhaustive-deps
  const crossValue = wantCross && cross?.key === crossKey ? (cross.data?.value ?? null) : null;
  // the offer Use it chose; it may be in neither the list nor the provider's picks
  const [crossChosen, setCrossChosen] = useState<Offer | null>(null);
  const pendingCross = useRef<Offer | null>(null);
  const takeCross = (o: Offer) => {
    setCount(1); // the picks are for one GPU
    if (o.provider === p?.kind) {
      setOfferId(o.id);
      setUserChose(true);
      setCrossChosen(o);
      return;
    }
    pendingCross.current = o;
    setKind(o.provider);
  };

  // One effect, so the provider reset always runs before the preselection: a
  // provider whose picks are already settled (back from a provider with no
  // key, or within the debounce) still gets its value pick selected.
  const shownKind = useRef<ProviderKind | null | undefined>(undefined);
  useEffect(() => {
    let chose = userChose;
    if (shownKind.current !== p?.kind) {
      shownKind.current = p?.kind;
      // a new provider starts from its own default pick, unless Use it switched to it
      const o = pendingCross.current?.provider === p?.kind ? pendingCross.current : null;
      pendingCross.current = null;
      setOfferId(o?.id ?? null);
      setUserChose(!!o);
      setCrossChosen(o);
      setOfferList(null); // another provider's offers never show under this one
      chose = !!o;
    }
    // the default is the value pick of each settled answer, and none when it has none
    if (!chose && picksNow) setOfferId(picksNow.data?.value?.offer.id ?? null);
  }, [p?.kind, picksNow, userChose]);
  const chooseOffer = (id: string) => {
    setOfferId(id);
    setUserChose(true);
  };
  // the picks' offers (the Pick carries each whole) beside the list, so a
  // selected pick outside the visible list is pinned first instead of lost
  const extra = [gpuRec?.value?.offer, gpuRec?.speed?.offer, crossChosen].filter(
    (o): o is Offer => !!o && o.provider === p?.kind && (!caps?.count_choice || count === 1),
  );
  const offer = offerList?.find(o => o.id === offerId) ?? extra.find(o => o.id === offerId) ?? null;
  const pinned = offer && !offerList?.some(o => o.id === offer.id) ? offer : null;
  // a 409's new quote; it holds while the list (cached 30 s) still shows the price it replaced
  const [requote, setRequote] = useState<{ id: string; was: number; price: number } | null>(null);
  const offerPrice = !offer ? 0 : requote?.id === offer.id && requote.was === offer.price_per_hour ? requote.price : offer.price_per_hour;

  // QuickPod templates
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

  // image: images.recommend picks it for the offer (QuickPod: checks the
  // template's), then images.check judges it on the offer's host
  const modelRef = arch ? { arch } : undefined;
  const offerRef = offer && p ? { provider: p.kind, offer_id: offer.id } : undefined;
  // the disk is left out: it follows the check's need, and the check judges it
  const recReq: ImageRecommendRequest = fromTemplate ? { template_id: templateId || undefined, offer: offerRef, model: modelRef } : { offer: offerRef, model: modelRef };
  const recKey = JSON.stringify([s.id, recReq]);
  const [rec, setRec] = useState<Answer<ImageRecommendation> | null>(null);
  const recSeq = useRef(0);
  const wantRec = !!p && configured && (!fromTemplate || !!templateId);
  useEffect(() => {
    if (!wantRec) return;
    const n = ++recSeq.current;
    const t = setTimeout(() => {
      recommendImage(s, recReq).then(
        data => n === recSeq.current && setRec({ key: recKey, data, error: null }),
        error => n === recSeq.current && setRec({ key: recKey, data: null, error }),
      );
    }, 300);
    return () => clearTimeout(t);
  }, [recKey, wantRec]); // eslint-disable-line react-hooks/exhaustive-deps
  const recNow = rec?.key === recKey ? rec : null;
  const image = fromTemplate ? (tpl?.image ?? '') : (picked ?? rec?.data?.choice?.image ?? DEFAULT_IMAGE);
  // the recommended image is checked once it is known, not the default first
  const imageSettled = fromTemplate || picked != null || !!recNow;

  const checkReq: ImageCheckRequest = { image, offer: offerRef, model: modelRef, disk_gb: diskOk ? diskGb : undefined };
  const [recheck, setRecheck] = useState(0); // bumped when a launch was refused for the image
  const checkKey = JSON.stringify([s.id, checkReq, recheck]);
  const [check, setCheck] = useState<Answer<ImageCheck> | null>(null);
  const checkSeq = useRef(0);
  const live = !!p && configured && !!image.trim();
  const wantCheck = live && imageSettled;
  useEffect(() => {
    if (!wantCheck) return;
    const n = ++checkSeq.current;
    const t = setTimeout(() => {
      checkImage(s, checkReq).then(
        data => n === checkSeq.current && setCheck({ key: checkKey, data, error: null }),
        error => n === checkSeq.current && setCheck({ key: checkKey, data: null, error }),
      );
    }, 300);
    return () => clearTimeout(t);
  }, [checkKey, wantCheck]); // eslint-disable-line react-hooks/exhaustive-deps
  const checkNow = check?.key === checkKey ? check : null;
  const checking = live && !checkNow;
  // while a check is in flight the last answer stays on screen, marked as checking
  const shown = checkNow ?? check;
  const blocks = checkNow?.data?.checks.filter(c => c.level === 'block') ?? [];
  // the server re-checks every launch, so a slow check stops holding Create after 3 s
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    setSlow(false);
    if (!checking) return;
    const t = setTimeout(() => setSlow(true), 3000);
    return () => clearTimeout(t);
  }, [checking, checkKey]);
  const need = shown?.data?.need ?? null;
  useEffect(() => {
    if (diskTouched || !need) return;
    const want = Math.ceil(Math.min(need.disk_gb, offer?.max_disk_gb ?? Infinity));
    if (want >= 20 && want <= 2000) setDisk(String(want));
  }, [need?.disk_gb, offer?.max_disk_gb, diskTouched]); // eslint-disable-line react-hooks/exhaustive-deps
  const [changing, setChanging] = useState(false);
  const offersAnchor = useRef<HTMLDivElement>(null);
  // QuickPod: the template's tag note (edit_template), which images.check does not give;
  // otherwise the resolver's notes on the recommended build
  const recNotes = (recNow?.data?.checks ?? []).filter(c =>
    fromTemplate ? c.action?.kind === 'edit_template' : picked == null && (c.code === 'image_same_unpublished' || c.code === 'image_skew'),
  );
  // this server's build won't run on the host, so the choice is a newer one:
  // the resolver says so only in the first why line, and it is never silent
  const recChoice = recNow?.data?.choice;
  if (!fromTemplate && picked == null && recChoice?.channel === 'latest' && recChoice.why[0]?.startsWith('Newer build than this server'))
    recNotes.push({ level: 'info', code: 'image_newer_build', message: `${recChoice.why[0]}.` });
  // QuickPod runs only the template's image, so a "use this image" fix means editing the template
  const forTemplate = (checks: ImageCheckItem[]): ImageCheckItem[] =>
    !fromTemplate
      ? checks
      : checks.map(c =>
          c.action?.kind === 'use_image' ? { ...c, action: { kind: 'edit_template', value: typeof c.action.value === 'string' ? imageTag(c.action.value) || undefined : undefined } } : c,
        );

  // settings
  const [name, setName] = useState('');
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
  const imageHolds = blocks.length > 0 || (checking && !slow);
  const ready = !!p && configured && !!offer && !diskError && !envParsed.error && (!caps?.template_required || !!templateId) && !!image.trim() && !imageHolds;

  const onAction = (a: CheckAction) => {
    switch (a.kind) {
      case 'filter_hosts':
        setShowAll(false);
        if (a.value === 'fast') setFast(true);
        break;
      case 'show_all':
        setShowAll(true);
        break;
      case 'set_disk':
        if (Number(a.value) > 0) {
          setDisk(String(a.value));
          setDiskTouched(true);
        }
        break;
      case 'use_image':
        if (fromTemplate) {
          if (p?.console_url) window.open(p.console_url, '_blank', 'noreferrer');
        } else if (typeof a.value === 'string' && a.value) setPicked(a.value);
        break;
      case 'pick_gpu':
        setOfferId(null);
        setUserChose(true);
        offersAnchor.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        break;
      case 'edit_template':
        if (p?.console_url) window.open(p.console_url, '_blank', 'noreferrer');
        break;
    }
  };

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
        setUserChose(true); // the picks may still name it until the cache turns
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
      } else if (code === 'image_blocked') {
        const d = e instanceof ApiError ? (e.details as { checks?: ImageCheckItem[] } | undefined) : undefined;
        setProblem({ text: msg, checks: Array.isArray(d?.checks) ? d.checks : undefined });
        setRecheck(n => n + 1); // judge again: the server saw something this dialog did not
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
                {shown?.data?.image && (
                  <span className="create-summary-image">
                    Image: {imageName(shown.data.image, image)}
                    {shown.data.image.download_bytes > 0 ? ` · ${gb(shown.data.image.download_bytes)}` : ''}
                    {shown.data.pull ? `, about ${minutes(shown.data.pull.image_min)} to download` : ''}
                  </span>
                )}
                <span className="create-summary-disk">
                  Disk: {diskGb} GB
                  {need ? (diskGb >= need.disk_gb ? ` (${diskParts(need, arch ? archName(arch) : undefined)})` : ` (needs ${need.disk_gb} GB)`) : ''}
                </span>
              </p>
            )}
            <button className="btn btn-ghost" onClick={onClose}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={create} disabled={!ready || busy} title={blocks.length > 0 ? 'Fix what blocks the launch first' : undefined}>
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
              {crossValue && (
                <p className="pick-cross">
                  <span>Best value across your providers: {pickLine(crossValue.offer)}</span>
                  {!(crossValue.offer.provider === kind && crossValue.offer.id === offerId) && (
                    <button type="button" className="btn btn-ghost btn-sm" onClick={() => takeCross(crossValue.offer)}>
                      Use it
                    </button>
                  )}
                </p>
              )}
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
                  <span>What will you train?</span>
                  <select value={arch} onChange={e => setArch(e.target.value)}>
                    <option value="">Not sure yet</option>
                    {modelArchs.map(m => (
                      <option key={m.arch} value={m.arch}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </label>
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
                  <input
                    className="offer-disk"
                    type="number"
                    min={20}
                    max={2000}
                    inputMode="numeric"
                    value={disk}
                    onChange={e => {
                      setDisk(e.target.value);
                      setDiskTouched(true);
                    }}
                    aria-invalid={!!diskError}
                  />
                </label>
                {caps?.reports_inet && (
                  <label className="check offer-fast">
                    <input type="checkbox" checked={fast} onChange={e => setFast(e.target.checked)} />
                    Fast hosts only (500 Mbps+)
                  </label>
                )}
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setRefresh(r => r + 1)} disabled={offersLoading}>
                  <RefreshCw />
                  Refresh
                </button>
              </div>
              {diskError ? (
                <small className="field-error">{diskError}</small>
              ) : (
                need && <small>Needs {diskBreakdown(need, arch ? archName(arch) : undefined)}.</small>
              )}
            </div>
            <div ref={offersAnchor} className="offer-block">
              <GpuPicks
                rec={gpuRec}
                loading={wantPicks && !picksNow}
                error={picksShown?.error ?? null}
                selectedOfferId={offer?.id ?? null}
                onPick={o => chooseOffer(o.id)}
              />
              <OfferHeader
                offers={offerList}
                showAll={showAll}
                onShowAll={setShowAll}
                name={shown?.data?.image?.go_server ? imageName(shown.data.image, image).replace(/ · .*$/, '') : ''}
                requires={shown?.data?.image?.requires ?? null}
              />
              <OfferList
                kind={p.kind}
                offers={offerList}
                loading={offersLoading && !offerList}
                error={offerError}
                selected={offer?.id ?? null}
                onSelect={chooseOffer}
                model={arch ? archName(arch) : ''}
                pinned={pinned}
                badge={id => pickBadge(gpuRec, id)}
              />
            </div>

            {caps?.template_required && (
              <TemplateStep tpls={tpls} error={tplError} templateId={templateId} setTemplateId={setTemplateId} tpl={tpl} onGuide={() => onGuide(p.kind)} />
            )}

            <div className="form-grid create-settings">
              {(!fromTemplate || tpl) && (
                <ImageRow
                  label={fromTemplate && tpl ? `Image (from template ${tpl.name})` : 'Image'}
                  image={image}
                  check={shown?.data ? { ...shown.data, checks: forTemplate(shown.data.checks) } : null}
                  checking={checking}
                  error={shown?.error ?? null}
                  hasHost={!!offer}
                  extra={recNotes}
                  onChange={fromTemplate ? undefined : () => setChanging(true)}
                  onAction={onAction}
                />
              )}
              <label className="field">
                <span>Name</span>
                <input value={name} onChange={e => setName(e.target.value)} placeholder={defaultName || 'Named after the GPU'} maxLength={64} />
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

            {changing && (
              <ImageDialog
                server={s}
                rec={rec?.data ?? null}
                recError={rec?.error ?? null}
                current={image}
                auto={picked == null}
                context={{ offer: offerRef, model: modelRef, disk_gb: diskOk ? diskGb : undefined }}
                onPick={setPicked}
                onClose={() => setChanging(false)}
              />
            )}

            {problem?.checks && problem.checks.length > 0 ? (
              <CheckList checks={forTemplate(problem.checks)} onAction={onAction} />
            ) : (
              problem && (
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
              )
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

/**
 * The line above the offers (IMAGE-PLAN §6.2): which hosts are left out for
 * the image, and the switch to show them marked instead.
 */
function OfferHeader({
  offers,
  showAll,
  onShowAll,
  name,
  requires,
}: {
  offers: Offer[] | null;
  showAll: boolean;
  onShowAll: (v: boolean) => void;
  name: string;
  requires: ImageRequirements | null;
}) {
  if (!offers) return null;
  // the offers were not judged (an image that could not be read): every host shows
  if (offers.length > 0 && offers.every(o => !o.fit)) {
    return showAll ? null : <p className="offer-head">Couldn't check which hosts run this image, so every host is shown.</p>;
  }
  if (showAll)
    return (
      <p className="offer-head">
        <span>Showing every host. Hosts that can't run the image are marked.</span>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => onShowAll(false)}>
          Show only hosts that fit
        </button>
      </p>
    );
  const terms = [requires?.cuda_min ? `CUDA ${requires.cuda_min}+` : '', requires ? gpuFloor(requires.sm ?? []) : ''].filter(Boolean);
  return (
    <p className="offer-head">
      <span>
        Only hosts that can run {name || 'the image'}
        {terms.length > 0 && ` (${terms.join(', ')})`} are shown.
      </span>
      <button type="button" className="btn btn-ghost btn-sm" onClick={() => onShowAll(true)}>
        Show all hosts
      </button>
    </p>
  );
}

const vramLine: Record<string, (m: string) => string> = { fits: m => `fits ${m}`, tight: m => `${m} fits only with a recipe`, no: m => `${m} doesn't fit` };

function OfferList({
  kind,
  offers,
  loading,
  error,
  selected,
  onSelect,
  model,
  pinned,
  badge,
}: {
  kind: ProviderKind;
  offers: Offer[] | null;
  loading: boolean;
  error: unknown;
  selected: string | null;
  onSelect: (id: string) => void;
  /** The display name of the model being trained, '' when not chosen. */
  model: string;
  /** A selected pick the list does not show (its filters differ): it goes first. */
  pinned: Offer | null;
  /** The pick an offer stands for ("Recommended", "Best speed"), or ''. */
  badge: (id: string) => string;
}) {
  if (error && !pinned) return <ErrorNote error={error} />;
  if ((loading || !offers) && !pinned)
    return (
      <div className="offer-list" aria-busy="true">
        {[0, 1, 2].map(i => (
          <div key={i} className="offer is-skeleton" aria-hidden />
        ))}
      </div>
    );
  const rest = loading || error ? [] : (offers ?? []);
  if (!rest.length && !pinned) return <p className="quiet-note offer-empty">No offers match. Loosen the filters.</p>;
  return (
    <>
      <div className="offer-list" role="radiogroup" aria-label="Offers">
        {(pinned ? [pinned, ...rest] : rest).map(o => {
          const on = o.id === selected;
          const tag = badge(o.id) || (o === pinned ? 'Recommended' : '');
          const fit = o.fit;
          const blocked = fit?.level === 'block';
          const facts = [
            fit?.pull_min != null ? `≈ ${minutes(fit.pull_min)} to download` : '',
            model && fit && vramLine[fit.vram] ? vramLine[fit.vram](model) : '',
            o.location || (kind === 'runpod' ? `${providerNames.runpod} picks the data center` : ''),
            o.reliability != null ? `${Math.round(o.reliability * 100)}% reliable` : '',
            availability[o.availability] ?? '',
            o.https ? 'https' : 'plain http',
          ].filter(Boolean);
          return (
            <button
              key={o.id}
              type="button"
              role="radio"
              aria-checked={on}
              aria-disabled={blocked || undefined}
              disabled={blocked}
              title={blocked ? fit?.summary : undefined}
              className={on ? 'offer is-selected' : blocked ? 'offer is-blocked' : 'offer'}
              onClick={() => !blocked && onSelect(o.id)}
            >
              <span className="offer-main">
                <span className="offer-gpu">
                  {o.gpu}
                  {o.gpu_count > 1 ? ` × ${o.gpu_count}` : ''}
                  <VramBadge gb={o.vram_gb} verdict={fit?.vram} model={model} />
                  {tag && <span className="badge tone-accent">{tag}</span>}
                </span>
                <span className="offer-facts">{facts.join(' · ')}</span>
                {blocked && fit?.summary && <span className="offer-why">{fit.summary}</span>}
              </span>
              <span className="offer-price">
                <span className="offer-rate">{perHour(o.price_per_hour)}</span>
                {o.stopped_price_per_hour != null && <span className="offer-stopped">{perHour(o.stopped_price_per_hour)} stopped</span>}
              </span>
              <span className="offer-fit">
                <FitBadge fit={fit} />
              </span>
            </button>
          );
        })}
      </div>
      {/* the pinned pick stands alone while the list loads, fails or is empty */}
      {pinned && error ? <ErrorNote error={error} /> : pinned && !loading && offers && !offers.length ? <p className="quiet-note offer-empty">No other offers match. Loosen the filters.</p> : null}
    </>
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

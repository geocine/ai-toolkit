// GPU servers on RunPod, Vast.ai and QuickPod, created through a controller:
// any aitk server the studio knows (/v1/providers, /v1/machines; docs/headless.md).
// The studio never calls a provider itself, and provider keys are sent to the
// controller once and never kept in the browser.
import { call, del, get, post, type Items, type Server } from './api';
import type {
  LaunchRequest,
  Machine,
  MachineLink,
  MachineLogs,
  MachinePatch,
  Offer,
  OfferQuery,
  Provider,
  ProviderKind,
  ProviderTemplate,
  ProviderTest,
  ProviderUpdate,
  SetupGuide,
} from './provisionTypes';

const k = (kind: ProviderKind) => encodeURIComponent(kind);
const m = (id: string) => encodeURIComponent(id);

/**
 * Query string from any flat object: every defined key becomes a param and
 * booleans travel as 1. Empty strings and false are left out, so new
 * OfferQuery keys need no change here.
 */
export function query(q: object): string {
  const p = new URLSearchParams();
  for (const [key, v] of Object.entries(q)) {
    if (v === undefined || v === null || v === '' || v === false) continue;
    p.set(key, v === true ? '1' : String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : '';
}

export const providers = async (s: Server) => (await get<Items<Provider>>(s, '/v1/providers')).items;
export const updateProvider = (s: Server, kind: ProviderKind, u: ProviderUpdate) => call<Provider>(s, 'PUT', `/v1/providers/${k(kind)}`, u);
/** Verifies the key at the provider first; a bad key is a provider_auth error and nothing is stored. */
export const setKey = (s: Server, kind: ProviderKind, apiKey: string) => updateProvider(s, kind, { api_key: apiKey });
/** Removes the stored key; a key from the server's environment still applies. */
export const removeKey = (s: Server, kind: ProviderKind) => del<void>(s, `/v1/providers/${k(kind)}`);
export const testKey = (s: Server, kind: ProviderKind) => post<ProviderTest>(s, `/v1/providers/${k(kind)}/test`);
/** Cheapest first. */
export const offers = async (s: Server, kind: ProviderKind, q: OfferQuery = {}) =>
  (await get<Items<Offer>>(s, `/v1/providers/${k(kind)}/offers${query(q)}`)).items;
export const templates = async (s: Server, kind: ProviderKind) => (await get<Items<ProviderTemplate>>(s, `/v1/providers/${k(kind)}/templates`)).items;
/** The manual template guide. It never takes a token; the studio fills it in. */
export const setup = (s: Server, kind: ProviderKind, o: { image?: string; disk_gb?: number; cors_origin?: string; generate_token?: boolean } = {}) =>
  get<SetupGuide>(s, `/v1/providers/${k(kind)}/setup${query(o)}`);

/** Newest first; active=true leaves out ended machines. */
export const machines = async (s: Server, active = false, signal?: AbortSignal) =>
  (await get<Items<Machine>>(s, `/v1/machines${active ? '?active=1' : ''}`, signal)).items;
/** 201 with a new machine, 200 with the existing one for a repeated request_id. */
export const launch = (s: Server, req: LaunchRequest) => post<Machine>(s, '/v1/machines', req);
export const getMachine = (s: Server, id: string, refresh = false) => get<Machine>(s, `/v1/machines/${m(id)}${refresh ? '?refresh=1' : ''}`);
/** Rename, or paste the template's token (re-probes at once). */
export const patchMachine = (s: Server, id: string, p: MachinePatch) => call<Machine>(s, 'PATCH', `/v1/machines/${m(id)}`, p);
export const stop = (s: Server, id: string) => post<Machine>(s, `/v1/machines/${m(id)}/stop`);
export const start = (s: Server, id: string) => post<Machine>(s, `/v1/machines/${m(id)}/start`);
/** Irreversible. confirmSelf is needed when the machine is the server answering. */
export const terminate = (s: Server, id: string, confirmSelf = false) =>
  post<Machine>(s, `/v1/machines/${m(id)}/terminate`, confirmSelf ? { confirm_self: true } : {});
/** Drops the row. Refused while the machine may still bill, unless force. */
export const forget = (s: Server, id: string, force = false) => del<void>(s, `/v1/machines/${m(id)}${force ? '?force=1' : ''}`);
/** The machine's URL and token; only for a ready machine that is not open. */
export const link = (s: Server, id: string) => post<MachineLink>(s, `/v1/machines/${m(id)}/link`);
export const logs = (s: Server, id: string, tail = 200) => get<MachineLogs>(s, `/v1/machines/${m(id)}/logs?tail=${tail}`);

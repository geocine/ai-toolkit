// GPU servers the studio's saved servers track (GET /v1/machines on each),
// and linking a ready one as an ordinary Server (PLAN §11, auto-link).
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, trimBase, type Server } from './api';
import { link, machines, providers } from './provision';
import type { Machine, MachineLink, MachineStatus, Provider, ProviderKind } from './provisionTypes';
import { load, save, uid } from './store';

/** A machine and the saved server (its owner, which holds the provider key) that tracks it. */
export interface Tracked {
  machine: Machine;
  owner: Server;
  provider: Provider | null;
}

const transitional: MachineStatus[] = ['creating', 'starting', 'running', 'stopping', 'terminating'];
export const isTransitional = (m: Machine) => transitional.includes(m.status);

const labels: Record<MachineStatus, string> = {
  creating: 'Renting',
  starting: 'Starting',
  running: 'Starting server',
  ready: 'Ready',
  stopping: 'Stopping',
  stopped: 'Stopped',
  terminating: 'Terminating',
  terminated: 'Terminated',
  failed: 'Failed',
  unknown: 'Unknown',
};
export const stateLabel = (s: MachineStatus) => labels[s] ?? s;

/** The job state whose StatePill colour fits a machine status. */
export function pillState(s: MachineStatus): string {
  switch (s) {
    case 'ready':
      return 'completed';
    case 'creating':
    case 'starting':
    case 'running':
      return 'running';
    case 'stopping':
    case 'terminating':
    case 'stopped':
      return 'stopped';
    case 'failed':
      return 'failed';
    default:
      return 'queued';
  }
}

/** A saved server that does not answer GET /v1/machines within this is skipped for the round. */
const POLL_TIMEOUT = 8_000;

export const providerNames: Record<ProviderKind, string> = { runpod: 'RunPod', vast: 'Vast.ai', quickpod: 'QuickPod' };

/**
 * Polls GET /v1/machines?active=1 on every saved server: every 5 s while a
 * machine is changing state, otherwise every 30 s, and only while the page is
 * visible. Each server gets 8 s and lands on its own. Servers that answer 404
 * (older images) or cannot be reached are skipped quietly. A machine listed by
 * two servers shows once, under the one that is not itself a rented machine.
 */
export function useMachines(servers: Server[]) {
  const [lists, setLists] = useState<Record<string, Machine[]>>({});
  const [provs, setProvs] = useState<Record<string, Provider[]>>({});
  const provsRef = useRef(provs);
  provsRef.current = provs;
  const serversRef = useRef(servers);
  serversRef.current = servers;
  const key = servers.map(s => `${s.id}|${s.base}|${s.token}`).join('\n');

  const poll = useCallback(async () => {
    const list = serversRef.current;
    const ids = new Set(list.map(s => s.id));
    setLists(v => (Object.keys(v).every(id => ids.has(id)) ? v : Object.fromEntries(Object.entries(v).filter(([id]) => ids.has(id)))));
    // each server lands on its own, so one that hangs never holds back the rest
    await Promise.all(
      list.map(async s => {
        let ms: Machine[];
        try {
          ms = await machines(s, true, AbortSignal.timeout(POLL_TIMEOUT));
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 404)) return; // keep what we had
          ms = [];
        }
        setLists(v => ({ ...v, [s.id]: ms }));
        // labels, console links and capabilities, once per server that has machines
        if (!ms.length || provsRef.current[s.id]) return;
        providers(s)
          .then(p => setProvs(v => ({ ...v, [s.id]: p })))
          .catch(() => {});
      }),
    );
  }, []);
  // one poll at a time: a loop restarted mid-poll waits for it instead of starting another
  const inflight = useRef<Promise<void> | null>(null);
  const pollOnce = useCallback(() => (inflight.current ??= poll().finally(() => (inflight.current = null))), [poll]);

  const items = useMemo(() => {
    const order = [...servers].sort((a, b) => Number(!!a.machine) - Number(!!b.machine));
    const seen = new Set<string>();
    const out: Tracked[] = [];
    for (const s of order) {
      for (const machine of lists[s.id] ?? []) {
        if (seen.has(machine.id)) continue;
        seen.add(machine.id);
        out.push({ machine, owner: s, provider: provs[s.id]?.find(p => p.kind === machine.provider) ?? null });
      }
    }
    return out.sort((a, b) => b.machine.created_at.localeCompare(a.machine.created_at));
  }, [servers, lists, provs]);

  // a change between busy and idle restarts the loop at the new pace
  const busy = items.some(t => isTransitional(t.machine));
  useEffect(() => {
    let stopped = false;
    let running = false; // this loop's own tick, so a tab switch never starts a second chain
    let t: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      if (running) return;
      running = true;
      try {
        if (document.visibilityState === 'visible') await pollOnce();
      } finally {
        running = false;
      }
      clearTimeout(t);
      if (!stopped) t = setTimeout(tick, busy ? 5_000 : 30_000);
    };
    const onVisible = () => {
      if (document.visibilityState !== 'visible' || running) return;
      clearTimeout(t);
      tick();
    };
    tick();
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      stopped = true;
      clearTimeout(t);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [key, pollOnce, busy]);

  /** Puts a machine the page just changed in place, before the next poll. */
  const put = useCallback((ownerId: string, m: Machine) => {
    setLists(v => {
      const cur = v[ownerId] ?? [];
      const next = cur.some(x => x.id === m.id) ? cur.map(x => (x.id === m.id ? m : x)) : [m, ...cur];
      return { ...v, [ownerId]: next };
    });
  }, []);
  const drop = useCallback((id: string) => {
    setLists(v => Object.fromEntries(Object.entries(v).map(([k, ms]) => [k, ms.filter(m => m.id !== id)])));
  }, []);

  return { items, reload: poll, put, drop };
}

/**
 * How the studio reaches a linked machine. Direct when the machine is https or
 * this page is plain http (no mixed content); otherwise through the
 * owner's relay with the owner's token.
 */
export function linkVia(owner: Server, l: MachineLink): { base: string; token: string; via: 'direct' | 'relay' } {
  if (l.secure || window.location.protocol === 'http:') return { base: trimBase(l.base_url), token: l.token, via: 'direct' };
  return { base: trimBase(owner.base) + l.relay_path, token: owner.token, via: 'relay' };
}

const LINKED = 'aitk.studio.linked';
const linking = new Set<string>();

/** Machines linked once stay linked, so a server removed by hand is not re-added. */
export const wasLinked = (id: string) => load<string[]>(LINKED, []).includes(id);

/** Can be linked now: ready, not open, and not linked from this browser before. */
export const canAutoLink = (m: Machine) => m.status === 'ready' && m.problem !== 'open_server' && !wasLinked(m.id) && !linking.has(m.id);

/**
 * Links a ready machine and returns the Server to save, or null when it is
 * not linkable (yet). The id is recorded in aitk.studio.linked.
 */
export async function autoLink(owner: Server, m: Machine): Promise<Server | null> {
  if (!canAutoLink(m)) return null;
  linking.add(m.id);
  try {
    const l = await link(owner, m.id);
    const { base, token, via } = linkVia(owner, l);
    save(LINKED, [...load<string[]>(LINKED, []).filter(x => x !== m.id), m.id]);
    return { id: uid(), name: l.name || m.name, base, token, machine: { id: m.id, control: owner.id, provider: m.provider, via } };
  } catch {
    return null; // tried again at the next poll
  } finally {
    linking.delete(m.id);
  }
}

/** Copies to the clipboard; false when the browser refuses (plain http, old browsers). */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/** "$0.47/h": two decimals, three under a cent. */
export const perHour = (n: number) => `$${n > 0 && n < 0.01 ? n.toFixed(3) : n.toFixed(2)}/h`;

// Servers the studio knows, the active one, and the user's presets.
import { createContext, useContext, useEffect, useMemo, useRef, type ReactNode } from 'react';
import type { Server } from './api';
import { syncPresets, type Local } from './presetSync';
import { builtInPresets, type Preset } from './presets';
import { useStored } from './store';

interface Studio {
  servers: Server[];
  setServers: (s: Server[] | ((p: Server[]) => Server[])) => void;
  active: Server | null;
  setActiveId: (id: string) => void;
  presets: Preset[];
  imported: Preset[];
  /** Adds an imported or copied preset. */
  addPreset: (p: Preset) => void;
  /** Stores an edited preset: a built-in one as an override of the shipped version. */
  savePreset: (p: Preset) => void;
  /** Drops the override of a built-in preset. */
  resetPreset: (id: string) => void;
  /** Deletes an imported preset (here and, at the next sync, on the servers). */
  deletePreset: (id: string) => void;
}

const Ctx = createContext<Studio | null>(null);

export function StudioProvider({ children }: { children: ReactNode }) {
  const [servers, setServers] = useStored<Server[]>('aitk.studio.servers', []);
  const [activeId, setActiveId] = useStored<string>('aitk.studio.active', '');
  const [imported, setImported] = useStored<Preset[]>('aitk.studio.presets', []);
  const [edits, setEdits] = useStored<Record<string, Preset>>('aitk.studio.presetEdits', {});
  const [deleted, setDeleted] = useStored<Record<string, number>>('aitk.studio.presetDeleted', {});
  const [resets, setResets] = useStored<Record<string, number>>('aitk.studio.presetResets', {});

  // presets follow the active server both ways (lib/presetSync.ts)
  const local = useRef<Local>({ imported, edits, deleted, resets });
  local.current = { imported, edits, deleted, resets };
  const activeServer = servers.find(s => s.id === activeId) ?? servers[0] ?? null;
  useEffect(() => {
    if (!activeServer) return;
    let cancelled = false;
    const run = async () => {
      const c = await syncPresets(activeServer, local.current);
      if (cancelled) return;
      if (c.imported) setImported(c.imported);
      if (c.edits) setEdits(c.edits);
      if (c.deleted) setDeleted(c.deleted);
      if (c.resets) setResets(c.resets);
    };
    // soon after a change here; every 30 s and on coming back to the tab for
    // what agents changed on the server
    const t = setTimeout(run, 800);
    const every = setInterval(() => document.visibilityState === 'visible' && run(), 30_000);
    const onVisible = () => document.visibilityState === 'visible' && run();
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      cancelled = true;
      clearTimeout(t);
      clearInterval(every);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [activeServer?.id, activeServer?.base, activeServer?.token, imported, edits, deleted, resets]); // eslint-disable-line react-hooks/exhaustive-deps

  const value = useMemo<Studio>(() => {
    const active = servers.find(s => s.id === activeId) ?? servers[0] ?? null;
    // an edited built-in shows the notes it ships with, then the ones its edits got
    const presets = [
      ...builtInPresets.map(b => {
        const e = edits[b.id];
        return e ? { ...e, notes: [...b.notes, ...e.notes.filter(n => !b.notes.some(s => s.title === n.title))] } : b;
      }),
      ...imported,
    ];
    const savePreset = (p: Preset) => {
      const at = { ...p, updatedAt: Date.now() };
      if (p.builtIn) setEdits(e => ({ ...e, [p.id]: { ...at, edited: true } }));
      else setImported(list => list.map(x => (x.id === p.id ? at : x)));
    };
    const addPreset = (p: Preset) => setImported(list => [...list, { ...p, updatedAt: Date.now() }]);
    const resetPreset = (id: string) => {
      setEdits(e => {
        const { [id]: _, ...rest } = e;
        return rest;
      });
      setResets(r => ({ ...r, [id]: Date.now() }));
    };
    const deletePreset = (id: string) => {
      setImported(list => list.filter(x => x.id !== id));
      setDeleted(d => ({ ...d, [id]: Date.now() }));
    };
    return { servers, setServers, active, setActiveId, presets, imported, addPreset, savePreset, resetPreset, deletePreset };
  }, [servers, activeId, imported, edits, setServers, setActiveId, setImported, setEdits, setDeleted, setResets]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStudio(): Studio {
  const v = useContext(Ctx);
  if (!v) throw new Error('useStudio outside StudioProvider');
  return v;
}

/** The active server, for pages that need one. */
export function useServer(): Server | null {
  return useStudio().active;
}

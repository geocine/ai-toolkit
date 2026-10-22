// Servers the studio knows, the active one, and the user's presets.
import { createContext, useContext, useMemo, type ReactNode } from 'react';
import type { Server } from './api';
import { builtInPresets, type Preset } from './presets';
import { useStored } from './store';

interface Studio {
  servers: Server[];
  setServers: (s: Server[] | ((p: Server[]) => Server[])) => void;
  active: Server | null;
  setActiveId: (id: string) => void;
  presets: Preset[];
  imported: Preset[];
  setImported: (p: Preset[] | ((p: Preset[]) => Preset[])) => void;
  /** Stores an edited preset: a built-in one as an override of the shipped version. */
  savePreset: (p: Preset) => void;
  /** Drops the override of a built-in preset. */
  resetPreset: (id: string) => void;
}

const Ctx = createContext<Studio | null>(null);

export function StudioProvider({ children }: { children: ReactNode }) {
  const [servers, setServers] = useStored<Server[]>('aitk.studio.servers', []);
  const [activeId, setActiveId] = useStored<string>('aitk.studio.active', '');
  const [imported, setImported] = useStored<Preset[]>('aitk.studio.presets', []);
  const [edits, setEdits] = useStored<Record<string, Preset>>('aitk.studio.presetEdits', {});
  const value = useMemo<Studio>(() => {
    const active = servers.find(s => s.id === activeId) ?? servers[0] ?? null;
    const presets = [...builtInPresets.map(b => edits[b.id] ?? b), ...imported];
    const savePreset = (p: Preset) => {
      if (p.builtIn) setEdits(e => ({ ...e, [p.id]: { ...p, edited: true } }));
      else setImported(list => list.map(x => (x.id === p.id ? p : x)));
    };
    const resetPreset = (id: string) =>
      setEdits(e => {
        const { [id]: _, ...rest } = e;
        return rest;
      });
    return { servers, setServers, active, setActiveId, presets, imported, setImported, savePreset, resetPreset };
  }, [servers, activeId, imported, edits, setServers, setActiveId, setImported, setEdits]);
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

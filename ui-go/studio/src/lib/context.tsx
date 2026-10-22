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
}

const Ctx = createContext<Studio | null>(null);

export function StudioProvider({ children }: { children: ReactNode }) {
  const [servers, setServers] = useStored<Server[]>('aitk.studio.servers', []);
  const [activeId, setActiveId] = useStored<string>('aitk.studio.active', '');
  const [imported, setImported] = useStored<Preset[]>('aitk.studio.presets', []);
  const value = useMemo<Studio>(() => {
    const active = servers.find(s => s.id === activeId) ?? servers[0] ?? null;
    return { servers, setServers, active, setActiveId, presets: [...builtInPresets, ...imported], imported, setImported };
  }, [servers, activeId, imported, setServers, setActiveId, setImported]);
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

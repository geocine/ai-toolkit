// Keeps the studio's presets and a server's /v1/presets in step, so MCP
// agents on that server can use them and presets an agent writes reach the
// studio. The studio's copy (localStorage) outlives any pod; each server it
// connects to gets the same presets. Per preset the newer updatedAt wins; a
// deletion is a tombstone with its own time.
import { call, del, get, type Items, type JobConfig, type Server } from './api';
import { builtInPresets, normalizePreset, type Preset } from './presets';

type Remote = Partial<Preset> & { id: string; base: JobConfig; updated_at: number; deleted?: boolean };

export interface Local {
  imported: Preset[];
  edits: Record<string, Preset>;
  /** imported presets deleted here, by id: when */
  deleted: Record<string, number>;
  /** built-in presets put back to the shipped version, by id: when */
  resets: Record<string, number>;
}

export interface Changes {
  imported?: Preset[];
  edits?: Record<string, Preset>;
  deleted?: Record<string, number>;
  resets?: Record<string, number>;
}

const isBuiltIn = (id: string) => builtInPresets.some(b => b.id === id);

/** What this studio holds for id: its time, and the preset (null for a deletion). */
function localOf(l: Local, id: string): { at: number; preset: Preset | null } | null {
  if (isBuiltIn(id)) {
    const e = l.edits[id];
    if (e) return { at: e.updatedAt ?? 1, preset: { ...e, builtIn: true, edited: true } };
    const b = builtInPresets.find(x => x.id === id)!;
    return { at: l.resets[id] ?? 1, preset: { ...b, builtIn: true, edited: false } };
  }
  const p = l.imported.find(x => x.id === id);
  if (p) return { at: p.updatedAt ?? 1, preset: p };
  if (l.deleted[id]) return { at: l.deleted[id], preset: null };
  return null;
}

const strip = ({ updatedAt: _u, ...p }: Preset) => p;

/** JSON with object keys sorted: the server stores configs with its own key order. */
export function canonical(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`;
  if (v && typeof v === 'object')
    return `{${Object.keys(v)
      .filter(k => (v as Record<string, unknown>)[k] !== undefined)
      .sort()
      .map(k => `${JSON.stringify(k)}:${canonical((v as Record<string, unknown>)[k])}`)
      .join(',')}}`;
  return JSON.stringify(v) ?? 'null';
}

/** One sync with server: pull newer server copies, then push newer local ones. Returns the local changes to store. */
export async function syncPresets(server: Server, l: Local): Promise<Changes> {
  let remote: Remote[];
  try {
    remote = (await get<Items<Remote>>(server, '/v1/presets?deleted=1')).items;
  } catch {
    return {}; // an older server without presets, or offline
  }
  const byId = new Map(remote.map(r => [r.id, r]));
  let imported = l.imported;
  let edits = l.edits;
  let deleted = l.deleted;
  let resets = l.resets;
  let changed = false;

  // pull
  for (const r of remote) {
    const mine = localOf({ imported, edits, deleted, resets }, r.id);
    if (mine && mine.at >= r.updated_at) continue;
    changed = true;
    if (isBuiltIn(r.id)) {
      if (!r.deleted && r.edited) {
        edits = { ...edits, [r.id]: { ...normalizePreset(r), builtIn: true, edited: true, updatedAt: r.updated_at } };
      } else {
        const { [r.id]: _, ...rest } = edits;
        edits = rest;
        resets = { ...resets, [r.id]: r.updated_at };
      }
    } else if (r.deleted) {
      imported = imported.filter(p => p.id !== r.id);
      deleted = { ...deleted, [r.id]: r.updated_at };
    } else {
      const p = { ...normalizePreset(r), builtIn: false, updatedAt: r.updated_at };
      imported = imported.some(x => x.id === r.id) ? imported.map(x => (x.id === r.id ? p : x)) : [...imported, p];
      const { [r.id]: _, ...rest } = deleted;
      deleted = rest;
    }
  }

  // push
  const ids = new Set([...builtInPresets.map(b => b.id), ...imported.map(p => p.id), ...Object.keys(deleted)]);
  for (const id of ids) {
    const mine = localOf({ imported, edits, deleted, resets }, id)!;
    const theirs = byId.get(id);
    // an unedited built-in follows this studio's shipped version, whatever its time
    const stale =
      !!theirs && isBuiltIn(id) && !!mine.preset && !mine.preset.edited && !theirs.edited && !theirs.deleted && canonical(theirs.base) !== canonical(mine.preset.base);
    if (theirs && theirs.updated_at >= mine.at && !stale) continue;
    try {
      if (mine.preset) {
        const { updated_at: _u, deleted: _d, ...body } = { ...strip(mine.preset), updated_at: mine.at, deleted: false };
        await call(server, 'PUT', `/v1/presets/${encodeURIComponent(id)}`, { ...body, updated_at: mine.at });
      } else if (theirs && !theirs.deleted) {
        await del(server, `/v1/presets/${encodeURIComponent(id)}?updated_at=${mine.at}`);
      }
    } catch {
      /* the next sync tries again */
    }
  }
  return changed ? { imported, edits, deleted, resets } : {};
}

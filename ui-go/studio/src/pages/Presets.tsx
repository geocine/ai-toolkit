import { ChevronDown, Download, FileUp, Lightbulb } from 'lucide-react';
import { useRef, useState } from 'react';
import { ErrorNote, NeedServer, PageHeader } from '../components/ui';
import { useStudio } from '../lib/context';
import { presetFromText, presetToYAML, type Preset } from '../lib/presets';
import { href } from '../lib/route';
import { DataPresets } from './DataPresets';

function saveText(name: string, text: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/yaml' }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

export function PresetNotes({ preset }: { preset: Preset }) {
  if (!preset.notes.length) return null;
  return (
    <div className="callout callout-info notes">
      <Lightbulb className="callout-icon" aria-hidden />
      <ul className="callout-body">
        {preset.notes.map(n => (
          <li key={n.title}>
            <strong>{n.title}.</strong> <span dangerouslySetInnerHTML={{ __html: md(n.body) }} />
          </li>
        ))}
      </ul>
    </div>
  );
}

// `code` only; notes are our own text
const md = (s: string) => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!).replace(/`([^`]+)`/g, '<code>$1</code>');

function PresetTabs({ tab }: { tab: 'settings' | 'data' }) {
  return (
    <nav className="tabs" aria-label="Preset kinds">
      <a className={tab === 'settings' ? 'tab is-active' : 'tab'} href={href({ page: 'presets' })} aria-current={tab === 'settings' ? 'page' : undefined}>
        Settings
      </a>
      <a className={tab === 'data' ? 'tab is-active' : 'tab'} href={href({ page: 'presets', tab: 'data' })} aria-current={tab === 'data' ? 'page' : undefined}>
        Data
      </a>
    </nav>
  );
}

export function Presets({ tab = 'settings' }: { tab?: 'settings' | 'data' }) {
  const { active } = useStudio();
  if (tab === 'data') {
    if (!active) return <NeedServer title="Presets" />;
    return (
      <>
        <PageHeader
          title="Presets"
          description="A data preset is a whole run in one zip: the dataset with its caches, checkpoints, optimizer state and samples. Load one onto any server and training continues where it stopped."
        />
        <PresetTabs tab="data" />
        <DataPresets key={active.id} server={active} />
      </>
    );
  }
  return <SettingsPresets />;
}

function SettingsPresets() {
  const { presets, setImported } = useStudio();
  const [error, setError] = useState<unknown>(null);
  const file = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState<string | null>(presets[0]?.id ?? null);

  const onFiles = async (files: FileList | null) => {
    setError(null);
    for (const f of Array.from(files ?? [])) {
      try {
        const p = presetFromText(await f.text(), f.name);
        setImported(list => [...list, p]);
        setOpen(p.id);
      } catch (e) {
        setError(new Error(`${f.name}: ${e instanceof Error ? e.message : e}`));
      }
    }
    if (file.current) file.current.value = '';
  };

  return (
    <>
      <PageHeader
        title="Presets"
        description="A preset is a full ai-toolkit job config plus the handful of knobs you change per run. Import any ai-toolkit YAML or JSON; the two settings traps found in your Krea files are checked on everything imported."
        actions={
          <>
            <input
              ref={file}
              type="file"
              accept=".yml,.yaml,.json"
              multiple
              className="visually-hidden"
              aria-label="Import ai-toolkit config files"
              onChange={e => onFiles(e.target.files)}
            />
            <button className="btn btn-primary" onClick={() => file.current?.click()}>
              <FileUp />
              Import config
            </button>
          </>
        }
      />
      <PresetTabs tab="settings" />
      <ErrorNote error={error} />
      <div className="preset-list">
        {presets.map(p => {
          const proc = p.base.config.process[0] ?? {};
          const isOpen = open === p.id;
          return (
            <article key={p.id} className={isOpen ? 'card preset is-open' : 'card preset'}>
              <button className="preset-head" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : p.id)}>
                <span className="preset-title">
                  <span className="preset-name">{p.name}</span>
                  <span className={p.builtIn ? 'badge tone-quiet' : 'badge tone-accent'}>{p.builtIn ? 'Built-in' : 'Imported'}</span>
                </span>
                <span className="preset-desc">{p.description}</span>
                <ChevronDown className="preset-chevron" aria-hidden />
              </button>
              {isOpen && (
                <div className="preset-body">
                  <dl className="facts">
                    <div className="fact-wide">
                      <dt>Model</dt>
                      <dd className="mono">{proc.model?.name_or_path ?? '–'}</dd>
                    </div>
                    <div>
                      <dt>Arch</dt>
                      <dd className="mono">{proc.model?.arch ?? '–'}</dd>
                    </div>
                    <div>
                      <dt>Network</dt>
                      <dd>
                        {proc.network?.type ?? 'lora'} · rank {p.defaults.rank}
                        {proc.network?.type === 'lokr' ? ` · factor ${p.defaults.lokrFactor}` : ''}
                      </dd>
                    </div>
                    <div>
                      <dt>Steps</dt>
                      <dd className="num">
                        {p.defaults.steps.toLocaleString()} · save every {p.defaults.saveEvery}
                      </dd>
                    </div>
                    <div>
                      <dt>Optimizer</dt>
                      <dd className="mono">{p.defaults.optimizer}</dd>
                    </div>
                    <div>
                      <dt>Resolution</dt>
                      <dd className="num">{p.defaults.resolution.join(' + ')}</dd>
                    </div>
                  </dl>
                  <PresetNotes preset={p} />
                  <div className="row">
                    <button className="btn" onClick={() => saveText(`${p.name.replace(/[^\w.-]+/g, '_')}.yml`, presetToYAML(p))}>
                      <Download />
                      Export YAML
                    </button>
                    {!p.builtIn && (
                      <button
                        className="btn btn-danger"
                        onClick={() => {
                          if (confirm(`Delete preset ${p.name}?`)) setImported(list => list.filter(x => x.id !== p.id));
                        }}
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </>
  );
}

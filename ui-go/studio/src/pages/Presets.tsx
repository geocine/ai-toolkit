import { useRef, useState } from 'react';
import { ErrorNote, Section } from '../components/ui';
import { useStudio } from '../lib/context';
import { presetFromText, presetToYAML, type Preset } from '../lib/presets';

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
    <ul className="notes">
      {preset.notes.map(n => (
        <li key={n.title}>
          <strong>{n.title}.</strong> <span dangerouslySetInnerHTML={{ __html: md(n.body) }} />
        </li>
      ))}
    </ul>
  );
}

// `code` only; notes are our own text
const md = (s: string) => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!).replace(/`([^`]+)`/g, '<code>$1</code>');

export function Presets() {
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
    <Section
      kicker="Recipes"
      title="Presets"
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
          <button className="btn btn-ink" onClick={() => file.current?.click()}>
            Import config…
          </button>
        </>
      }
    >
      <p className="lede">
        A preset is a full ai-toolkit job config plus the handful of knobs you change per run. Import any ai-toolkit YAML or
        JSON — the two settings traps found in your Krea files are checked on everything imported.
      </p>
      <ErrorNote error={error} />
      <div className="preset-list">
        {presets.map((p, i) => {
          const proc = p.base.config.process[0] ?? {};
          return (
            <article key={p.id} className={open === p.id ? 'preset preset-open' : 'preset'} style={{ animationDelay: `${i * 60}ms` }}>
              <button className="preset-head" onClick={() => setOpen(open === p.id ? null : p.id)}>
                <span className="preset-tag">{p.builtIn ? 'built-in' : 'imported'}</span>
                <span className="preset-name">{p.name}</span>
                <span className="preset-desc">{p.description}</span>
              </button>
              {open === p.id && (
                <div className="preset-body">
                  <dl className="spec">
                    <dt>Model</dt>
                    <dd className="mono">{proc.model?.name_or_path ?? '–'}</dd>
                    <dt>Arch</dt>
                    <dd className="mono">{proc.model?.arch ?? '–'}</dd>
                    <dt>Network</dt>
                    <dd className="mono">
                      {proc.network?.type ?? 'lora'} · {p.defaults.rank}
                      {proc.network?.type === 'lokr' ? ` · factor ${p.defaults.lokrFactor}` : ''}
                    </dd>
                    <dt>Steps</dt>
                    <dd className="mono">
                      {p.defaults.steps} · save every {p.defaults.saveEvery}
                    </dd>
                    <dt>Optimizer</dt>
                    <dd className="mono">{p.defaults.optimizer}</dd>
                    <dt>Resolution</dt>
                    <dd className="mono">{p.defaults.resolution.join(' + ')}</dd>
                  </dl>
                  <PresetNotes preset={p} />
                  <div className="row">
                    <button className="btn" onClick={() => saveText(`${p.name.replace(/[^\w.-]+/g, '_')}.yml`, presetToYAML(p))}>
                      Export YAML
                    </button>
                    {!p.builtIn && (
                      <button
                        className="btn btn-quiet"
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
    </Section>
  );
}

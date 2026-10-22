// YAML editor for preset configs: CodeMirror 6, loaded only when a preset is
// edited. Colours come from the studio's tokens, so it follows the theme.
import { indentWithTab } from '@codemirror/commands';
import { yaml } from '@codemirror/lang-yaml';
import { HighlightStyle, indentUnit, syntaxHighlighting } from '@codemirror/language';
import { linter, lintGutter, type Diagnostic } from '@codemirror/lint';
import { EditorState, Prec } from '@codemirror/state';
import { EditorView, keymap } from '@codemirror/view';
import { tags as t } from '@lezer/highlight';
import { basicSetup } from 'codemirror';
import { useEffect, useRef } from 'react';
import YAML from 'yaml';

const highlight = HighlightStyle.define([
  { tag: [t.propertyName, t.definition(t.propertyName)], color: 'var(--accent)' },
  { tag: [t.string, t.special(t.string), t.content], color: 'var(--ok)' },
  { tag: [t.number, t.bool, t.null, t.keyword, t.atom], color: 'var(--warn)' },
  { tag: [t.comment, t.lineComment], color: 'var(--text-3)', fontStyle: 'italic' },
  { tag: [t.punctuation, t.separator, t.meta, t.squareBracket, t.brace], color: 'var(--text-3)' },
]);

const theme = EditorView.theme({
  '&': {
    height: 'min(70vh, 640px)',
    border: '1px solid var(--border-strong)',
    borderRadius: 'var(--r-sm)',
    background: 'var(--surface)',
    color: 'var(--text)',
    fontSize: '12.5px',
    overflow: 'hidden',
  },
  '&.cm-focused': { outline: 'none', borderColor: 'var(--accent)', boxShadow: '0 0 0 3px var(--accent-soft)' },
  '.cm-scroller': { fontFamily: 'var(--font-mono)', lineHeight: '1.6' },
  '.cm-content': { caretColor: 'var(--text)', padding: '8px 0' },
  '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--text)' },
  '.cm-gutters': { background: 'var(--surface-2)', color: 'var(--text-3)', border: 'none' },
  '.cm-activeLine': { background: 'color-mix(in srgb, var(--surface-2) 70%, transparent)' },
  '.cm-activeLineGutter': { background: 'var(--border)', color: 'var(--text-2)' },
  '&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection': {
    background: 'color-mix(in srgb, var(--accent) 24%, transparent) !important',
  },
  '.cm-matchingBracket': { background: 'var(--accent-soft)', outline: '1px solid color-mix(in srgb, var(--accent) 40%, transparent)' },
  '.cm-searchMatch': { background: 'color-mix(in srgb, var(--warn) 22%, transparent)' },
  '.cm-foldPlaceholder': { background: 'var(--surface-2)', border: '1px solid var(--border)', color: 'var(--text-2)' },
  '.cm-tooltip': { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--r-sm)', color: 'var(--text)' },
  '.cm-tooltip-autocomplete > ul > li[aria-selected]': { background: 'var(--accent-soft)', color: 'var(--text)' },
  '.cm-diagnostic-error': { borderLeft: 'none' },
  // the search panel: undo the studio's full-width form controls
  '.cm-panels': { background: 'var(--surface-2)', color: 'var(--text)', borderColor: 'var(--border)' },
  '.cm-panel.cm-search': { padding: '6px 8px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '6px' },
  '.cm-panel input:not([type=checkbox])': { width: 'auto', height: '26px', padding: '0 8px', fontSize: '12px' },
  '.cm-panel label': { display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '12px' },
  '.cm-panel .cm-button': { height: '26px', backgroundImage: 'none', background: 'var(--surface)', border: '1px solid var(--border-strong)', borderRadius: 'var(--r-sm)', color: 'var(--text)', fontSize: '12px' },
  '.cm-panel button[name=close]': { position: 'static', marginLeft: 'auto', color: 'var(--text-2)' },
});

// every YAML problem, live, on its line (the save re-checks the whole config)
const yamlLint = linter(
  view => {
    const text = view.state.doc.toString();
    const out: Diagnostic[] = [];
    for (const e of YAML.parseDocument(text).errors) {
      const from = Math.min(e.pos[0], text.length);
      out.push({ from, to: Math.min(Math.max(e.pos[1], from + 1), text.length), severity: 'error', message: e.message.split('\n')[0].replace(/:$/, '') });
    }
    return out;
  },
  { delay: 300 },
);

export default function ConfigEditor({ value, labelledBy, onChange, onSave }: { value: string; labelledBy: string; onChange: (text: string) => void; onSave: () => void }) {
  const host = useRef<HTMLDivElement>(null);
  const cb = useRef({ onChange, onSave });
  cb.current = { onChange, onSave };

  useEffect(() => {
    const view = new EditorView({
      parent: host.current!,
      state: EditorState.create({
        doc: value,
        extensions: [
          Prec.high(keymap.of([{ key: 'Mod-s', preventDefault: true, run: () => (cb.current.onSave(), true) }, indentWithTab])),
          basicSetup,
          yaml(),
          indentUnit.of('  '),
          EditorState.tabSize.of(2),
          syntaxHighlighting(highlight),
          yamlLint,
          lintGutter(),
          theme,
          EditorView.contentAttributes.of({ 'aria-labelledby': labelledBy, spellcheck: 'false' }),
          EditorView.updateListener.of(u => {
            if (u.docChanged) cb.current.onChange(u.state.doc.toString());
          }),
        ],
      }),
    });
    return () => view.destroy();
    // the editor owns the text after the first render
  }, []);

  return <div ref={host} className="config-editor" />;
}

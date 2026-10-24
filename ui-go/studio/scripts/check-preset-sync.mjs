// Checks that an unedited built-in preset is not pushed again when the server
// only stored its config with another key order (studio/src/lib/presetSync.ts):
// the server answers Go-marshalled JSON with sorted keys.
//
//   node studio/scripts/check-preset-sync.mjs
//
// presetSync.ts is bundled to a temporary ES module with the esbuild the studio
// already installs (through vite). Exit status 1 on any failure.

import { mkdtempSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const studio = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dir = mkdtempSync(join(tmpdir(), 'aitk-preset-sync-'));
const out = join(dir, 'presetSync.mjs');
try {
  createRequire(join(studio, 'package.json'))('esbuild').buildSync({
    entryPoints: [join(studio, 'src', 'lib', 'presetSync.ts')], bundle: true, format: 'esm', platform: 'neutral', target: 'es2022', outfile: out, logLevel: 'error',
  });
  const { canonical } = await import(pathToFileURL(out).href);
  const failures = [];
  const eq = (a, b, want) => {
    if ((canonical(a) === canonical(b)) !== want) failures.push(`${JSON.stringify(a)} vs ${JSON.stringify(b)}: want ${want ? 'equal' : 'different'}`);
  };
  eq({ b: 1, a: { d: [1, { y: 2, x: 1 }], c: null } }, { a: { c: null, d: [1, { x: 1, y: 2 }] }, b: 1 }, true);
  eq({ a: 1, u: undefined }, { a: 1 }, true);
  eq({ a: [1, 2] }, { a: [2, 1] }, false);
  eq({ a: 1 }, { a: '1' }, false);
  eq({ a: { b: 1 } }, { a: { b: 1, c: 0 } }, false);
  if (failures.length) {
    console.error(failures.join('\n'));
    process.exitCode = 1;
  } else console.log('preset sync: canonical compare ok');
} finally {
  rmSync(dir, { recursive: true, force: true });
}

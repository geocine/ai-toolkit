// Checks that the studio's box filter and dHash (studio/src/lib/photos.ts)
// give the same bits as the server's (internal/photocheck), against the
// golden file the Go tests write: internal/photocheck/testdata/dhash_golden.json.
//
//   node studio/scripts/check-dhash.mjs
//
// photos.ts is bundled to a temporary ES module with the esbuild the studio
// already installs (through vite); nothing else is needed. Exit status 1 on
// any difference.

import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const studio = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const golden = JSON.parse(readFileSync(join(studio, '..', 'internal', 'photocheck', 'testdata', 'dhash_golden.json'), 'utf8'));
const entry = join(studio, 'src', 'lib', 'photos.ts');

const dir = mkdtempSync(join(tmpdir(), 'aitk-dhash-'));
const out = join(dir, 'photos.mjs');
try {
  await bundle(entry, out);
  const { boxDown, dhash, thumb32 } = await import(pathToFileURL(out).href);
  const failures = check({ boxDown, dhash, thumb32 });
  if (failures.length) {
    for (const f of failures) console.error('check-dhash: ' + f);
    process.exitCode = 1;
  } else {
    console.log(`check-dhash: ok (${golden.box.length} box filters, dhash ${golden.dhash}, dhash_flip ${golden.dhash_flip}, thumb32)`);
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}

function check({ boxDown, dhash, thumb32 }) {
  const failures = [];
  const { w, h } = golden;
  if (!Array.isArray(golden.grid) || golden.grid.length !== w * h) return [`golden grid is not ${w} x ${h}`];
  const grid = Float32Array.from(golden.grid);
  for (const b of golden.box) {
    const got = boxDown(grid, w, h, b.w, b.h);
    const want = Float32Array.from(b.out); // the Go float32 values, as JSON shortest decimals
    if (got.length !== want.length) {
      failures.push(`boxDown ${w}x${h} -> ${b.w}x${b.h}: ${got.length} values, want ${want.length}`);
      continue;
    }
    const bad = got.findIndex((v, i) => v !== want[i]);
    if (bad >= 0) failures.push(`boxDown ${w}x${h} -> ${b.w}x${b.h}: value ${bad} is ${got[bad]}, want ${want[bad]}`);
  }
  const d = dhash(grid, w, h);
  if (d.h !== golden.dhash) failures.push(`dhash is ${d.h}, want ${golden.dhash}`);
  if (d.flip !== golden.dhash_flip) failures.push(`dhash_flip is ${d.flip}, want ${golden.dhash_flip}`);
  if (golden.thumb32 !== undefined && thumb32(grid, w, h) !== golden.thumb32) failures.push('thumb32 differs from the golden base64');
  return failures;
}

/** Bundles entry to out with esbuild's API, or its CLI through npx when the package cannot be loaded. */
async function bundle(entry, out) {
  const opts = { entryPoints: [entry], bundle: true, format: 'esm', platform: 'neutral', target: 'es2022', outfile: out, logLevel: 'error' };
  let esbuild = null;
  try {
    esbuild = createRequire(join(studio, 'package.json'))('esbuild');
  } catch {
    esbuild = null;
  }
  if (esbuild) {
    esbuild.buildSync(opts);
    return;
  }
  const r = spawnSync('npx', ['--no-install', 'esbuild', entry, '--bundle', '--format=esm', '--platform=neutral', '--target=es2022', `--outfile=${out}`, '--log-level=error'], {
    cwd: studio,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });
  if (r.status !== 0) throw new Error('esbuild failed; run npm install in studio/');
}

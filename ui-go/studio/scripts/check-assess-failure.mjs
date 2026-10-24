// Checks what the New run > Images step shows when the live photo check
// (POST /v1/photos/assess) fails: assessFailure in studio/src/lib/photos.ts.
// A 404 is an older server with no photo checks, where uploading still works,
// so it shows nothing; every other failure is shown as it came.
//
//   node studio/scripts/check-assess-failure.mjs
//
// photos.ts and api.ts are bundled together to a temporary ES module with the
// esbuild the studio already installs (through vite), so ApiError is the same
// class on both sides. Exit status 1 on any difference.

import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const studio = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const lib = join(studio, 'src', 'lib');

const dir = mkdtempSync(join(tmpdir(), 'aitk-assess-'));
const entry = join(dir, 'entry.ts');
const out = join(dir, 'entry.mjs');
try {
  const spec = p => JSON.stringify(join(lib, p).replace(/\\/g, '/'));
  writeFileSync(entry, `export { assessFailure } from ${spec('photos.ts')};\nexport { ApiError } from ${spec('api.ts')};\n`);
  await bundle(entry, out);
  const { assessFailure, ApiError } = await import(pathToFileURL(out).href);
  const failures = check({ assessFailure, ApiError });
  if (failures.length) {
    for (const f of failures) console.error('check-assess-failure: ' + f);
    process.exitCode = 1;
  } else {
    console.log('check-assess-failure: ok (404 quiet; 400, 500, 0 and a network TypeError shown)');
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}

function check({ assessFailure, ApiError }) {
  const failures = [];
  const older = new ApiError(404, 'not_found', 'no route /v1/photos/assess');
  if (assessFailure(older) !== null) failures.push('a 404 (an older server) is shown; want nothing');
  for (const [status, code] of [
    [400, 'bad_request'],
    [500, 'internal'],
    [0, 'network'],
  ]) {
    const e = new ApiError(status, code, `failed with ${status}`);
    if (assessFailure(e) !== e) failures.push(`a ${status} is not shown as it came`);
  }
  const drop = new TypeError('Failed to fetch'); // a brief network drop
  if (assessFailure(drop) !== drop) failures.push('a network TypeError is not shown as it came');
  if (assessFailure(null) !== null) failures.push('no error turns into one');
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

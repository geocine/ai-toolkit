// Checks the launch price cap (studio/src/lib/priceCap.ts): never below the
// quote, at most 2 % plus a cent above it. Prices are real dph_total values
// from the Vast mock fixtures and sub-cent QuickPod and RunPod quotes.
//
//   node studio/scripts/check-price-cap.mjs
//
// priceCap.ts is bundled to a temporary ES module with the esbuild the studio
// already installs (through vite). Exit status 1 on any failure.

import { mkdtempSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const studio = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dir = mkdtempSync(join(tmpdir(), 'aitk-price-cap-'));
const out = join(dir, 'priceCap.mjs');
try {
  createRequire(join(studio, 'package.json'))('esbuild').buildSync({
    entryPoints: [join(studio, 'src', 'lib', 'priceCap.ts')], bundle: true, format: 'esm', platform: 'neutral', target: 'es2022', outfile: out, logLevel: 'error',
  });
  const { priceCap } = await import(pathToFileURL(out).href);
  const prices = [0.19074074, 0.12222222, 0.1813, 0.181, 0.052, 0.1003, 0.0001, 0.01, 0.1, 0.18, 0.2449, 0.25, 0.34, 0.45, 0.69, 1, 1.99, 2.4999, 3.2];
  for (let i = 1; i <= 5000; i++) prices.push(i / 1000, i / 10000 + 0.00003);
  const failures = [];
  for (const p of prices) {
    const cap = priceCap(p);
    if (!(cap >= p)) failures.push(`price ${p}: cap ${cap} is below it`);
    if (cap > p * 1.02 + 0.01 + 1e-9) failures.push(`price ${p}: cap ${cap} is more than 2 % and a cent above it`);
    if (Math.abs(cap * 100 - Math.round(cap * 100)) > 1e-6) failures.push(`price ${p}: cap ${cap} is not whole cents`);
  }
  if (priceCap(0.19074074) !== 0.2) failures.push(`priceCap(0.19074074) = ${priceCap(0.19074074)}, want 0.2`);
  if (priceCap(0.45) !== 0.46) failures.push(`priceCap(0.45) = ${priceCap(0.45)}, want 0.46`);
  if (failures.length) {
    for (const f of failures.slice(0, 20)) console.error('check-price-cap: ' + f);
    process.exitCode = 1;
  } else {
    console.log(`check-price-cap: ok (${prices.length} prices)`);
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}

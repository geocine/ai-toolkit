import path from 'path';

function isWithinAllowedDirs(resolvedPath: string, allowedDirs: string[]) {
  return allowedDirs.some(allowedDir => {
    const resolvedAllowedDir = path.resolve(allowedDir);
    return resolvedPath === resolvedAllowedDir || resolvedPath.startsWith(resolvedAllowedDir + path.sep);
  });
}

/**
 * Resolve a requested (already decoded) file path to an absolute path that is
 * still under one of the allowed roots, or null when it is not.
 *
 * A path that arrives without its leading root (e.g. a proxy decoded `%2F` and
 * collapsed the resulting `//`, so `/mnt/out/x.png` shows up as `mnt/out/x.png`)
 * is retried against each allowed root's filesystem root.
 *
 * Mirrors src/server/safePath.ts used by the Next.js catch-all routes.
 */
export function resolveAllowedPath(filePath: string, allowedDirs: string[]) {
  const candidates = [filePath];

  if (!path.isAbsolute(filePath)) {
    for (const allowedDir of allowedDirs) {
      const root = path.parse(path.resolve(allowedDir)).root;
      if (root) {
        candidates.push(path.join(root, filePath));
      }
    }
  }

  for (const candidate of [...new Set(candidates)]) {
    const resolved = path.resolve(candidate);
    if (isWithinAllowedDirs(resolved, allowedDirs)) {
      return resolved;
    }
  }

  return null;
}

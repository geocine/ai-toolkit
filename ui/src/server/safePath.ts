import path from 'path';

export type CatchAllPathParam = string | string[];

function isWithinAllowedDirs(resolvedPath: string, allowedDirs: string[]) {
  return allowedDirs.some(allowedDir => {
    const resolvedAllowedDir = path.resolve(allowedDir);
    return resolvedPath === resolvedAllowedDir || resolvedPath.startsWith(resolvedAllowedDir + path.sep);
  });
}

export function resolveAllowedCatchAllPath(param: CatchAllPathParam, allowedDirs: string[]) {
  const parts = Array.isArray(param) ? param : [param];
  const decodedParts = parts.map(part => decodeURIComponent(part));
  const decodedPath = decodedParts.length === 1 ? decodedParts[0] : decodedParts.join(path.sep);

  const candidates = [decodedPath];

  if (!path.isAbsolute(decodedPath)) {
    for (const allowedDir of allowedDirs) {
      const root = path.parse(allowedDir).root;
      if (root) {
        candidates.push(path.join(root, decodedPath));
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

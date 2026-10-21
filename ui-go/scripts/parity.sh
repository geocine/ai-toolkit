#!/usr/bin/env bash
# Diff the Go API against upstream's Next server: same requests, same fixtures,
# a fresh database on each side, no queue worker on either.
#
#   ui-go/scripts/parity.sh            # build both, run every step
#   ui-go/scripts/parity.sh -v         # also print matching steps
#   ui-go/scripts/parity.sh -only job  # run steps whose name contains "job"
#
# Two worktrees of HEAD are kept in $PARITY_DIR (default: ../.aitk-parity next
# to the repo) so later runs reuse the Next build; delete the folder to reset.
# The Go server is built from the current working tree, so uncommitted ui-go
# changes are what gets tested.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
WORK=${PARITY_DIR:-$(dirname "$REPO")/.aitk-parity}
PORT_A=${PORT_A:-18801}
PORT_B=${PORT_B:-18802}
HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) WINDOWS=1 ;; *) WINDOWS=0 ;; esac

mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)

ensure_worktree() {
  local dir=$1
  if git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # move it to HEAD; untracked build output (node_modules, .next) stays
    git -C "$dir" checkout -q --detach --force "$HEAD_SHA"
    return
  fi
  rm -rf "$dir"
  git -C "$REPO" worktree prune
  git -C "$REPO" worktree add --detach --force "$dir" "$HEAD_SHA" >/dev/null
}

ensure_worktree "$WORK/next"
ensure_worktree "$WORK/go"

# Next side: upstream's UI, rebuilt only when ui/ itself changed (i.e. after a rebase).
UI_TREE=$(git -C "$REPO" rev-parse "$HEAD_SHA:ui")
if [[ ! -f "$WORK/next/ui/.parity-built" || "$(cat "$WORK/next/ui/.parity-built")" != "$UI_TREE" ]]; then
  echo "== building upstream Next UI in $WORK/next/ui"
  (cd "$WORK/next/ui" && npm ci --no-audit --no-fund && npx prisma generate && npm run build)
  echo "$UI_TREE" > "$WORK/next/ui/.parity-built"
fi

# Fresh state on both sides: empty database from upstream's schema, no job or dataset folders.
for side in next go; do
  rm -rf "$WORK/$side/aitk_db.db"* "$WORK/$side/datasets" "$WORK/$side/output" "$WORK/$side/data" "$WORK/$side/outside-secret.png"
done
(cd "$WORK/next/ui" && npx prisma db push --skip-generate >/dev/null)
cp "$WORK/next/aitk_db.db" "$WORK/go/aitk_db.db"

echo "== building Go server from the working tree"
EXE=""; [[ $WINDOWS == 1 ]] && EXE=.exe
(cd "$REPO/ui-go" && go build -o "$WORK/aitk-server$EXE" .)

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    if [[ $WINDOWS == 1 ]]; then
      taskkill //PID "$(cat /proc/$pid/winpid 2>/dev/null || echo $pid)" //T //F >/dev/null 2>&1 || kill "$pid" 2>/dev/null || true
    else
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

(cd "$WORK/next/ui" && exec node dist/cron/fileServer.js start --port "$PORT_A") > "$WORK/next.log" 2>&1 &
PIDS+=($!)
GO_ROOT="$WORK/go"; [[ $WINDOWS == 1 ]] && GO_ROOT=$(cygpath -w "$GO_ROOT")
(cd "$WORK/go/ui-go" 2>/dev/null || cd "$WORK/go"; AI_TOOLKIT_ROOT="$GO_ROOT" exec "$WORK/aitk-server$EXE" --port "$PORT_B" --no-worker) > "$WORK/go.log" 2>&1 &
PIDS+=($!)

wait_up() {
  for _ in $(seq 1 120); do
    curl -s -o /dev/null "http://127.0.0.1:$1/api/auth" && return 0
    sleep 0.5
  done
  echo "server on :$1 did not start; see $WORK/$2.log" >&2
  exit 1
}
wait_up "$PORT_A" next
wait_up "$PORT_B" go

ROOT_A="$WORK/next"; ROOT_B="$WORK/go"
if [[ $WINDOWS == 1 ]]; then ROOT_A=$(cygpath -w "$ROOT_A"); ROOT_B=$(cygpath -w "$ROOT_B"); fi

cd "$REPO/ui-go"
go run ./cmd/parity -a "http://127.0.0.1:$PORT_A" -root-a "$ROOT_A" \
                    -b "http://127.0.0.1:$PORT_B" -root-b "$ROOT_B" "$@"


# ---- ui-go: appended to upstream's /start.sh in place of its UI launch ----
echo "Starting AI Toolkit (Go server)..."
# Upstream relies on compose mounting these; a bare `docker run` has none, and
# dataset create (a non-recursive mkdir, like upstream) would fail without them.
mkdir -p /app/ai-toolkit/datasets /app/ai-toolkit/output /app/ai-toolkit/data
# The schema belongs to upstream (ui/prisma/schema.prisma). Bring the database
# (possibly a mounted one from an older release) up to date before serving.
cd /app/ai-toolkit/ui && npx prisma db push --skip-generate
cd /app/ai-toolkit/ui-go && exec ./aitk-server --port 8675

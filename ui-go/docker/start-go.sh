
# ---- ui-go: appended to upstream's /start.sh in place of its UI launch ----
echo "Starting AI Toolkit (Go server)..."
# The schema belongs to upstream (ui/prisma/schema.prisma). Bring the database
# (possibly a mounted one from an older release) up to date before serving.
cd /app/ai-toolkit/ui && npx prisma db push --skip-generate
cd /app/ai-toolkit/ui-go && exec ./aitk-server --port 8675

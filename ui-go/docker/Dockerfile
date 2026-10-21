# Fork image: upstream's own image plus the Go server.
#
# The base is upstream's docker/Dockerfile built unchanged (see
# ui-go/scripts/build_and_push.sh), so Python, CUDA, torch, Node and the
# Prisma CLI all follow upstream after a rebase with no edits here. This layer
# only adds the Go binary and swaps the UI launch at the end of upstream's
# /start.sh for it.
#
# Build from the repo root:
#   docker build -f ui-go/docker/Dockerfile --build-arg BASE_IMAGE=<upstream image> .

ARG BASE_IMAGE=aitoolkit-upstream:local

FROM golang:1.26 AS go-build
WORKDIR /src
COPY ui-go/go.mod ui-go/go.sum ./
RUN go mod download
COPY ui-go/ ./
ARG GIT_SHA=
RUN CGO_ENABLED=0 go test ./internal/... && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w -X github.com/geocine/aitoolkit/handlers/v1.buildVersion=${GIT_SHA}" -o /out/aitk-server .

# The frontend: whatever ui-go/web builds to dist/ is served at / by the Go server.
FROM node:22-alpine AS web-build
WORKDIR /web
COPY ui-go/web/package.json ui-go/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui-go/web/ ./
RUN npm run build

FROM ${BASE_IMAGE}

COPY --from=go-build /out/aitk-server /app/ai-toolkit/ui-go/aitk-server
COPY --from=web-build /web/dist /app/ai-toolkit/ui-go/web/dist
COPY ui-go/docker/start-go.sh /tmp/start-go.sh

# Keep upstream's pod setup (ssh, env export) and replace only its final
# "Starting AI Toolkit UI" step. Fail the build if upstream changed that step
# rather than silently starting the Next UI.
RUN sed -i 's/\r$//' /start.sh /tmp/start-go.sh && \
    if ! grep -q '^echo "Starting AI Toolkit UI' /start.sh; then \
        echo 'upstream docker/start.sh no longer has the "Starting AI Toolkit UI" step; update ui-go/docker/Dockerfile' >&2; \
        exit 1; \
    fi && \
    sed -i '/^echo "Starting AI Toolkit UI/,$d' /start.sh && \
    cat /tmp/start-go.sh >> /start.sh && \
    rm /tmp/start-go.sh && \
    chmod +x /start.sh

EXPOSE 8675
WORKDIR /
CMD ["/start.sh"]

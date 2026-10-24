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

# The studio (ui-go/studio), served at /studio/ so one server URL works from a phone.
FROM node:22-alpine AS studio-build
WORKDIR /studio
COPY ui-go/studio/package.json ui-go/studio/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui-go/studio/ ./
RUN npm run build

FROM ${BASE_IMAGE}

COPY --from=go-build /out/aitk-server /app/ai-toolkit/ui-go/aitk-server
COPY --from=web-build /web/dist /app/ai-toolkit/ui-go/web/dist
COPY --from=studio-build /studio/dist /app/ai-toolkit/ui-go/studio/dist
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

# What hosts this image runs on, read by the server's image check before a GPU
# server starts it. ui-go/scripts/build_and_push.sh computes every value from
# the base it just built; a plain docker build leaves them empty, which the
# check treats as missing. Kept last so a new value only rebuilds this layer.
ARG AITK_VERSION= AITK_REVISION= AITK_REVISION_SHORT= AITK_CREATED= AITK_UPSTREAM= AITK_BASE_NAME= \
    AITK_FLAVOR= AITK_CUDA= AITK_DRIVER_MIN= AITK_TORCH= AITK_SM= AITK_PTX= AITK_SM_FLASH_ATTN= AITK_BASE_UNPACKED=
LABEL org.opencontainers.image.title="aitoolkit-go" \
      org.opencontainers.image.version="${AITK_VERSION}" \
      org.opencontainers.image.revision="${AITK_REVISION}" \
      org.opencontainers.image.created="${AITK_CREATED}" \
      org.opencontainers.image.source="https://github.com/geocine/ai-toolkit" \
      org.opencontainers.image.base.name="${AITK_BASE_NAME}" \
      io.github.geocine.aitk.schema="1" \
      io.github.geocine.aitk.server="go" \
      io.github.geocine.aitk.api="1" \
      io.github.geocine.aitk.revision="${AITK_REVISION_SHORT}" \
      io.github.geocine.aitk.upstream="${AITK_UPSTREAM}" \
      io.github.geocine.aitk.flavor="${AITK_FLAVOR}" \
      io.github.geocine.aitk.cuda="${AITK_CUDA}" \
      io.github.geocine.aitk.driver_min="${AITK_DRIVER_MIN}" \
      io.github.geocine.aitk.torch="${AITK_TORCH}" \
      io.github.geocine.aitk.sm="${AITK_SM}" \
      io.github.geocine.aitk.ptx="${AITK_PTX}" \
      io.github.geocine.aitk.sm.flash_attn="${AITK_SM_FLASH_ATTN}" \
      io.github.geocine.aitk.platform="linux/amd64" \
      io.github.geocine.aitk.port="8675" \
      io.github.geocine.aitk.health="/v1/health" \
      io.github.geocine.aitk.size.base_unpacked="${AITK_BASE_UNPACKED}"

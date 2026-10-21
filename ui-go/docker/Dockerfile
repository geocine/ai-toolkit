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
RUN CGO_ENABLED=0 go test ./internal/... && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/aitk-server .

FROM ${BASE_IMAGE}

COPY --from=go-build /out/aitk-server /app/ai-toolkit/ui-go/aitk-server
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

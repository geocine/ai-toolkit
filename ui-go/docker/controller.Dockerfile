# Controller image: the Go server and the studio, no CUDA, no Python (docs/controller.md).
# Build from the repo root: docker build -f ui-go/docker/controller.Dockerfile -t aitk-controller .
FROM golang:1.26 AS go-build
WORKDIR /src
COPY ui-go/go.mod ui-go/go.sum ./
RUN go mod download
COPY ui-go/ ./
ARG GIT_SHA=
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w -X github.com/geocine/aitoolkit/handlers/v1.buildVersion=${GIT_SHA}" -o /out/aitk-server . && \
    mkdir -p /out/data

FROM node:22-alpine AS studio-build
WORKDIR /studio
COPY ui-go/studio/package.json ui-go/studio/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui-go/studio/ ./
RUN npm run build

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=go-build /out/aitk-server /aitk-server
COPY --from=go-build --chown=65532:65532 /out/data /data
COPY --from=studio-build /studio/dist /studio
ENV AITK_CONTROLLER=1 AITK_CONTROLLER_DIR=/data AITK_STUDIO_DIR=/studio
VOLUME /data
EXPOSE 8675
ENTRYPOINT ["/aitk-server", "--host", "0.0.0.0", "--port", "8675"]

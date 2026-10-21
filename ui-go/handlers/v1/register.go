// Package v1 is the app API for headless clients (docs/headless.md):
// system, jobs, queues, datasets, files, settings and keys, plus the OpenAPI
// document for every /v1 endpoint. Remotes, uploads, templates, portable jobs
// and events live in their own packages.
package v1

import (
	"net/http"

	"github.com/geocine/aitoolkit/internal/httpx"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /v1/health", health)
	httpx.Handle(mux, "GET /v1/system", system)
	httpx.Handle(mux, "GET /v1/openapi.json", openapi)

	httpx.Handle(mux, "GET /v1/jobs", listJobs)
	httpx.Handle(mux, "POST /v1/jobs", createJob)
	httpx.Handle(mux, "GET /v1/jobs/{id}", getJob)
	httpx.Handle(mux, "PATCH /v1/jobs/{id}", patchJob)
	httpx.Handle(mux, "DELETE /v1/jobs/{id}", deleteJob)
	httpx.Handle(mux, "POST /v1/jobs/{id}/start", startJob)
	httpx.Handle(mux, "POST /v1/jobs/{id}/stop", stopJob)
	httpx.Handle(mux, "POST /v1/jobs/{id}/kill", killJob)
	httpx.Handle(mux, "POST /v1/jobs/{id}/save_now", flagJob("save_now"))
	httpx.Handle(mux, "POST /v1/jobs/{id}/sample_now", flagJob("sample_now"))
	httpx.Handle(mux, "GET /v1/jobs/{id}/attempts", jobAttempts)
	httpx.Handle(mux, "GET /v1/jobs/{id}/log", jobLog)
	httpx.Handle(mux, "GET /v1/jobs/{id}/loss", jobLoss)
	httpx.Handle(mux, "GET /v1/jobs/{id}/checkpoints", jobCheckpoints)
	httpx.Handle(mux, "GET /v1/jobs/{id}/samples", jobSamples)

	httpx.Handle(mux, "GET /v1/queues", listQueues)
	httpx.Handle(mux, "POST /v1/queues/{gpu_ids}/start", queueAction(true))
	httpx.Handle(mux, "POST /v1/queues/{gpu_ids}/stop", queueAction(false))

	httpx.Handle(mux, "GET /v1/datasets", listDatasets)
	httpx.Handle(mux, "POST /v1/datasets", createDataset)
	httpx.Handle(mux, "GET /v1/datasets/{name}", getDataset)
	httpx.Handle(mux, "DELETE /v1/datasets/{name}", deleteDataset)
	httpx.Handle(mux, "POST /v1/datasets/{name}/files", uploadDatasetFiles)
	httpx.Handle(mux, "POST /v1/datasets/{name}/archive", uploadDatasetArchive)
	httpx.Handle(mux, "PUT /v1/datasets/{name}/captions/{file...}", putCaption)
	httpx.Handle(mux, "DELETE /v1/datasets/{name}/files/{file...}", deleteDatasetFile)

	httpx.Handle(mux, "GET /v1/files", listFiles)
	httpx.Handle(mux, "GET /v1/files/content", getFile)
	httpx.Handle(mux, "PUT /v1/files/content", putFile)
	httpx.Handle(mux, "DELETE /v1/files/content", deleteFile)

	httpx.Handle(mux, "GET /v1/settings", getSettings)
	httpx.Handle(mux, "PATCH /v1/settings", patchSettings)

	httpx.Handle(mux, "GET /v1/keys", listKeys)
	httpx.Handle(mux, "POST /v1/keys", createKey)
	httpx.Handle(mux, "DELETE /v1/keys/{id}", deleteKey)
}

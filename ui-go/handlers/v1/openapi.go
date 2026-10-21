package v1

import (
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"sync"
)

// The OpenAPI 3.1 document for every /v1 endpoint in docs/headless.md
// (including the ones other packages implement), built from the table below
// so it stays compact and consistent.

type op struct {
	method, path, tag, summary string
	query                      []string // "name:type" (type: string|integer|boolean)
	body                       string   // request schema ($ref name, "binary", "multipart") or ""
	status                     int      // success status
	resp                       string   // response schema ($ref name, "items:<Name>", "binary", "sse", "") ; "" = no body
}

var ops = []op{
	{"GET", "/v1/health", "system", "Liveness and versions (public)", nil, "", 200, "Health"},
	{"GET", "/v1/system", "system", "GPUs, CPU, disks", nil, "", 200, "System"},
	{"GET", "/v1/openapi.json", "system", "This document", nil, "", 200, "object"},

	{"GET", "/v1/jobs", "jobs", "List jobs, newest first", []string{"state:string", "type:string"}, "", 200, "items:Job"},
	{"POST", "/v1/jobs", "jobs", "Create a job (draft)", nil, "JobCreate", 201, "Job"},
	{"GET", "/v1/jobs/{id}", "jobs", "Get a job", nil, "", 200, "Job"},
	{"PATCH", "/v1/jobs/{id}", "jobs", "Change a job (409 while queued/running)", nil, "JobPatch", 200, "Job"},
	{"DELETE", "/v1/jobs/{id}", "jobs", "Delete a job (409 while running)", []string{"files:boolean"}, "", 204, ""},
	{"POST", "/v1/jobs/{id}/start", "jobs", "Queue the job and run its GPU queue (resumes from the newest checkpoint)", nil, "", 200, "Job"},
	{"POST", "/v1/jobs/{id}/stop", "jobs", "Stop gracefully", nil, "", 200, "Job"},
	{"POST", "/v1/jobs/{id}/kill", "jobs", "Force stop", nil, "", 200, "Job"},
	{"POST", "/v1/jobs/{id}/save_now", "jobs", "Save a checkpoint on the next step", nil, "", 200, "Job"},
	{"POST", "/v1/jobs/{id}/sample_now", "jobs", "Sample on the next step", nil, "", 200, "Job"},
	{"GET", "/v1/jobs/{id}/attempts", "jobs", "Launches of the job", nil, "", 200, "items:Attempt"},
	{"GET", "/v1/jobs/{id}/log", "jobs", "A window of log.txt by byte offset (negative = from the end)", []string{"offset:integer", "limit:integer"}, "", 200, "LogChunk"},
	{"GET", "/v1/jobs/{id}/loss", "jobs", "Loss history", []string{"key:string", "after_step:integer", "limit:integer", "stride:integer"}, "", 200, "Loss"},
	{"GET", "/v1/jobs/{id}/checkpoints", "jobs", "Saved checkpoints", []string{"hash:boolean"}, "", 200, "items:Checkpoint"},
	{"GET", "/v1/jobs/{id}/samples", "jobs", "Generated samples", nil, "", 200, "items:Sample"},
	{"GET", "/v1/jobs/{id}/sync", "portable", "Portable sync status", nil, "", 200, "SyncStatus"},
	{"POST", "/v1/jobs/{id}/sync", "portable", "Sync the job to its remote now", nil, "", 200, "SyncStatus"},

	{"GET", "/v1/queues", "queues", "GPU queues", nil, "", 200, "items:Queue"},
	{"POST", "/v1/queues/{gpu_ids}/start", "queues", "Run a queue (created if missing)", nil, "", 200, "Queue"},
	{"POST", "/v1/queues/{gpu_ids}/stop", "queues", "Stop a queue", nil, "", 200, "Queue"},

	{"GET", "/v1/datasets", "datasets", "List datasets", nil, "", 200, "items:Dataset"},
	{"POST", "/v1/datasets", "datasets", "Create a dataset", nil, "NameBody", 201, "Dataset"},
	{"GET", "/v1/datasets/{name}", "datasets", "Dataset with a page of items", []string{"cursor:string", "limit:integer"}, "", 200, "DatasetDetail"},
	{"DELETE", "/v1/datasets/{name}", "datasets", "Delete a dataset", nil, "", 204, ""},
	{"POST", "/v1/datasets/{name}/files", "datasets", "Upload files (multipart files[])", nil, "multipart", 200, "items:string"},
	{"POST", "/v1/datasets/{name}/archive", "datasets", "Upload a zip (body or multipart file)", nil, "binary", 200, "ArchiveResult"},
	{"PUT", "/v1/datasets/{name}/captions/{file}", "datasets", "Set a file's caption", nil, "CaptionBody", 200, "DatasetItem"},
	{"DELETE", "/v1/datasets/{name}/files/{file}", "datasets", "Delete a file and its caption", nil, "", 204, ""},

	{"GET", "/v1/files", "files", "List a folder", []string{"root:string", "path:string"}, "", 200, "FileListing"},
	{"GET", "/v1/files/content", "files", "Download a file (Range, ETag; thumb=1 for media)", []string{"root:string", "path:string", "thumb:boolean"}, "", 200, "binary"},
	{"PUT", "/v1/files/content", "files", "Upload a file (atomic)", []string{"root:string", "path:string", "overwrite:boolean"}, "binary", 200, "FileWritten"},
	{"DELETE", "/v1/files/content", "files", "Delete a file", []string{"root:string", "path:string"}, "", 204, ""},

	{"GET", "/v1/remotes", "remotes", "Upload/sync targets", nil, "", 200, "items:Remote"},
	{"POST", "/v1/remotes", "remotes", "Add a remote (credentials are write-only)", nil, "RemoteCreate", 201, "Remote"},
	{"DELETE", "/v1/remotes/{id}", "remotes", "Remove a remote", nil, "", 204, ""},
	{"POST", "/v1/remotes/{id}/test", "remotes", "Check a remote is reachable and writable", nil, "", 200, "RemoteTest"},
	{"POST", "/v1/uploads", "remotes", "Upload a file or checkpoint to a remote", nil, "UploadCreate", 202, "Upload"},
	{"GET", "/v1/uploads", "remotes", "Uploads", []string{"job_id:string", "state:string"}, "", 200, "items:Upload"},
	{"GET", "/v1/uploads/{id}", "remotes", "One upload", nil, "", 200, "Upload"},

	{"GET", "/v1/templates", "templates", "Model archs and the default job config (from upstream's UI)", nil, "", 200, "Templates"},
	{"POST", "/v1/templates/render", "templates", "Build a job config for an arch", nil, "TemplateRender", 200, "RenderedConfig"},

	{"GET", "/v1/portable/jobs", "portable", "Portable jobs on a remote", []string{"remote:string"}, "", 200, "items:PortableJob"},
	{"POST", "/v1/portable/import", "portable", "Import a job from a remote and continue it here", nil, "PortableImport", 201, "Job"},

	{"GET", "/v1/settings", "settings", "Server settings", nil, "", 200, "Settings"},
	{"PATCH", "/v1/settings", "settings", "Change settings (hf_token is write-only)", nil, "SettingsPatch", 200, "Settings"},
	{"GET", "/v1/keys", "keys", "API keys", nil, "", 200, "items:APIKey"},
	{"POST", "/v1/keys", "keys", "Create an API key (the secret is returned once)", nil, "NameBody", 201, "APIKeyCreated"},
	{"DELETE", "/v1/keys/{id}", "keys", "Revoke an API key", nil, "", 204, ""},

	{"GET", "/v1/events", "events", "Stored events after a cursor", []string{"after:integer", "job_id:string", "types:string", "limit:integer"}, "", 200, "EventPage"},
	{"GET", "/v1/events/stream", "events", "Server-Sent Events (honors Last-Event-ID)", []string{"after:integer", "job_id:string", "types:string"}, "", 200, "sse"},
}

func obj(props map[string]any, required ...string) map[string]any {
	m := map[string]any{"type": "object", "properties": props}
	if len(required) > 0 {
		m["required"] = required
	}
	return m
}

func ref(name string) map[string]any { return map[string]any{"$ref": "#/components/schemas/" + name} }

func str() map[string]any       { return map[string]any{"type": "string"} }
func strOrNull() map[string]any { return map[string]any{"type": []string{"string", "null"}} }
func integer() map[string]any   { return map[string]any{"type": "integer"} }
func intOrNull() map[string]any { return map[string]any{"type": []string{"integer", "null"}} }
func boolean() map[string]any   { return map[string]any{"type": "boolean"} }
func num() map[string]any       { return map[string]any{"type": "number"} }
func dateTime() map[string]any  { return map[string]any{"type": "string", "format": "date-time"} }
func arrayOf(item map[string]any) map[string]any {
	return map[string]any{"type": "array", "items": item}
}
func enum(values ...string) map[string]any { return map[string]any{"type": "string", "enum": values} }
func nullable(m map[string]any) map[string]any {
	return map[string]any{"oneOf": []any{m, map[string]any{"type": "null"}}}
}
func anyObject() map[string]any {
	return map[string]any{"type": "object", "additionalProperties": true}
}

var states = []string{"draft", "queued", "starting", "running", "stopping", "stopped", "completed", "failed", "interrupted"}

func schemas() map[string]any {
	attempt := obj(map[string]any{"n": integer(), "started_at": dateTime(), "ended_at": strOrNull(),
		"end_state": strOrNull(), "end_reason": strOrNull(), "start_step": integer(), "end_step": intOrNull(),
		"auto": boolean(), "pid": intOrNull()})
	checkpoint := obj(map[string]any{"name": str(), "path": str(), "step": intOrNull(), "size": integer(),
		"modified": dateTime(), "sha256": str()})
	policy := obj(map[string]any{"auto_resume": boolean(), "max_auto_resumes": integer(),
		"auto_upload_remote": strOrNull(), "sync_remote": strOrNull()})
	policyPatch := obj(map[string]any{"auto_resume": map[string]any{"type": []string{"boolean", "null"}},
		"max_auto_resumes": intOrNull(), "auto_upload_remote": strOrNull(), "sync_remote": strOrNull()})
	dataset := obj(map[string]any{"name": str(), "path": str(), "images": integer(), "captions": integer(),
		"size": integer(), "modified": dateTime()})
	datasetItem := obj(map[string]any{"file": str(), "caption": strOrNull(), "size": integer(),
		"width": integer(), "height": integer()})
	upload := obj(map[string]any{"id": str(), "remote_id": str(), "job_id": strOrNull(), "root": str(), "path": str(),
		"url": strOrNull(), "state": enum("queued", "running", "done", "failed"), "bytes": integer(),
		"bytes_done": integer(), "sha256": strOrNull(), "error": strOrNull(), "created_at": dateTime(), "updated_at": dateTime()})
	lease := nullable(obj(map[string]any{"node_id": str(), "node_label": str(), "heartbeat_at": dateTime(), "expires_at": dateTime()}))
	return map[string]any{
		"Error":  obj(map[string]any{"error": obj(map[string]any{"code": str(), "message": str()}, "code", "message")}, "error"),
		"Health": obj(map[string]any{"ok": boolean(), "version": str(), "upstream_version": strOrNull(), "started_at": dateTime()}),
		"System": obj(map[string]any{"gpus": arrayOf(anyObject()), "has_nvidia_smi": boolean(), "cpu": anyObject(),
			"disks":        arrayOf(obj(map[string]any{"name": str(), "path": str(), "free": integer(), "total": integer()})),
			"toolkit_root": str()}),
		"Job": obj(map[string]any{"id": str(), "name": str(), "type": str(), "ref": strOrNull(), "gpu_ids": str(),
			"state": enum(states...), "status": str(), "info": str(), "step": integer(), "total_steps": intOrNull(),
			"speed": str(), "queue_position": integer(), "pid": intOrNull(), "created_at": dateTime(), "updated_at": dateTime(),
			"config": anyObject(), "policy": policy, "attempts": integer(), "last_attempt": nullable(attempt),
			"latest_checkpoint": nullable(checkpoint)}),
		"JobCreate": obj(map[string]any{"name": str(), "gpu_ids": str(), "config": anyObject(), "type": str(),
			"ref": strOrNull(), "policy": policyPatch}, "name", "config"),
		"JobPatch": obj(map[string]any{"name": str(), "gpu_ids": str(), "config": anyObject(), "type": str(),
			"ref": strOrNull(), "policy": policyPatch}),
		"Attempt":    attempt,
		"Checkpoint": checkpoint,
		"Sample":     obj(map[string]any{"name": str(), "path": str(), "step": intOrNull(), "index": intOrNull(), "modified": dateTime()}),
		"LogChunk":   obj(map[string]any{"offset": integer(), "next_offset": integer(), "size": integer(), "text": str()}),
		"Loss": obj(map[string]any{"keys": arrayOf(str()), "key": strOrNull(),
			"points": arrayOf(obj(map[string]any{"step": integer(), "value": num(), "wall_time": map[string]any{"type": []string{"number", "null"}}}))}),
		"Queue":         obj(map[string]any{"gpu_ids": str(), "running": boolean(), "current_job_id": strOrNull(), "queued": integer()}),
		"Dataset":       dataset,
		"DatasetItem":   datasetItem,
		"DatasetDetail": map[string]any{"allOf": []any{ref("Dataset"), obj(map[string]any{"items": arrayOf(ref("DatasetItem")), "next_cursor": strOrNull()})}},
		"ArchiveResult": obj(map[string]any{"added": integer(), "skipped": arrayOf(str())}),
		"NameBody":      obj(map[string]any{"name": str()}, "name"),
		"CaptionBody":   obj(map[string]any{"caption": str()}, "caption"),
		"FileListing": obj(map[string]any{"root": str(), "path": str(),
			"items": arrayOf(obj(map[string]any{"name": str(), "type": enum("file", "dir"), "size": integer(), "modified": dateTime()}))}),
		"FileWritten": obj(map[string]any{"root": str(), "path": str(), "size": integer(), "sha256": str()}),
		"Remote":      obj(map[string]any{"id": str(), "name": str(), "kind": enum("s3", "hf", "fs"), "config": anyObject(), "created_at": dateTime()}),
		"RemoteCreate": obj(map[string]any{"name": str(), "kind": enum("s3", "hf", "fs"), "config": anyObject(),
			"credentials": anyObject()}, "name", "kind"),
		"RemoteTest":   obj(map[string]any{"ok": boolean(), "detail": str()}),
		"UploadCreate": obj(map[string]any{"remote": str(), "root": str(), "path": str(), "job_id": str(), "checkpoint": str()}, "remote"),
		"Upload":       upload,
		"Templates": obj(map[string]any{"archs": arrayOf(obj(map[string]any{"name": str(), "label": str(), "group": str(), "defaults": anyObject()})),
			"default_config": anyObject()}),
		"TemplateRender": obj(map[string]any{"arch": str(), "name": str(), "datasets": arrayOf(str()), "overrides": anyObject()}, "arch", "name"),
		"RenderedConfig": obj(map[string]any{"config": anyObject()}),
		"PortableJob": obj(map[string]any{"name": str(), "step": integer(), "state": str(), "lease": lease,
			"updated_at": dateTime(), "datasets": arrayOf(str())}),
		"PortableImport": obj(map[string]any{"remote": str(), "name": str(), "gpu_ids": str(), "start": boolean(), "force": boolean()}, "remote", "name"),
		"SyncStatus":     obj(map[string]any{"remote": strOrNull(), "last_sync_at": strOrNull(), "last_error": strOrNull(), "lease": lease}),
		"Settings": obj(map[string]any{"training_folder": str(), "datasets_folder": str(), "data_root": str(), "models_path": str(),
			"hf_token_set": boolean(), "auto_resume_default": boolean(), "max_auto_resumes_default": integer()}),
		"SettingsPatch": obj(map[string]any{"training_folder": str(), "datasets_folder": str(), "data_root": str(),
			"models_path": str(), "hf_token": str(), "auto_resume_default": boolean(), "max_auto_resumes_default": integer()}),
		"APIKey":        obj(map[string]any{"id": str(), "name": str(), "prefix": str(), "created_at": dateTime(), "last_used_at": strOrNull()}),
		"APIKeyCreated": obj(map[string]any{"id": str(), "name": str(), "prefix": str(), "created_at": dateTime(), "key": str()}),
		"Event":         obj(map[string]any{"id": integer(), "type": str(), "job_id": strOrNull(), "at": dateTime(), "data": anyObject()}),
		"EventPage":     obj(map[string]any{"items": arrayOf(ref("Event")), "next_cursor": strOrNull()}),
	}
}

var pathParam = regexp.MustCompile(`\{([^}]+)\}`)

func content(schema string) map[string]any {
	switch {
	case schema == "binary":
		return map[string]any{"application/octet-stream": map[string]any{"schema": map[string]any{"type": "string", "format": "binary"}}}
	case schema == "multipart":
		return map[string]any{"multipart/form-data": map[string]any{"schema": obj(map[string]any{
			"files[]": arrayOf(map[string]any{"type": "string", "format": "binary"})})}}
	case schema == "sse":
		return map[string]any{"text/event-stream": map[string]any{"schema": str()}}
	case schema == "object":
		return map[string]any{"application/json": map[string]any{"schema": anyObject()}}
	case strings.HasPrefix(schema, "items:"):
		inner := strings.TrimPrefix(schema, "items:")
		item := ref(inner)
		if inner == "string" {
			item = str()
		}
		return map[string]any{"application/json": map[string]any{"schema": obj(map[string]any{"items": arrayOf(item)})}}
	}
	return map[string]any{"application/json": map[string]any{"schema": ref(schema)}}
}

func buildOpenAPI() map[string]any {
	paths := map[string]any{}
	for _, o := range ops {
		item, _ := paths[o.path].(map[string]any)
		if item == nil {
			item = map[string]any{}
			paths[o.path] = item
		}
		var params []any
		for _, m := range pathParam.FindAllStringSubmatch(o.path, -1) {
			params = append(params, map[string]any{"name": m[1], "in": "path", "required": true, "schema": str()})
		}
		for _, q := range o.query {
			name, typ, _ := strings.Cut(q, ":")
			params = append(params, map[string]any{"name": name, "in": "query", "schema": map[string]any{"type": typ}})
		}
		responses := map[string]any{}
		ok := map[string]any{"description": "OK"}
		if o.resp != "" {
			ok["content"] = content(o.resp)
		}
		responses[itoa(o.status)] = ok
		responses["default"] = map[string]any{"description": "Error", "content": content("Error")}
		operation := map[string]any{
			"tags": []string{o.tag}, "summary": o.summary,
			"operationId": operationID(o), "responses": responses,
		}
		if len(params) > 0 {
			operation["parameters"] = params
		}
		if o.body != "" {
			operation["requestBody"] = map[string]any{"required": true, "content": content(o.body)}
		}
		if o.path == "/v1/health" {
			operation["security"] = []any{}
		}
		item[strings.ToLower(o.method)] = operation
	}
	return map[string]any{
		"openapi": "3.1.0",
		"info": map[string]any{"title": "ai-toolkit headless server", "version": "1",
			"description": "App API for clients and agents. See ui-go/docs/headless.md."},
		"paths":      paths,
		"components": map[string]any{"schemas": schemas(), "securitySchemes": map[string]any{"bearer": map[string]any{"type": "http", "scheme": "bearer"}}},
		"security":   []any{map[string]any{"bearer": []any{}}},
	}
}

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

func operationID(o op) string {
	parts := []string{strings.ToLower(o.method)}
	for _, seg := range strings.Split(strings.TrimPrefix(o.path, "/v1/"), "/") {
		seg = strings.Trim(seg, "{}")
		seg = strings.ReplaceAll(strings.ReplaceAll(seg, ".", "_"), "-", "_")
		parts = append(parts, seg)
	}
	return strings.Join(parts, "_")
}

var (
	openapiOnce sync.Once
	openapiJSON []byte
)

func openapi(w http.ResponseWriter, r *http.Request) {
	openapiOnce.Do(func() {
		openapiJSON, _ = json.MarshalIndent(buildOpenAPI(), "", "  ")
	})
	w.Header().Set("Content-Type", "application/json")
	w.Write(openapiJSON)
}

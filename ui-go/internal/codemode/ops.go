package codemode

import (
	"fmt"
	"sort"
	"strings"
)

// Param is one input of an operation. Path params are positional arguments
// (in order); query and body params go in one trailing object.
type Param struct {
	Name     string
	Type     string // TypeScript type
	In       string // "path", "query" or "body"
	Optional bool
	Doc      string
}

// Op is one /v1 operation as the `aitk` client exposes it. Ops is the single
// source for both the TypeScript declaration and the sandbox client, so the
// two cannot drift.
type Op struct {
	Group  string
	Name   string
	Method string
	Path   string // with {param} placeholders; {param...} keeps slashes
	Params []Param
	Result string // TypeScript type
	Doc    string
	Text   string // "response": body is returned as text; "request": the text param is the raw body
}

func p(name, typ, in, doc string) Param { return Param{Name: name, Type: typ, In: in, Doc: doc} }
func opt(name, typ, in, doc string) Param {
	return Param{Name: name, Type: typ, In: in, Optional: true, Doc: doc}
}

// Ops mirrors docs/headless.md, Resources.
var Ops = []Op{
	// system
	{Group: "system", Name: "health", Method: "GET", Path: "/v1/health", Result: "Health", Doc: "Server liveness and versions."},
	{Group: "system", Name: "info", Method: "GET", Path: "/v1/system", Result: "SystemInfo", Doc: "GPUs, CPU, disks and the toolkit root."},

	// jobs
	{Group: "jobs", Name: "list", Method: "GET", Path: "/v1/jobs", Result: "Items<Job>", Doc: "Jobs, newest first.",
		Params: []Param{opt("state", "JobState", "query", "filter by client state"), opt("type", "string", "query", "filter by job type (train, caption, inference)")}},
	{Group: "jobs", Name: "create", Method: "POST", Path: "/v1/jobs", Result: "Job",
		Doc: "Create a job (state draft). Build `config` with templates.render.",
		Params: []Param{p("name", "string", "body", "unique job name (also the output folder name)"),
			p("gpu_ids", "string", "body", `GPU index list, e.g. "0" or "0,1"`),
			p("config", "JobConfig", "body", "the ai-toolkit job config object"),
			opt("type", "string", "body", "job type (default train)"),
			opt("ref", "string", "body", "free reference (e.g. dataset path for caption jobs)"),
			opt("policy", "JobPolicy", "body", "headless behavior: auto resume, auto upload, sync")}},
	{Group: "jobs", Name: "get", Method: "GET", Path: "/v1/jobs/{id}", Result: "Job", Doc: "One job with its state, attempts and latest checkpoint.",
		Params: []Param{p("id", "string", "path", "job id")}},
	{Group: "jobs", Name: "update", Method: "PATCH", Path: "/v1/jobs/{id}", Result: "Job", Doc: "Change a job that is not queued or running (409 otherwise).",
		Params: []Param{p("id", "string", "path", "job id"), opt("name", "string", "body", ""), opt("gpu_ids", "string", "body", ""),
			opt("config", "JobConfig", "body", ""), opt("policy", "JobPolicy", "body", "")}},
	{Group: "jobs", Name: "delete", Method: "DELETE", Path: "/v1/jobs/{id}", Result: "null", Doc: "Delete a job that is not running.",
		Params: []Param{p("id", "string", "path", "job id"), opt("files", "boolean", "query", "also delete its output folder")}},
	{Group: "jobs", Name: "start", Method: "POST", Path: "/v1/jobs/{id}/start", Result: "Job",
		Doc:    "Queue the job and make sure its GPU queue runs. A job with checkpoints resumes from the latest one.",
		Params: []Param{p("id", "string", "path", "job id")}},
	{Group: "jobs", Name: "stop", Method: "POST", Path: "/v1/jobs/{id}/stop", Result: "Job", Doc: "Graceful stop: the trainer saves and exits.",
		Params: []Param{p("id", "string", "path", "job id")}},
	{Group: "jobs", Name: "kill", Method: "POST", Path: "/v1/jobs/{id}/kill", Result: "Job", Doc: "Force stop (for a hung trainer).",
		Params: []Param{p("id", "string", "path", "job id")}},
	{Group: "jobs", Name: "saveNow", Method: "POST", Path: "/v1/jobs/{id}/save_now", Result: "Job", Doc: "Save a checkpoint at the next step.",
		Params: []Param{p("id", "string", "path", "job id")}},
	{Group: "jobs", Name: "sampleNow", Method: "POST", Path: "/v1/jobs/{id}/sample_now", Result: "Job", Doc: "Generate samples at the next step.",
		Params: []Param{p("id", "string", "path", "job id")}},
	{Group: "jobs", Name: "attempts", Method: "GET", Path: "/v1/jobs/{id}/attempts", Result: "Items<Attempt>", Doc: "Every launch of the job.",
		Params: []Param{p("id", "string", "path", "job id")}},
	{Group: "jobs", Name: "log", Method: "GET", Path: "/v1/jobs/{id}/log", Result: "LogChunk",
		Doc: "A slice of log.txt by byte offset (negative offset = from the end).",
		Params: []Param{p("id", "string", "path", "job id"), opt("offset", "number", "query", "byte offset; negative counts from the end"),
			opt("limit", "number", "query", "bytes, default 65536, max 1048576")}},
	{Group: "jobs", Name: "loss", Method: "GET", Path: "/v1/jobs/{id}/loss", Result: "LossSeries", Doc: "Logged metric points (loss by default).",
		Params: []Param{p("id", "string", "path", "job id"), opt("key", "string", "query", "metric key, see the keys field"),
			opt("after_step", "number", "query", ""), opt("limit", "number", "query", ""), opt("stride", "number", "query", "")}},
	{Group: "jobs", Name: "checkpoints", Method: "GET", Path: "/v1/jobs/{id}/checkpoints", Result: "Items<Checkpoint>",
		Doc:    "Saved model files, by step (the final step-less file last).",
		Params: []Param{p("id", "string", "path", "job id"), opt("hash", "boolean", "query", "include sha256 (slow on first call)")}},
	{Group: "jobs", Name: "samples", Method: "GET", Path: "/v1/jobs/{id}/samples", Result: "Items<Sample>", Doc: "Generated samples.",
		Params: []Param{p("id", "string", "path", "job id")}},
	{Group: "jobs", Name: "sync", Method: "POST", Path: "/v1/jobs/{id}/sync", Result: "SyncStatus",
		Doc:    "Push the job's resumable state to its sync_remote now (portable jobs).",
		Params: []Param{p("id", "string", "path", "job id")}},
	{Group: "jobs", Name: "syncStatus", Method: "GET", Path: "/v1/jobs/{id}/sync", Result: "SyncStatus", Doc: "Last sync and lease of a portable job.",
		Params: []Param{p("id", "string", "path", "job id")}},

	// queues
	{Group: "queues", Name: "list", Method: "GET", Path: "/v1/queues", Result: "Items<Queue>", Doc: "GPU queues."},
	{Group: "queues", Name: "start", Method: "POST", Path: "/v1/queues/{gpu_ids}/start", Result: "Queue", Doc: "Run a GPU queue.",
		Params: []Param{p("gpu_ids", "string", "path", `e.g. "0"`)}},
	{Group: "queues", Name: "stop", Method: "POST", Path: "/v1/queues/{gpu_ids}/stop", Result: "Queue", Doc: "Stop a GPU queue (stops its running job).",
		Params: []Param{p("gpu_ids", "string", "path", `e.g. "0"`)}},

	// datasets
	{Group: "datasets", Name: "list", Method: "GET", Path: "/v1/datasets", Result: "Items<Dataset>", Doc: "Datasets in the datasets folder."},
	{Group: "datasets", Name: "create", Method: "POST", Path: "/v1/datasets", Result: "Dataset", Doc: "Create an empty dataset folder.",
		Params: []Param{p("name", "string", "body", "safe folder name")}},
	{Group: "datasets", Name: "get", Method: "GET", Path: "/v1/datasets/{name}", Result: "DatasetDetail", Doc: "A dataset and a page of its items with captions.",
		Params: []Param{p("name", "string", "path", ""), opt("cursor", "string", "query", ""), opt("limit", "number", "query", "")}},
	{Group: "datasets", Name: "setCaption", Method: "PUT", Path: "/v1/datasets/{name}/captions/{file...}", Result: "DatasetItem",
		Doc:    "Write the caption of one item.",
		Params: []Param{p("name", "string", "path", ""), p("file", "string", "path", "item file name (may contain /)"), p("caption", "string", "body", "")}},
	{Group: "datasets", Name: "deleteFile", Method: "DELETE", Path: "/v1/datasets/{name}/files/{file...}", Result: "null",
		Doc: "Delete one item and its caption.", Params: []Param{p("name", "string", "path", ""), p("file", "string", "path", "")}},
	{Group: "datasets", Name: "delete", Method: "DELETE", Path: "/v1/datasets/{name}", Result: "null", Doc: "Delete a dataset.",
		Params: []Param{p("name", "string", "path", "")}},

	// files
	{Group: "files", Name: "list", Method: "GET", Path: "/v1/files", Result: "FileListing", Doc: "List a folder inside an allowed root.",
		Params: []Param{p("root", "FileRoot", "query", ""), opt("path", "string", "query", "relative folder, default the root")}},
	{Group: "files", Name: "readText", Method: "GET", Path: "/v1/files/content", Result: "string", Text: "response",
		Doc:    "Read a small text file (max 1 MiB) such as a caption or config.",
		Params: []Param{p("root", "FileRoot", "query", ""), p("path", "string", "query", "")}},
	{Group: "files", Name: "writeText", Method: "PUT", Path: "/v1/files/content", Result: "FileWritten", Text: "request",
		Doc: "Write a text file atomically.",
		Params: []Param{p("root", "FileRoot", "query", ""), p("path", "string", "query", ""), p("text", "string", "body", "file contents"),
			opt("overwrite", "boolean", "query", "")}},
	{Group: "files", Name: "delete", Method: "DELETE", Path: "/v1/files/content", Result: "null", Doc: "Delete a file.",
		Params: []Param{p("root", "FileRoot", "query", ""), p("path", "string", "query", "")}},

	// remotes and uploads
	{Group: "remotes", Name: "list", Method: "GET", Path: "/v1/remotes", Result: "Items<Remote>", Doc: "Upload / sync targets (secrets never returned)."},
	{Group: "remotes", Name: "create", Method: "POST", Path: "/v1/remotes", Result: "Remote", Doc: "Add an R2/S3, Hugging Face or shared-folder remote.",
		Params: []Param{p("name", "string", "body", ""), p("kind", `"s3" | "hf" | "fs"`, "body", ""), p("config", "RemoteConfig", "body", ""),
			opt("credentials", "RemoteCredentials", "body", "write-only")}},
	{Group: "remotes", Name: "delete", Method: "DELETE", Path: "/v1/remotes/{id}", Result: "null", Doc: "Remove a remote.",
		Params: []Param{p("id", "string", "path", "")}},
	{Group: "remotes", Name: "test", Method: "POST", Path: "/v1/remotes/{id}/test", Result: "{ ok: boolean; detail: string }",
		Doc: "Check a remote is reachable and writable.", Params: []Param{p("id", "string", "path", "")}},
	{Group: "uploads", Name: "create", Method: "POST", Path: "/v1/uploads", Result: "Upload",
		Doc: "Upload a file ({remote, root, path}) or a job checkpoint ({remote, job_id, checkpoint}) in the background.",
		Params: []Param{p("remote", "string", "body", "remote id or name"), opt("root", "FileRoot", "body", ""), opt("path", "string", "body", ""),
			opt("job_id", "string", "body", ""), opt("checkpoint", "string", "body", "checkpoint file name")}},
	{Group: "uploads", Name: "list", Method: "GET", Path: "/v1/uploads", Result: "Items<Upload>", Doc: "Uploads, newest first.",
		Params: []Param{opt("job_id", "string", "query", ""), opt("state", "UploadState", "query", "")}},
	{Group: "uploads", Name: "get", Method: "GET", Path: "/v1/uploads/{id}", Result: "Upload", Doc: "One upload.",
		Params: []Param{p("id", "string", "path", "")}},

	// settings and keys
	{Group: "settings", Name: "get", Method: "GET", Path: "/v1/settings", Result: "Settings", Doc: "Server settings."},
	{Group: "settings", Name: "update", Method: "PATCH", Path: "/v1/settings", Result: "Settings", Doc: "Change settings (hf_token is write-only).",
		Params: []Param{opt("training_folder", "string", "body", ""), opt("datasets_folder", "string", "body", ""), opt("data_root", "string", "body", ""),
			opt("models_path", "string", "body", ""), opt("hf_token", "string", "body", ""), opt("auto_resume_default", "boolean", "body", ""),
			opt("max_auto_resumes_default", "number", "body", "")}},
	{Group: "keys", Name: "list", Method: "GET", Path: "/v1/keys", Result: "Items<ApiKey>", Doc: "Active API keys."},
	{Group: "keys", Name: "create", Method: "POST", Path: "/v1/keys", Result: "ApiKey & { key: string }", Doc: "Create a key; `key` is shown once.",
		Params: []Param{p("name", "string", "body", "")}},
	{Group: "keys", Name: "revoke", Method: "DELETE", Path: "/v1/keys/{id}", Result: "null", Doc: "Revoke a key.",
		Params: []Param{p("id", "string", "path", "")}},

	// templates
	{Group: "templates", Name: "list", Method: "GET", Path: "/v1/templates", Result: "TemplateCatalog",
		Doc: "Model archs (from upstream's UI definitions) and the default job config."},
	{Group: "templates", Name: "render", Method: "POST", Path: "/v1/templates/render", Result: "RenderedTemplate",
		Doc: "Build a job config for an arch like the upstream UI does, then apply dotted-path overrides. Read `warnings`.",
		Params: []Param{opt("arch", "string", "body", "arch name from templates.list"), opt("name", "string", "body", "job name"),
			opt("datasets", "string[]", "body", "dataset names"),
			opt("overrides", "Record<string, unknown>", "body", `e.g. {"config.process[0].train.steps": 1500}`)}},

	// portable jobs
	{Group: "portable", Name: "list", Method: "GET", Path: "/v1/portable/jobs", Result: "Items<PortableJob>",
		Doc:    "Jobs whose resumable state is on a remote (to continue on this server).",
		Params: []Param{p("remote", "string", "query", "remote id or name")}},
	{Group: "portable", Name: "import", Method: "POST", Path: "/v1/portable/import", Result: "Job",
		Doc: "Import a portable job from a remote and (with start) continue it here.",
		Params: []Param{p("remote", "string", "body", ""), p("name", "string", "body", "job name on the remote"),
			opt("gpu_ids", "string", "body", ""), opt("start", "boolean", "body", ""),
			opt("force", "boolean", "body", "take over a lease another live server holds")}},

	// events
	{Group: "events", Name: "list", Method: "GET", Path: "/v1/events", Result: "EventPage", Doc: "Events after a cursor (oldest first).",
		Params: []Param{opt("after", "number", "query", "last event id seen"), opt("job_id", "string", "query", ""),
			opt("types", "string[]", "query", "event types to include"), opt("limit", "number", "query", "max 5000")}},
}

// pathParams returns the op's path params in order.
func (o Op) pathParams() []Param {
	var out []Param
	for _, p := range o.Params {
		if p.In == "path" {
			out = append(out, p)
		}
	}
	return out
}

func (o Op) restParams() []Param {
	var out []Param
	for _, p := range o.Params {
		if p.In != "path" {
			out = append(out, p)
		}
	}
	return out
}

func (o Op) signature() string {
	var args []string
	for _, p := range o.pathParams() {
		args = append(args, p.Name+": string")
	}
	if rest := o.restParams(); len(rest) > 0 {
		required := false
		fields := make([]string, 0, len(rest))
		for _, p := range rest {
			q := ""
			if p.Optional {
				q = "?"
			} else {
				required = true
			}
			field := p.Name + q + ": " + p.Type
			if p.Doc != "" {
				field = "/** " + p.Doc + " */ " + field
			}
			fields = append(fields, field)
		}
		q := "?"
		if required {
			q = ""
		}
		args = append(args, "params"+q+": { "+strings.Join(fields, "; ")+" }")
	}
	return fmt.Sprintf("%s(%s): Promise<%s>;", o.Name, strings.Join(args, ", "), o.Result)
}

// Declaration renders the TypeScript declaration of `aitk`. With search, only
// operations whose group, name, path or doc contain it are listed (types are
// always included).
func Declaration(search string) string {
	search = strings.ToLower(strings.TrimSpace(search))
	groups := map[string][]Op{}
	var order []string
	for _, o := range Ops {
		if search != "" {
			hay := strings.ToLower(o.Group + " " + o.Name + " " + o.Path + " " + o.Doc)
			if !strings.Contains(hay, search) && !strings.Contains(strings.ToLower(o.Group+"."+o.Name), search) {
				continue
			}
		}
		if _, ok := groups[o.Group]; !ok {
			order = append(order, o.Group)
		}
		groups[o.Group] = append(groups[o.Group], o)
	}
	var b strings.Builder
	b.WriteString(typesTS)
	b.WriteString("\ndeclare const aitk: {\n")
	for _, g := range order {
		fmt.Fprintf(&b, "  %s: {\n", g)
		for _, o := range groups[g] {
			fmt.Fprintf(&b, "    /** %s (%s %s) */\n    %s\n", o.Doc, o.Method, o.Path, o.signature())
		}
		if g == "jobs" {
			b.WriteString(waitTS)
		}
		b.WriteString("  };\n")
	}
	b.WriteString("};\n")
	b.WriteString(globalsTS)
	return b.String()
}

// OpNames lists group.name for every op, sorted.
func OpNames() []string {
	out := make([]string, 0, len(Ops))
	for _, o := range Ops {
		out = append(out, o.Group+"."+o.Name)
	}
	sort.Strings(out)
	return out
}

const waitTS = `    /** Poll jobs.get until the job reaches one of the states (default: completed, failed, stopped, interrupted). Counts toward the call limit. */
    wait(id: string, options?: { until?: JobState[]; timeout_ms?: number; poll_ms?: number }): Promise<Job>;
`

const globalsTS = `
/** Wait without blocking the server (counts toward the run's time limit). */
declare function sleep(ms: number): Promise<void>;
/** Captured and returned as logs (1 MiB cap). */
declare const console: { log(...a: unknown[]): void; info(...a: unknown[]): void; warn(...a: unknown[]): void; error(...a: unknown[]): void };
/** Thrown by aitk calls that fail: code is the /v1 error code (not_found, conflict, ...), status the HTTP status. */
declare class AitkError extends Error { code: string; status: number; }
`

const typesTS = `type Items<T> = { items: T[]; next_cursor?: string | null };
type JobState = "draft" | "queued" | "starting" | "running" | "stopping" | "stopped" | "completed" | "failed" | "interrupted";
type JobConfig = { job: string; config: { name: string; process: Record<string, any>[] }; meta?: Record<string, any> };
type JobPolicy = { auto_resume?: boolean | null; max_auto_resumes?: number | null; auto_upload_remote?: string | null; sync_remote?: string | null };
type Attempt = { n: number; started_at: string; ended_at: string | null; end_state: string | null; end_reason: string | null; start_step: number; end_step: number | null; auto: boolean; pid: number | null };
type Checkpoint = { name: string; path: string; step: number | null; size: number; modified: string; sha256?: string };
type Sample = { name: string; path: string; step: number | null; index: number | null; modified: string };
type Job = {
  id: string; name: string; type: string; ref: string | null; gpu_ids: string;
  state: JobState; status: string; info: string; step: number; total_steps: number | null; speed: string;
  queue_position: number; pid: number | null; created_at: string; updated_at: string;
  config: JobConfig; policy: JobPolicy; attempts: number; last_attempt: Attempt | null; latest_checkpoint: Checkpoint | null;
};
type LogChunk = { offset: number; next_offset: number; size: number; text: string };
type LossSeries = { keys: string[]; key: string; points: { step: number; value: number; wall_time: number }[] };
type Queue = { gpu_ids: string; running: boolean; current_job_id: string | null; queued: number };
type Dataset = { name: string; path: string; images: number; captions: number; size: number; modified: string };
type DatasetItem = { file: string; caption: string | null; size: number; width?: number; height?: number };
type DatasetDetail = Dataset & { items: DatasetItem[]; next_cursor: string | null };
type FileRoot = "datasets" | "training" | "data" | "models";
type FileListing = { root: FileRoot; path: string; items: { name: string; type: "file" | "dir"; size: number; modified: string }[] };
type FileWritten = { root: FileRoot; path: string; size: number; sha256: string };
type RemoteConfig =
  | { endpoint: string; bucket: string; region?: string; prefix?: string; path_style?: boolean }
  | { repo_id: string; repo_type?: "model" | "dataset"; private?: boolean; path_in_repo?: string; revision?: string }
  | { path: string };
type RemoteCredentials = { access_key_id: string; secret_access_key: string } | { token: string };
type Remote = { id: string; name: string; kind: "s3" | "hf" | "fs"; config: RemoteConfig; created_at: string };
type UploadState = "queued" | "running" | "done" | "failed";
type Upload = { id: string; remote_id: string; job_id: string | null; root: string; path: string; url: string | null; state: UploadState; bytes: number; bytes_done: number; sha256: string | null; error: string | null; created_at: string; updated_at: string };
type Lease = { node_id: string; node_label: string; heartbeat_at: string; expires_at: string } | null;
type SyncStatus = { remote: string | null; last_sync_at: string | null; last_error: string | null; lease: Lease };
type PortableJob = { name: string; step: number; state: string; lease: Lease; updated_at: string; datasets: string[] };
type Settings = { training_folder: string; datasets_folder: string; data_root: string; models_path: string; hf_token_set: boolean; auto_resume_default: boolean; max_auto_resumes_default: number };
type ApiKey = { id: string; name: string; prefix: string; created_at: string; last_used_at: string | null };
type TemplateArch = { name: string; label: string; group: string; source: string; defaults: Record<string, [unknown, unknown]>; disable_sections: string[]; additional_sections: string[]; controls: unknown[]; is_video_model: boolean; python_arch: string; python_accepted: boolean };
type TemplateCatalog = { archs: TemplateArch[]; default_config: JobConfig; errors: string[] };
type RenderedTemplate = { config: JobConfig; warnings: string[] };
type AitkEvent = { id: number; type: string; job_id: string | null; at: string; data: any };
type EventPage = { items: AitkEvent[]; next_cursor: string | null };
type Health = { ok: boolean; version: string; upstream_version: string; started_at: string };
type SystemInfo = { gpus: any[]; cpu: any; disks: { path: string; free: number; total: number }[]; toolkit_root: string };
`

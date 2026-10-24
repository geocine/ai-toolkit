package codemode

// GPU providers and the machines rented on them (handlers/v1machines,
// docs/headless.md "GPU providers and machines"). The types are
// studio/src/lib/provisionTypes.ts without `export` and its import line;
// ops_provision_test.go keeps the two equal.

func init() {
	Ops = append(Ops, provisionOps...)
	RegisterTypes(provisionTypesTS)
	RegisterHelper("machines", machineWaitTS)
	RegisterPrelude(machineWaitJS)
}

var provisionOps = []Op{
	// GPU providers
	{Group: "providers", Name: "list", Method: "GET", Path: "/v1/providers", Result: "Items<Provider>",
		Doc: "GPU providers this server can rent from (RunPod, Vast.ai, QuickPod): capabilities, whether a key is configured (the key itself is never returned), running machines and max_running. Makes no provider calls; simulated is true in simulate mode."},
	{Group: "providers", Name: "setKey", Method: "PUT", Path: "/v1/providers/{kind}", Result: "Provider",
		Doc: "Set the provider's API key (checked with the provider first: provider_auth when it is rejected) or max_running. The key is stored on this server, write-only.",
		Params: []Param{p("kind", "ProviderKind", "path", ""), opt("api_key", "string", "body", "the account API key"),
			opt("max_running", "number", "body", "1 to 20 machines at once (default 2)")}},
	{Group: "providers", Name: "removeKey", Method: "DELETE", Path: "/v1/providers/{kind}", Result: "null",
		Doc: "Remove the stored key (a key from the server's env still applies).", Params: []Param{p("kind", "ProviderKind", "path", "")}},
	{Group: "providers", Name: "test", Method: "POST", Path: "/v1/providers/{kind}/test", Result: "ProviderTest",
		Doc: "Check the key with the provider now; ok false (still 200) says why. credit is the balance when the provider reports it.", Params: []Param{p("kind", "ProviderKind", "path", "")}},
	{Group: "providers", Name: "offers", Method: "GET", Path: "/v1/providers/{kind}/offers", Result: "Items<Offer>",
		Doc: "GPU offers, cheapest first. price_per_hour is what the machine costs while running, with the disk asked for. host and fit say whether the image runs there. Results are cached 30 s. Show the user the price before machines.launch.",
		Params: []Param{p("kind", "ProviderKind", "path", ""), opt("gpu", "string", "query", "substring of the GPU name, e.g. 4090"),
			opt("min_vram_gb", "number", "query", "per GPU"), opt("max_vram_gb", "number", "query", "per GPU, exclusive: keeps offers below it"),
			opt("max_price", "number", "query", "$/h ceiling for the whole machine"), opt("gpu_count", "number", "query", "default 1"),
			opt("cloud", `"secure" | "community"`, "query", "RunPod only (default secure)"), opt("disk_gb", "number", "query", "prices the disk (default 100)"),
			opt("limit", "number", "query", "default 50, max 200"), opt("image", "string", "query", "the image the hosts must run (default ghcr.io/geocine/aitoolkit-go:latest)"),
			opt("model", "string", "query", "model arch (flux, qwen_image, ...) for VRAM badges and the disk need"),
			opt("show_all", "boolean", "query", "include offers the image cannot run, marked in fit"),
			opt("fast", "boolean", "query", "only hosts with fast download and disk (offers that do not report them pass)")}},
	{Group: "providers", Name: "templates", Method: "GET", Path: "/v1/providers/{kind}/templates", Result: "Items<ProviderTemplate>",
		Doc: "The account's templates; QuickPod launches need template_id (pick one with ours and publishes_port).", Params: []Param{p("kind", "ProviderKind", "path", "")}},
	{Group: "providers", Name: "setup", Method: "GET", Path: "/v1/providers/{kind}/setup", Result: "SetupGuide",
		Doc: "The manual setup guide for the provider's console (image, port, env, the fields to fill): for a person who creates the server themselves. Needs no key and never takes a token.",
		Params: []Param{p("kind", "ProviderKind", "path", ""), opt("image", "string", "query", ""), opt("disk_gb", "number", "query", ""),
			opt("cors_origin", "string", "query", "the studio origin the server must allow"), opt("generate_token", "boolean", "query", "fill in a fresh AI_TOOLKIT_AUTH token (returned in token)")}},
	{Group: "providers", Name: "instances", Method: "GET", Path: "/v1/providers/{kind}/instances", Result: "Items<ProviderInstance>",
		Doc:    "Every aitk server (label aitk-...) on the provider account, newest first, including ones this server lost track of. machine_id is the machine here that tracks it; null means it is not tracked and may still be billing: providers.adopt it, or terminate it in the provider console. Never returns env values.",
		Params: []Param{p("kind", "ProviderKind", "path", "")}},
	{Group: "providers", Name: "adopt", Method: "POST", Path: "/v1/providers/{kind}/adopt", Result: "Machine",
		Doc: "Track an instance from providers.instances (machine_id null) as a machine here, so it can be watched, linked, stopped and terminated. The token is read from the provider's env echo when token_known (RunPod, Vast.ai); otherwise pass the server's AI_TOOLKIT_AUTH (QuickPod always, or a RunPod secret reference). 409 when it is already tracked. It rents nothing.",
		Params: []Param{p("kind", "ProviderKind", "path", ""), p("provider_id", "string", "body", "from providers.instances"),
			opt("name", "string", "body", "default <gpu> on <provider>"), opt("token", "string", "body", "its AI_TOOLKIT_AUTH; required when token_known is false")}},

	// machines (GPU servers this server rented)
	{Group: "machines", Name: "list", Method: "GET", Path: "/v1/machines", Result: "Items<Machine>", Doc: "GPU servers this server rented, newest first.",
		Params: []Param{opt("active", "boolean", "query", "only the ones not ended")}},
	{Group: "machines", Name: "launch", Method: "POST", Path: "/v1/machines", Result: "Machine",
		Doc: "Rents a GPU and costs money from the moment it returns. First show the user the offer's price_per_hour and get their OK; max_price_per_hour is required and enforced (409 if the price rose). Use a request_id so a retry never rents twice. copy_remote puts that remote's keys into the machine's env, visible in the provider console. The server re-checks the image and refuses with image_blocked (the checks are in error.details); run images.check or images.recommend first. Then machines.wait and machines.link.",
		Params: []Param{p("provider", "ProviderKind", "body", ""), p("offer_id", "string", "body", "an offer id from providers.offers"),
			p("max_price_per_hour", "number", "body", "the price the user agreed to; refused when the offer now costs more"),
			opt("request_id", "string", "body", "8-64 of [A-Za-z0-9_-]; repeating it returns the same machine"),
			opt("name", "string", "body", "default <gpu> on <provider>"), opt("image", "string", "body", "default ghcr.io/geocine/aitoolkit-go:latest"),
			opt("disk_gb", "number", "body", "20 to 2000, at most the offer's max_disk_gb"), opt("gpu_count", "number", "body", "RunPod only"),
			opt("cloud", `"secure" | "community"`, "body", "RunPod only"), opt("template_id", "string", "body", "QuickPod: required (providers.templates)"),
			opt("cors_origins", "string[]", "body", "origins the machine allows (default the caller's Origin)"), opt("studio_url", "string", "body", "AITK_STUDIO_URL on the machine"),
			opt("copy_remote", "string", "body", "an S3 or Hugging Face remote whose keys the machine gets, to sync results"),
			opt("copy_hf_token", "boolean", "body", "copy this server's Hugging Face token"), opt("copy_notify", "boolean", "body", "copy this server's notification sinks"),
			opt("when_done", "WhenDone", "body", "default destroy with copy_remote (QuickPod also needs self_destroy_key), else keep"),
			opt("self_destroy_key", "boolean", "body", "put the account key in the machine's env so it can delete itself"),
			opt("env", "Record<string, string>", "body", "extra AITK_*, HF_* or TZ variables"),
			opt("skip_image_check", "boolean", "body", "only for custom images or registries that cannot be read; never skips a host block")}},
	{Group: "machines", Name: "get", Method: "GET", Path: "/v1/machines/{id}", Result: "Machine",
		Doc:    "One machine (id or live name): status, problem and detail, price, base_url once ready. The token is never in it (machines.link).",
		Params: []Param{p("id", "string", "path", ""), opt("refresh", "boolean", "query", "re-read it from the provider now (at most every 5 s)")}},
	{Group: "machines", Name: "update", Method: "PATCH", Path: "/v1/machines/{id}", Result: "Machine",
		Doc:    "Rename a machine, or store the token its template set (problem token_mismatch); a token re-probes at once.",
		Params: []Param{p("id", "string", "path", ""), opt("name", "string", "body", ""), opt("token", "string", "body", "the AI_TOOLKIT_AUTH the template set")}},
	{Group: "machines", Name: "stop", Method: "POST", Path: "/v1/machines/{id}/stop", Result: "Machine",
		Doc:    "Stop the machine: compute is released, the disk may still bill (on RunPod the container disk is wiped). 409 unless stop is in actions.",
		Params: []Param{p("id", "string", "path", "")}},
	{Group: "machines", Name: "start", Method: "POST", Path: "/v1/machines/{id}/start", Result: "Machine",
		Doc: "Start a stopped machine; it bills again from now. 409 unless start is in actions.", Params: []Param{p("id", "string", "path", "")}},
	{Group: "machines", Name: "terminate", Method: "POST", Path: "/v1/machines/{id}/terminate", Result: "Machine",
		Doc:    "Deletes the machine and its disk at the provider; anything not on a remote is lost. Irreversible.",
		Params: []Param{p("id", "string", "path", ""), opt("confirm_self", "boolean", "body", "required when it is the server answering (self)")}},
	{Group: "machines", Name: "forget", Method: "DELETE", Path: "/v1/machines/{id}", Result: "null",
		Doc:    "Remove an ended machine from the list. force forgets a live one here only: it keeps running and billing at the provider.",
		Params: []Param{p("id", "string", "path", ""), opt("force", "boolean", "query", "")}},
	{Group: "machines", Name: "link", Method: "POST", Path: "/v1/machines/{id}/link", Result: "MachineLink",
		Doc:    "The ready machine's base_url and token: connect to it as a server (or its /mcp). The only call that returns the token; 409 until it is ready.",
		Params: []Param{p("id", "string", "path", "")}},
	{Group: "machines", Name: "logs", Method: "GET", Path: "/v1/machines/{id}/logs", Result: "MachineLogs",
		Doc: "The container's log tail from the provider (256 KiB at most).", Params: []Param{p("id", "string", "path", ""), opt("tail", "number", "query", "lines, default 200, max 2000")}},
}

const machineWaitTS = `    /** Poll machines.get (refreshing it) until the machine's status is one of until (default: ready, failed, stopped, terminated) or timeout_ms (default 100000) passes, then return the last Machine. Check status and problem: ready with problem open_server cannot be linked. Counts toward the call limit. */
    wait(id: string, options?: { until?: MachineStatus[]; timeout_ms?: number; poll_ms?: number }): Promise<Machine>;
`

const machineWaitJS = `aitk.machines.wait = async function (id, options) {
  const o = options || {};
  const until = o.until || ['ready', 'failed', 'stopped', 'terminated'];
  const poll = Math.max(500, o.poll_ms || 5000);
  const deadline = Date.now() + (o.timeout_ms === undefined ? 100000 : o.timeout_ms);
  for (;;) {
    let m;
    try {
      m = await aitk.machines.get(id, { refresh: true });
    } catch (e) {
      // a rate-limited or failed provider read is not the machine failing:
      // report the stored row and poll again
      if (!(e instanceof AitkError) || (e.code !== 'rate_limited' && e.code !== 'remote_failed')) throw e;
      m = await aitk.machines.get(id);
    }
    if (until.includes(m.status) || Date.now() + poll > deadline) return m;
    await sleep(poll);
  }
};
`

const provisionTypesTS = `// GPU servers created on RunPod, Vast.ai and QuickPod (internal/provision).
// These types mirror the Go JSON exactly; internal/codemode carries the same
// text for the MCP declaration.

type ProviderKind = 'runpod' | 'vast' | 'quickpod';
type MachineStatus = 'creating' | 'starting' | 'running' | 'ready' | 'stopping' | 'stopped' | 'terminating' | 'terminated' | 'failed' | 'unknown';
type MachineProblem = 'token_mismatch' | 'open_server' | 'no_port' | 'not_ready_timeout' | 'unreachable' | 'never_listed' | 'provider_error' | 'not_refreshed' | 'cuda_failed' | 'pull_slow' | 'image_pull_failed';
type WhenDone = 'keep' | 'stop' | 'destroy';
interface ProviderCaps { template_required: boolean; cloud_choice: boolean; count_choice: boolean; stop_start: boolean; stop_keeps_disk: boolean; https: 'yes' | 'no' | 'unconfirmed'; self_destroy: 'builtin' | 'key'; credit: boolean; env_count_max: number; env_value_max: number; env_total_max: number; env_flag_string: boolean; image_digest_ref: 'yes' | 'no' | 'unconfirmed'; reports_inet: boolean; reports_disk_bw: boolean }
interface Provider { kind: ProviderKind; label: string; stage: 'stable' | 'beta'; capabilities: ProviderCaps; unconfirmed: string[]; console_url: string; keys_url: string; key_scopes: string[]; configured: boolean; key_source: 'api' | 'env' | 'sim' | ''; key_hint: string | null; max_running: number; running: number; simulated: boolean }
interface ProviderUpdate { api_key?: string; max_running?: number }
interface ProviderTest { ok: boolean; detail: string; credit: number | null }
interface OfferQuery { gpu?: string; min_vram_gb?: number; max_price?: number; gpu_count?: number; cloud?: 'secure' | 'community'; disk_gb?: number; limit?: number; image?: string; model?: string; show_all?: boolean; fast?: boolean; max_vram_gb?: number }
interface Offer { id: string; provider: ProviderKind; gpu: string; gpu_count: number; vram_gb: number; price_per_hour: number; stopped_price_per_hour: number | null; location: string; reliability: number | null; availability: 'high' | 'medium' | 'low' | 'available'; max_disk_gb: number | null; cloud?: string; https: boolean; host: HostFacts | null; fit: OfferFit | null }
interface ProviderTemplate { id: string; name: string; image: string; ours: boolean; publishes_port: boolean; has_token: boolean }
interface LaunchRequest { provider: ProviderKind; offer_id: string; max_price_per_hour: number; request_id?: string; name?: string; image?: string; disk_gb?: number; gpu_count?: number; cloud?: 'secure' | 'community'; template_id?: string; cors_origins?: string[]; studio_url?: string; copy_remote?: string; copy_hf_token?: boolean; copy_notify?: boolean; when_done?: WhenDone; self_destroy_key?: boolean; env?: Record<string, string>; skip_image_check?: boolean }
interface Machine { id: string; name: string; label: string; provider: ProviderKind; provider_id: string; status: MachineStatus; problem: MachineProblem | null; detail: string; provider_status: string; gpu: string; gpu_count: number; vram_gb: number; location: string; image: string; image_digest: string | null; image_launch_ref: string | null; image_warnings: string[]; disk_gb: number; base_url: string | null; secure: boolean; relay_path: string; price_per_hour: number | null; stopped_price_per_hour: number | null; when_done: WhenDone; self_destroy_key: boolean; copied_remote: string; actions: ('stop' | 'start' | 'terminate' | 'forget')[]; self: boolean; simulated: boolean; warnings: string[]; created_at: string; updated_at: string; ready_at: string | null; ended_at: string | null }
interface MachinePatch { name?: string; token?: string }
interface MachineLink { name: string; base_url: string; token: string; secure: boolean; relay_path: string }
interface MachineLogs { text: string; source: string; truncated: boolean }
interface GuideEnv { name: string; value: string; secret: boolean; note: string }
interface GuideField { label: string; value: string; note: string }
interface SetupGuide { kind: ProviderKind; image: string; port: number; disk_gb: number; env: GuideEnv[]; env_block: string; docker_options: string; fields: GuideField[]; steps: string[]; url_pattern: string; console_url: string; keys_url: string; key_scopes: string[]; token?: string }
interface ProviderInstance { provider_id: string; label: string; status: MachineStatus; provider_status: string; gpu: string; gpu_count: number; price_per_hour: number | null; machine_id: string | null; token_known: boolean; created_at: string | null }
interface AdoptRequest { provider_id: string; name?: string; token?: string }

// GPU recommender (internal/gpupick): best value and best speed in the
// smallest VRAM tier the model fits. It only preselects; it never launches.
interface GpuPick { kind: 'value' | 'speed'; offer: Offer; tier_gb: number; speed: number; speed_source: string; per_dollar: number; reason: string }
interface GpuRecommendation {
  providers: ProviderKind[]; model: ModelRef | null; start_tier_gb: number; tier_gb: number; default: 'value' | '';
  value: GpuPick | null; speed: GpuPick | null; need: DiskNeed;
  tiers: { tier_gb: number; offers: number; qualified: number; rejected: Record<string, number> }[];
  // provider is '' for image and model notes
  notes: { provider: ProviderKind | ''; code: string; message: string; offers: number }[];
  thresholds: { min_inet_down_mbps: number; min_disk_bw_mbps: number; min_reliability: number; ladder: number[] };
  message: string;
}
interface RecommendQuery { model?: string; image?: string; disk_gb?: number; cloud?: 'secure' | 'community'; max_price?: number }
`

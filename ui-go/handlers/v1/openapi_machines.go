package v1

// The GPU provider and machine endpoints (handlers/v1machines) in the
// OpenAPI document. Their shapes are in docs/headless.md, "GPU providers and
// machines"; bodies are plain objects here, the relay is raw bytes.
func init() {
	offerQuery := []string{"gpu:string", "min_vram_gb:integer", "max_price:number", "gpu_count:integer", "cloud:string",
		"disk_gb:integer", "limit:integer", "image:string", "model:string", "show_all:boolean", "fast:boolean", "max_vram_gb:integer"}
	ops = append(ops,
		op{"GET", "/v1/providers", "Providers", "GPU providers: capabilities, whether a key is set (never the key), running machines", nil, "", 200, "object"},
		op{"PUT", "/v1/providers/{kind}", "Providers", "Set the provider's API key (verified first; write-only) or max_running", nil, "object", 200, "object"},
		op{"DELETE", "/v1/providers/{kind}", "Providers", "Remove the stored key (an env key still applies)", nil, "", 204, ""},
		op{"POST", "/v1/providers/{kind}/test", "Providers", "Check the key with the provider ({ok, detail, credit}; 200 even when not ok)", nil, "", 200, "object"},
		op{"GET", "/v1/providers/{kind}/offers", "Providers", "GPU offers, cheapest first, with host facts and how well the image fits", offerQuery, "", 200, "object"},
		op{"GET", "/v1/providers/{kind}/templates", "Providers", "The account's templates (QuickPod launches need one)", nil, "", 200, "object"},
		op{"GET", "/v1/providers/{kind}/setup", "Providers", "Manual setup guide for the provider's console (no key needed)", []string{"image:string", "disk_gb:integer", "cors_origin:string", "generate_token:boolean"}, "", 200, "object"},
		op{"GET", "/v1/providers/{kind}/instances", "Providers", "Every aitk-labelled server on the provider account; machine_id when tracked here (never env values)", nil, "", 200, "object"},
		op{"POST", "/v1/providers/{kind}/adopt", "Providers", "Track an untracked aitk server as a machine here (token from the provider's env echo, or given)", nil, "object", 201, "object"},

		op{"GET", "/v1/machines", "Machines", "GPU servers this server rented, newest first", []string{"active:boolean"}, "", 200, "object"},
		op{"POST", "/v1/machines", "Machines", "Rent a GPU server (costs money; max_price_per_hour required; 200 for a repeated request_id)", nil, "object", 201, "object"},
		op{"GET", "/v1/machines/{id}", "Machines", "One machine (refresh=1 re-reads it from the provider)", []string{"refresh:boolean"}, "", 200, "object"},
		op{"PATCH", "/v1/machines/{id}", "Machines", "Rename, or store the token the template set (re-probes)", nil, "object", 200, "object"},
		op{"POST", "/v1/machines/{id}/stop", "Machines", "Stop the machine (compute released; disk may still bill)", nil, "", 200, "object"},
		op{"POST", "/v1/machines/{id}/start", "Machines", "Start a stopped machine", nil, "", 200, "object"},
		op{"POST", "/v1/machines/{id}/terminate", "Machines", "Delete the machine and its disk at the provider (irreversible); optional body confirm_self: true when it is this server", nil, "", 200, "object"},
		op{"DELETE", "/v1/machines/{id}", "Machines", "Forget an ended machine here (force=1 forgets a live one; it keeps billing)", []string{"force:boolean"}, "", 204, ""},
		op{"POST", "/v1/machines/{id}/link", "Machines", "The ready machine's URL and token, to add it as a server", nil, "", 200, "object"},
		op{"GET", "/v1/machines/{id}/logs", "Machines", "Container log tail from the provider", []string{"tail:integer"}, "", 200, "object"},
		op{"GET", "/v1/machines/{id}/relay/{path}", "Machines", "Relay a request to the machine (plain-http hosts)", nil, "", 200, "binary"},
		op{"POST", "/v1/machines/{id}/relay/{path}", "Machines", "Relay a request to the machine", nil, "binary", 200, "binary"},
		op{"PUT", "/v1/machines/{id}/relay/{path}", "Machines", "Relay a request to the machine", nil, "binary", 200, "binary"},
		op{"PATCH", "/v1/machines/{id}/relay/{path}", "Machines", "Relay a request to the machine", nil, "binary", 200, "binary"},
		op{"DELETE", "/v1/machines/{id}/relay/{path}", "Machines", "Relay a request to the machine", nil, "", 200, "binary"},
	)
}

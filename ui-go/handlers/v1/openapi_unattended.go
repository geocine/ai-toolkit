package v1

// The unattended-run endpoints (handlers/v1unattended: when done and
// notifications) in the OpenAPI document. Their shapes are in
// docs/headless.md, "Unattended runs"; bodies are plain objects here.
func init() {
	ops = append(ops,
		op{"GET", "/v1/lifecycle", "lifecycle", "When done: mode, state, countdown deadline, reason, provider, results check", nil, "", 200, "object"},
		op{"PATCH", "/v1/lifecycle", "lifecycle", "Change when done at runtime ({mode, on, delay_s, dry_run})", nil, "object", 200, "object"},
		op{"POST", "/v1/lifecycle/cancel", "lifecycle", "Cancel the countdown (the server stays up until the next job ends)", nil, "", 200, "object"},
		op{"POST", "/v1/lifecycle/extend", "lifecycle", "Push the countdown deadline out ({minutes})", nil, "object", 200, "object"},
		op{"POST", "/v1/lifecycle/now", "lifecycle", "Verify results and stop/destroy now (409 while a job is queued or running)", nil, "", 200, "object"},
		op{"GET", "/v1/notify", "notify", "Notification sinks (secrets never returned), event filter, link bases", nil, "", 200, "object"},
		op{"PATCH", "/v1/notify", "notify", "Change the event filter, public/studio URL, sample interval", nil, "object", 200, "object"},
		op{"POST", "/v1/notify/sinks", "notify", "Add a sink ({url}: ntfy, Discord webhook, telegram://, or a JSON webhook)", nil, "object", 201, "object"},
		op{"DELETE", "/v1/notify/sinks/{id}", "notify", "Remove a sink", nil, "", 204, ""},
		op{"POST", "/v1/notify/test", "notify", "Send a test notification and report each sink's answer", nil, "object", 200, "object"},
	)
}

// Package jobstate maps upstream's job status (plus what the fork knows) to
// the client-facing lifecycle of /v1 (docs/headless.md, Jobs).
package jobstate

import "strings"

// Client-facing states.
const (
	Draft       = "draft"
	Queued      = "queued"
	Starting    = "starting"
	Running     = "running"
	Stopping    = "stopping"
	Stopped     = "stopped"
	Completed   = "completed"
	Failed      = "failed"
	Interrupted = "interrupted"
)

// interruptedPrefixes are the infos written when a trainer died without
// writing its own ending (by the supervisor, or the launcher seeing a signal
// or an unwatched exit): the process was cut off, so the job can resume.
var interruptedPrefixes = []string{
	"Interrupted:",
	"Job process terminated with signal",
	"Job process exited unexpectedly.",
}

// EndState classifies a job that left running/stopping with this status.
func EndState(status, info string) string {
	switch status {
	case "completed":
		return Completed
	case "stopped":
		return Stopped
	case "error":
		for _, p := range interruptedPrefixes {
			if strings.HasPrefix(info, p) {
				return Interrupted
			}
		}
		return Failed
	}
	return status
}

// FromStatus is the client state for an upstream status. A stopped job that
// never ran is a draft (upstream creates jobs as "stopped").
func FromStatus(status string, step int, info string) string {
	switch status {
	case "stopped":
		if step == 0 && info == "" {
			return Draft
		}
		return Stopped
	case "queued":
		return Queued
	case "running":
		if step == 0 && (info == "" || strings.HasPrefix(info, "Starting") || strings.HasPrefix(info, "Loading")) {
			return Starting
		}
		return Running
	case "stopping":
		return Stopping
	case "completed", "error":
		return EndState(status, info)
	}
	return status
}

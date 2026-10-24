package quickpod

import (
	"time"

	"github.com/geocine/aitoolkit/internal/provision"
)

// creatingFor is how long a pod with no Docker state yet reads as creating
// (the console shows "setting up"); after that it is starting.
const creatingFor = 60 * time.Second

// statusOf maps a pod's Docker State and intended_state onto a status
// (spec: quickpod.md §6.2, the console's derived badge). Only "running" and
// "exited" are confirmed State values; "dead" is Docker's state for a
// container that failed and is read as failed (UNCONFIRMED that QuickPod
// reports it), anything else is unknown with the raw value kept.
func statusOf(p rawPod, now time.Time) (provision.Status, string) {
	switch {
	case p.Destroyed:
		return provision.StatusTerminated, ""
	case p.State == "dead":
		return provision.StatusFailed, "QuickPod reports the container as dead."
	case p.State == "running" && p.Intended == "exited":
		return provision.StatusStopping, ""
	case p.State == "running":
		return provision.StatusRunning, ""
	case p.Intended == "running":
		if p.State == "" && now.Sub(time.UnixMilli(p.CreatedAt)) < creatingFor {
			return provision.StatusCreating, ""
		}
		return provision.StatusStarting, ""
	case p.State == "exited":
		return provision.StatusStopped, ""
	}
	return provision.StatusUnknown, "QuickPod reports state " + quote(p.State) + " (intended " + quote(p.Intended) + ")."
}

// providerStatus is the raw "<State>/<intended_state>" pair shown next to
// the status ("destroyed" once destroyed).
func providerStatus(p rawPod) string {
	if p.Destroyed {
		return "destroyed"
	}
	return dash(p.State) + "/" + dash(p.Intended)
}

// actionsFor lists the lifecycle calls that make sense in a status.
func actionsFor(s provision.Status) []string {
	switch s {
	case provision.StatusCreating, provision.StatusStarting, provision.StatusRunning:
		return []string{"stop", "terminate"}
	case provision.StatusStopped:
		return []string{"start", "terminate"}
	case provision.StatusTerminated:
		return nil
	}
	return []string{"terminate"}
}

func dash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func quote(s string) string { return `"` + s + `"` }

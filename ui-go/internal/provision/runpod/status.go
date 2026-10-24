package runpod

import (
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/provision"
)

// mapStatus maps PodStatus (spec: runpod.md §4.1). EXITED is stopped; the
// service adds a detail when the stop was not ours.
func mapStatus(raw string) (provision.Status, string) {
	switch raw {
	case "PROVISIONING":
		return provision.StatusCreating, ""
	case "STARTING":
		return provision.StatusStarting, ""
	case "RUNNING":
		return provision.StatusRunning, ""
	case "EXITED":
		return provision.StatusStopped, ""
	case "ERROR":
		return provision.StatusFailed, "RunPod reports the container in an unrecoverable error state."
	case "TERMINATED":
		return provision.StatusTerminated, ""
	}
	return provision.StatusUnknown, ""
}

// ruleWords are the documented cross-field rules a createPod 400 can name.
// Any other 400 on create is taken as "could not be placed": the capacity
// text itself is UNCONFIRMED, so a capacity regex would misfire on it, while
// our bodies never send the fields these rules are about.
var ruleWords = regexp.MustCompile(`(?i)cuda|mount|persistent|network volume|global|exclusive|together`)

// classify maps a v2 status (spec: runpod.md §3.1 createPod semantics).
func classify(status int, msg string, r req) provision.Class {
	switch {
	case status == http.StatusUnauthorized:
		return provision.ClassAuth
	case status == http.StatusPaymentRequired:
		return provision.ClassCredit
	case status == http.StatusForbidden && r.create:
		return provision.ClassCapacity // the account cannot use this pool; another may work
	case status == http.StatusForbidden:
		return provision.ClassAuth
	case status == http.StatusBadRequest && r.create && !ruleWords.MatchString(msg):
		return provision.ClassCapacity
	case status == http.StatusNotFound:
		return provision.ClassNotFound
	case status == http.StatusConflict:
		return provision.ClassConflict
	case status == http.StatusTooManyRequests:
		return provision.ClassRateLimited
	case status >= 500:
		return provision.ClassUnavailable
	}
	return provision.ClassInvalid // 400 elsewhere, 413, 422 and the rest
}

// apiError decodes application/problem+json {title,status,detail,errors?}.
// The provider's text is always in the message, redacted and clipped.
func apiError(res provision.Response, r req, secrets []string) *provision.Error {
	var p struct {
		Title  string   `json:"title"`
		Detail string   `json:"detail"`
		Errors []string `json:"errors"`
	}
	_ = json.Unmarshal(res.Body, &p)
	msg := strings.TrimSpace(p.Detail)
	if msg == "" {
		msg = strings.TrimSpace(p.Title)
	}
	if msg == "" && !json.Valid(res.Body) {
		msg = strings.TrimSpace(string(res.Body))
	}
	if msg == "" {
		msg = http.StatusText(res.Status)
	}
	if len(p.Errors) > 0 {
		msg += ": " + strings.Join(p.Errors, "; ")
	}
	class := classify(res.Status, msg, r)
	switch {
	case res.Status == http.StatusForbidden && r.create:
		msg = "Your account can't use this GPU pool (" + msg + ")"
	case res.Status == http.StatusForbidden && r.scope != "":
		msg = "the key lacks the " + r.scope + " scope (" + msg + ")"
	}
	e := &provision.Error{Provider: provision.RunPod, Class: class, Status: res.Status,
		Message: provision.Clip(provision.Redact(msg, secrets...), 300)}
	if class == provision.ClassRateLimited {
		e.RetryAfter = provision.RetryAfter(res.Header, time.Now())
	}
	return e
}

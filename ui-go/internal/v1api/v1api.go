// Package v1api holds the conventions every /v1 handler follows (see
// docs/headless.md): JSON bodies, {"error": {code, message}} errors and
// {"items": [...]} lists.
package v1api

import (
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"strconv"
	"time"
)

// Error codes and their HTTP status.
const (
	CodeInvalidRequest = "invalid_request" // 400
	CodeUnauthorized   = "unauthorized"    // 401
	CodeForbidden      = "forbidden"       // 403
	CodeNotFound       = "not_found"       // 404
	CodeConflict       = "conflict"        // 409
	CodeDatasetExists  = "dataset_exists"  // 409: a dataset of that name holds other files
	CodeTooLarge       = "too_large"       // 413
	CodeInvalidConfig  = "invalid_config"  // 422
	CodeImageBlocked   = "image_blocked"   // 422: the image cannot run on the chosen host; details carry the checks
	CodeInternal       = "internal"        // 500
	CodeRemoteFailed   = "remote_failed"   // 502

	// GPU servers (internal/provision).
	CodeProviderAuth       = "provider_auth"       // 400: the provider rejected the key (never 401: that means our token)
	CodeInsufficientCredit = "insufficient_credit" // 402
	CodeNoCapacity         = "no_capacity"         // 409
	CodeRateLimited        = "rate_limited"        // 429
)

var statusOf = map[string]int{
	CodeInvalidRequest: http.StatusBadRequest,
	CodeUnauthorized:   http.StatusUnauthorized,
	CodeForbidden:      http.StatusForbidden,
	CodeNotFound:       http.StatusNotFound,
	CodeConflict:       http.StatusConflict,
	CodeDatasetExists:  http.StatusConflict,
	CodeTooLarge:       http.StatusRequestEntityTooLarge,
	CodeInvalidConfig:  http.StatusUnprocessableEntity,
	CodeImageBlocked:   http.StatusUnprocessableEntity,
	CodeInternal:       http.StatusInternalServerError,
	CodeRemoteFailed:   http.StatusBadGateway,

	CodeProviderAuth:       http.StatusBadRequest,
	CodeInsufficientCredit: http.StatusPaymentRequired,
	CodeNoCapacity:         http.StatusConflict,
	CodeRateLimited:        http.StatusTooManyRequests,
}

// JSON writes v with status.
func JSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		log.Printf("v1api: encode response: %v", err)
	}
}

// OK writes v with 200.
func OK(w http.ResponseWriter, v any) { JSON(w, http.StatusOK, v) }

// Created writes v with 201.
func Created(w http.ResponseWriter, v any) { JSON(w, http.StatusCreated, v) }

// NoContent writes 204.
func NoContent(w http.ResponseWriter) { w.WriteHeader(http.StatusNoContent) }

// Items writes {"items": items}; a nil slice is sent as [].
func Items[T any](w http.ResponseWriter, items []T) {
	if items == nil {
		items = []T{}
	}
	OK(w, map[string]any{"items": items})
}

// ItemsPage writes {"items": items, "next_cursor": cursor} ("" -> null).
func ItemsPage[T any](w http.ResponseWriter, items []T, cursor string) {
	if items == nil {
		items = []T{}
	}
	var next any
	if cursor != "" {
		next = cursor
	}
	OK(w, map[string]any{"items": items, "next_cursor": next})
}

// Fail writes {"error": {"code", "message"}} with the code's status.
func Fail(w http.ResponseWriter, code, message string) {
	JSON(w, StatusOf(code), map[string]any{"error": map[string]string{"code": code, "message": message}})
}

// FailDetails writes {"error": {"code", "message", "details"}} with the code's
// status, for errors a client acts on (a blocked launch carries its checks).
func FailDetails(w http.ResponseWriter, code, message string, details any) {
	JSON(w, StatusOf(code), map[string]any{"error": map[string]any{"code": code, "message": message, "details": details}})
}

// FailAfter is Fail plus Retry-After (whole seconds, min 1) when after > 0.
func FailAfter(w http.ResponseWriter, code, message string, after time.Duration) {
	if after > 0 {
		w.Header().Set("Retry-After", strconv.Itoa(max(1, int(after.Round(time.Second)/time.Second))))
	}
	Fail(w, code, message)
}

// StatusOf is the HTTP status of a code; unknown codes are 500.
func StatusOf(code string) int {
	if status, ok := statusOf[code]; ok {
		return status
	}
	return http.StatusInternalServerError
}

// Internal logs err and answers 500 internal.
func Internal(w http.ResponseWriter, r *http.Request, err error) {
	log.Printf("%s %s: %v", r.Method, r.URL.Path, err)
	Fail(w, CodeInternal, "internal error")
}

// Problem is an error carrying a v1 code, for helpers that decide the response.
type Problem struct{ Code, Message string }

func (p *Problem) Error() string { return p.Code + ": " + p.Message }

// Errorf builds a Problem.
func Errorf(code, message string) error { return &Problem{Code: code, Message: message} }

// Respond writes err: its code when it is a Problem, else 500.
func Respond(w http.ResponseWriter, r *http.Request, err error) {
	var p *Problem
	if errors.As(err, &p) {
		Fail(w, p.Code, p.Message)
		return
	}
	Internal(w, r, err)
}

// MaxJSONBody caps JSON request bodies.
const MaxJSONBody = 16 << 20

// Decode reads a JSON body into v, answering 400 itself when it fails.
func Decode(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(io.LimitReader(r.Body, MaxJSONBody))
	if err := dec.Decode(v); err != nil {
		if errors.Is(err, io.EOF) {
			Fail(w, CodeInvalidRequest, "request body is empty")
		} else {
			Fail(w, CodeInvalidRequest, "invalid JSON body: "+err.Error())
		}
		return false
	}
	return true
}

// Time renders a time for /v1 responses (RFC 3339, UTC, millisecond precision).
func Time(t time.Time) string { return t.UTC().Format("2006-01-02T15:04:05.000Z07:00") }

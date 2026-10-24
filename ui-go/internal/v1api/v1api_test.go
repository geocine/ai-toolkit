package v1api

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestFailStatus(t *testing.T) {
	cases := map[string]int{
		CodeInvalidRequest: 400,
		CodeUnauthorized:   401,
		CodeNotFound:       404,
		CodeConflict:       409,
		CodeInvalidConfig:  422,
		CodeImageBlocked:   422,
		CodeRemoteFailed:   502,
		"no_such_code":     500,

		CodeProviderAuth:       400,
		CodeInsufficientCredit: 402,
		CodeNoCapacity:         409,
		CodeRateLimited:        429,
	}
	for code, want := range cases {
		rec := httptest.NewRecorder()
		Fail(rec, code, "m")
		if rec.Code != want || StatusOf(code) != want {
			t.Errorf("%s: %d (StatusOf %d), want %d", code, rec.Code, StatusOf(code), want)
		}
		var body struct {
			Error map[string]any `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil || body.Error["code"] != code || body.Error["message"] != "m" {
			t.Errorf("%s body: %s", code, rec.Body)
		}
		if _, ok := body.Error["details"]; ok {
			t.Errorf("%s: Fail must not send details", code)
		}
	}
}

func TestFailDetails(t *testing.T) {
	type check struct {
		Level string `json:"level"`
		Code  string `json:"code"`
	}
	rec := httptest.NewRecorder()
	FailDetails(rec, CodeImageBlocked, "This host can't run the image.", map[string]any{"checks": []check{{"block", "host_cuda"}}})
	if rec.Code != http.StatusUnprocessableEntity || rec.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("status %d %v", rec.Code, rec.Header())
	}
	var body struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
			Details struct {
				Checks []check `json:"checks"`
			} `json:"details"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	e := body.Error
	if e.Code != "image_blocked" || e.Message != "This host can't run the image." || len(e.Details.Checks) != 1 || e.Details.Checks[0].Code != "host_cuda" {
		t.Fatalf("body %s", rec.Body)
	}

	// nil details are sent as null, so clients can always read the key.
	rec = httptest.NewRecorder()
	FailDetails(rec, CodeConflict, "busy", nil)
	var raw struct {
		Error map[string]json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil || rec.Code != 409 || string(raw.Error["details"]) != "null" {
		t.Fatalf("nil details: %d %s", rec.Code, rec.Body)
	}
}

func TestRespondProblem(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/machines", nil)
	Respond(rec, req, Errorf(CodeImageBlocked, "blocked"))
	if rec.Code != 422 {
		t.Fatalf("Problem: %d", rec.Code)
	}
	rec = httptest.NewRecorder()
	Respond(rec, req, errors.New("boom"))
	if rec.Code != 500 {
		t.Fatalf("plain error: %d", rec.Code)
	}
}

func TestFailAfter(t *testing.T) {
	for _, c := range []struct {
		after time.Duration
		want  string
	}{{12 * time.Second, "12"}, {1500 * time.Millisecond, "2"}, {200 * time.Millisecond, "1"}, {0, ""}} {
		rec := httptest.NewRecorder()
		FailAfter(rec, CodeRateLimited, "slow down", c.after)
		if rec.Code != http.StatusTooManyRequests || rec.Header().Get("Retry-After") != c.want {
			t.Errorf("after %v: %d Retry-After %q, want %q", c.after, rec.Code, rec.Header().Get("Retry-After"), c.want)
		}
	}
}

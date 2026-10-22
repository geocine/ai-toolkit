// Package v1unattended serves /v1/lifecycle (when done: stop or destroy the
// server once the work is finished and its results are safe) and /v1/notify
// (where notifications go). See docs/headless.md, "Unattended runs".
package v1unattended

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/geocine/aitoolkit/internal/bg"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/lifecycle"
	"github.com/geocine/aitoolkit/internal/notify"
	"github.com/geocine/aitoolkit/internal/v1api"
)

func Register(mux *http.ServeMux) {
	bg.Go("notify", notify.Run)
	bg.Go("lifecycle", lifecycle.Start)

	httpx.Handle(mux, "GET /v1/lifecycle", getLifecycle)
	httpx.Handle(mux, "PATCH /v1/lifecycle", patchLifecycle)
	httpx.Handle(mux, "POST /v1/lifecycle/cancel", lifecycleAction(func(*http.Request) (lifecycle.Status, error) {
		return lifecycle.Cancel()
	}))
	httpx.Handle(mux, "POST /v1/lifecycle/extend", extendLifecycle)
	httpx.Handle(mux, "POST /v1/lifecycle/now", lifecycleAction(func(*http.Request) (lifecycle.Status, error) {
		return lifecycle.Now()
	}))

	httpx.Handle(mux, "GET /v1/notify", getNotify)
	httpx.Handle(mux, "PATCH /v1/notify", patchNotify)
	httpx.Handle(mux, "POST /v1/notify/sinks", addSink)
	httpx.Handle(mux, "DELETE /v1/notify/sinks/{id}", deleteSink)
	httpx.Handle(mux, "POST /v1/notify/test", testNotify)
}

func respondLifecycle(w http.ResponseWriter, r *http.Request, s lifecycle.Status, err error) {
	switch {
	case errors.Is(err, lifecycle.ErrConflict):
		v1api.Fail(w, v1api.CodeConflict, err.Error())
	case errors.Is(err, lifecycle.ErrInvalid):
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
	case err != nil:
		v1api.Internal(w, r, err)
	default:
		v1api.OK(w, s)
	}
}

func getLifecycle(w http.ResponseWriter, r *http.Request) {
	s, err := lifecycle.Get()
	respondLifecycle(w, r, s, err)
}

func lifecycleAction(fn func(*http.Request) (lifecycle.Status, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		s, err := fn(r)
		respondLifecycle(w, r, s, err)
	}
}

func patchLifecycle(w http.ResponseWriter, r *http.Request) {
	var p lifecycle.Patch
	if !v1api.Decode(w, r, &p) {
		return
	}
	s, err := lifecycle.Update(r.Context(), p)
	respondLifecycle(w, r, s, err)
}

func extendLifecycle(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Minutes *float64 `json:"minutes"`
	}
	if !v1api.Decode(w, r, &body) {
		return
	}
	if body.Minutes == nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "minutes is required")
		return
	}
	s, err := lifecycle.Extend(time.Duration(*body.Minutes * float64(time.Minute)))
	respondLifecycle(w, r, s, err)
}

// sinkView is a sink as the API shows it: the URL (which holds the secret)
// never leaves the server.
type sinkView struct {
	ID        string    `json:"id"`
	Kind      string    `json:"kind"`
	Target    string    `json:"target"`
	Source    string    `json:"source"`
	CreatedAt time.Time `json:"created_at"`
}

type notifyView struct {
	Sinks        []sinkView `json:"sinks"`
	Events       []string   `json:"events"` // null = every kind
	Kinds        []string   `json:"kinds"`
	PublicURL    string     `json:"public_url"`
	StudioURL    string     `json:"studio_url"`
	SampleEveryS int        `json:"sample_every_s"`
}

func view(c notify.Config) notifyView {
	v := notifyView{Sinks: []sinkView{}, Events: c.Events, Kinds: notify.Kinds, PublicURL: notify.PublicURL(c),
		StudioURL: notify.StudioURL(c), SampleEveryS: c.SampleEvery}
	if v.SampleEveryS == 0 {
		v.SampleEveryS = int(notify.DefaultSampleEvery.Seconds())
	}
	for _, s := range c.Sinks {
		kind, target := notify.SinkInfo(s.URL)
		v.Sinks = append(v.Sinks, sinkView{ID: s.ID, Kind: kind, Target: target, Source: s.Source, CreatedAt: s.CreatedAt})
	}
	return v
}

func getNotify(w http.ResponseWriter, r *http.Request) {
	v1api.OK(w, view(notify.Current()))
}

func patchNotify(w http.ResponseWriter, r *http.Request) {
	// events: absent = unchanged, null = every kind, [] = none
	var body map[string]any
	if !v1api.Decode(w, r, &body) {
		return
	}
	var events []string
	if raw, ok := body["events"]; ok && raw != nil {
		list, ok := raw.([]any)
		if !ok {
			v1api.Fail(w, v1api.CodeInvalidRequest, "events must be an array of kinds or null")
			return
		}
		events = []string{}
		for _, e := range list {
			s, ok := e.(string)
			if !ok {
				v1api.Fail(w, v1api.CodeInvalidRequest, "events must be strings")
				return
			}
			events = append(events, s)
		}
		if err := notify.ValidEvents(events); err != nil {
			v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
			return
		}
	}
	str := func(key string) (string, bool, bool) {
		raw, ok := body[key]
		if !ok {
			return "", false, true
		}
		if raw == nil {
			return "", true, true
		}
		s, isStr := raw.(string)
		return s, true, isStr
	}
	pub, setPub, okPub := str("public_url")
	studio, setStudio, okStudio := str("studio_url")
	if !okPub || !okStudio {
		v1api.Fail(w, v1api.CodeInvalidRequest, "public_url and studio_url must be strings")
		return
	}
	var every *int
	if raw, ok := body["sample_every_s"]; ok {
		n, isNum := raw.(float64)
		if !isNum || n < 0 {
			v1api.Fail(w, v1api.CodeInvalidRequest, "sample_every_s must be a non-negative number")
			return
		}
		v := int(n)
		every = &v
	}
	c, err := notify.Update(r.Context(), func(c *notify.Config) error {
		if _, ok := body["events"]; ok {
			c.Events = events
		}
		if setPub {
			c.PublicURL = pub
		}
		if setStudio {
			c.StudioURL = studio
		}
		if every != nil {
			c.SampleEvery = *every
		}
		return nil
	})
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	v1api.OK(w, view(c))
}

func addSink(w http.ResponseWriter, r *http.Request) {
	var body struct {
		URL string `json:"url"`
	}
	if !v1api.Decode(w, r, &body) {
		return
	}
	s, err := notify.NewSink(body.URL, "api")
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, err.Error())
		return
	}
	dup := false
	c, err := notify.Update(r.Context(), func(c *notify.Config) error {
		for _, x := range c.Sinks {
			if x.ID == s.ID {
				dup = true
				return nil
			}
		}
		c.Sinks = append(c.Sinks, s)
		return nil
	})
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if dup {
		v1api.Fail(w, v1api.CodeConflict, "that sink is already configured")
		return
	}
	for _, sv := range view(c).Sinks {
		if sv.ID == s.ID {
			v1api.Created(w, sv)
			return
		}
	}
}

func deleteSink(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	found := false
	_, err := notify.Update(r.Context(), func(c *notify.Config) error {
		kept := c.Sinks[:0]
		for _, s := range c.Sinks {
			if s.ID == id {
				found = true
				continue
			}
			kept = append(kept, s)
		}
		c.Sinks = kept
		return nil
	})
	if err != nil {
		v1api.Internal(w, r, err)
		return
	}
	if !found {
		v1api.Fail(w, v1api.CodeNotFound, "sink not found")
		return
	}
	v1api.NoContent(w)
}

const testTimeout = 45 * time.Second

func testNotify(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Sink    string `json:"sink"`
		Message string `json:"message"`
	}
	if r.ContentLength != 0 && !v1api.Decode(w, r, &body) {
		return
	}
	c := notify.Current()
	if len(c.Sinks) == 0 {
		v1api.Fail(w, v1api.CodeConflict, "no notification sinks are configured (AITK_NOTIFY or POST /v1/notify/sinks)")
		return
	}
	var ids []string
	if body.Sink != "" {
		ids = []string{body.Sink}
		known := false
		for _, s := range c.Sinks {
			known = known || s.ID == body.Sink
		}
		if !known {
			v1api.Fail(w, v1api.CodeNotFound, "sink not found")
			return
		}
	}
	text := body.Message
	if text == "" {
		text = "Notifications from this ai-toolkit server reach you here."
	}
	ctx, cancel := context.WithTimeout(r.Context(), testTimeout)
	defer cancel()
	results := notify.SendNow(ctx, notify.Message{Kind: notify.KindTest, Level: notify.LevelInfo,
		Title: "ai-toolkit: test notification", Body: text, URL: notify.StudioURL(c)}, ids)
	v1api.Items(w, results)
}

// Package v1recommend serves the GPU recommender: the default offer pick on
// one provider (GET /v1/providers/{kind}/recommend) and across every
// provider with a key (GET /v1/hosts/recommend). The work is done by
// internal/gpupick, which reads offers through provision.Default (so the
// 30 s offers cache applies); see docs/headless.md, "GPU recommendation".
//
// It only suggests: nothing here rents a GPU or calls a provider other than
// for offers.
package v1recommend

import (
	"net/http"
	"slices"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/gpupick"
	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Register adds the routes. The image check and the model sizer are set up
// here unless something (v1images, simulate mode, a test) already set them.
func Register(mux *http.ServeMux) {
	if imagecheck.Default == nil {
		imagecheck.Init(false)
	}
	if hostfit.DefaultSizer == nil {
		hostfit.InitSizer(false)
	}

	httpx.Handle(mux, "GET /v1/providers/{kind}/recommend", providerRecommend)
	httpx.Handle(mux, "GET /v1/hosts/recommend", hostsRecommend)
}

// ---- helpers ----

// respond writes err: an adapter error through provision.V1 (with
// Retry-After on 429), else a v1api.Problem or 500.
func respond(w http.ResponseWriter, r *http.Request, err error) {
	if code, msg, after, ok := provision.V1(err); ok {
		v1api.FailAfter(w, code, msg, after)
		return
	}
	v1api.Respond(w, r, err)
}

func invalid(msg string) error { return v1api.Errorf(v1api.CodeInvalidRequest, msg) }

// query reads what both routes share: model, image, disk_gb, cloud and
// max_price, with the bounds the offers route and a launch apply.
func query(r *http.Request) (gpupick.Query, error) {
	get := func(name string) string { return strings.TrimSpace(r.URL.Query().Get(name)) }
	var q gpupick.Query
	if a := get("model"); a != "" {
		q.Model = &hostfit.ModelRef{Arch: a}
	}
	q.Image = get("image")
	if len(q.Image) > 1024 || strings.ContainsAny(q.Image, " \t\r\n") {
		return q, invalid("image must be an image reference without spaces")
	}
	if v := get("disk_gb"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 20 || n > 2000 {
			return q, invalid("disk_gb must be a whole number from 20 to 2000")
		}
		q.DiskGB = n
	}
	q.Cloud = get("cloud")
	if q.Cloud != "" && q.Cloud != "secure" && q.Cloud != "community" {
		return q, invalid("cloud must be secure or community")
	}
	if v := get("max_price"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || !(f > 0) || f > 1e6 {
			return q, invalid("max_price must be a price above 0 ($/h)")
		}
		q.MaxPrice = f
	}
	return q, nil
}

// ---- GET /v1/providers/{kind}/recommend ----

// providerRecommend is the pick on one provider. A provider error ends it
// with that error (429 with Retry-After, 402, 400 provider_auth, 502):
// the tiers it did not search would make "nothing qualifies" a guess.
func providerRecommend(w http.ResponseWriter, r *http.Request) {
	k, ok := provision.ParseKind(r.PathValue("kind"))
	if ok {
		_, ok = provision.Lookup(k)
	}
	if !ok {
		v1api.Fail(w, v1api.CodeNotFound, "unknown provider")
		return
	}
	q, err := query(r)
	if err != nil {
		respond(w, r, err)
		return
	}
	q.Providers = []provision.Kind{k}
	rec, err := gpupick.Recommend(r.Context(), q)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.OK(w, rec)
}

// ---- GET /v1/hosts/recommend ----

// hostsRecommend is the pick across providers. A provider that fails is a
// note, never the answer's status: the others still give picks. It fails
// only when no listed provider has a key, a name is not a provider, or
// every listed provider is turned off on this server.
func hostsRecommend(w http.ResponseWriter, r *http.Request) {
	q, err := query(r)
	if err != nil {
		respond(w, r, err)
		return
	}
	for _, name := range strings.Split(r.URL.Query().Get("providers"), ",") {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		k, ok := provision.ParseKind(name)
		if !ok {
			v1api.Fail(w, v1api.CodeInvalidRequest, "providers: "+strconv.Quote(provision.Clip(name, 40))+" is not a provider (runpod, vast or quickpod)")
			return
		}
		if !slices.Contains(q.Providers, k) {
			q.Providers = append(q.Providers, k)
		}
	}
	// a provider turned off by AITK_PROVISION_PROVIDERS is a note when others
	// are listed too; when every listed one is off it is the request's fault
	var off []string
	for _, k := range q.Providers {
		if _, ok := provision.Lookup(k); !ok {
			off = append(off, strconv.Quote(string(k)))
		}
	}
	if len(off) > 0 && len(off) == len(q.Providers) {
		verb := " isn't"
		if len(off) > 1 {
			verb = " aren't"
		}
		v1api.Fail(w, v1api.CodeInvalidRequest, "providers: "+strings.Join(off, ", ")+verb+" enabled on this server")
		return
	}
	rec, err := gpupick.Recommend(r.Context(), q)
	if err != nil {
		if len(rec.Providers) == 0 {
			// nothing was searched: no key, or the providers could not be listed
			respond(w, r, err)
			return
		}
		// every provider searched failed; the notes say why
		if rec.Message == "" {
			rec.Message = failedMessage(rec)
		}
	}
	v1api.OK(w, rec)
}

// failedMessage is the sentence for a cross-provider run where every
// provider searched failed before any offer came back, from the notes: an
// image the registry says is missing or private first, as gpupick puts it.
func failedMessage(rec gpupick.Recommendation) string {
	var parts, why []string
	for _, n := range rec.Notes {
		switch {
		case n.Provider == "" && (n.Code == "image_denied" || n.Code == "image_not_found"):
			parts = append(parts, n.Message)
		case n.Provider != "" && n.Code != gpupick.NoteNotConfigured && n.Code != gpupick.NoteNotDecorated:
			why = append(why, strings.TrimSuffix(n.Message, "."))
		}
	}
	if len(why) == 0 {
		parts = append(parts, "No provider could be searched right now. Try again in a moment.")
	} else {
		parts = append(parts, "No provider could be searched right now: "+strings.Join(why, "; ")+".")
	}
	return strings.Join(parts, " ")
}

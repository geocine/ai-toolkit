// Package v1machines serves /v1/providers (GPU provider keys, offers,
// templates and the manual setup guides) and /v1/machines (GPU servers this
// server rented: launch, watch, stop, link, relay). The work is done by
// provision.Default; see docs/headless.md, "GPU providers and machines".
package v1machines

import (
	"bytes"
	"context"
	"crypto/subtle"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/bg"
	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/provision/guide"
	"github.com/geocine/aitoolkit/internal/provision/relay"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/studioapi"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// relayMethods are the methods the relay forwards (one route each).
var relayMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE"}

func Register(mux *http.ServeMux) {
	bg.Go("machines", func(ctx context.Context) {
		if m := provision.Default; m != nil {
			m.Run(ctx)
		}
	})

	httpx.Handle(mux, "GET /v1/providers", listProviders)
	httpx.Handle(mux, "PUT /v1/providers/{kind}", setKey)
	httpx.Handle(mux, "DELETE /v1/providers/{kind}", removeKey)
	httpx.Handle(mux, "POST /v1/providers/{kind}/test", testKey)
	httpx.Handle(mux, "GET /v1/providers/{kind}/offers", offers)
	httpx.Handle(mux, "GET /v1/providers/{kind}/templates", templates)
	httpx.Handle(mux, "GET /v1/providers/{kind}/setup", setup)
	httpx.Handle(mux, "GET /v1/providers/{kind}/instances", instances)
	httpx.Handle(mux, "POST /v1/providers/{kind}/adopt", adopt)

	httpx.Handle(mux, "GET /v1/machines", listMachines)
	httpx.Handle(mux, "POST /v1/machines", launch)
	httpx.Handle(mux, "GET /v1/machines/{id}", getMachine)
	httpx.Handle(mux, "PATCH /v1/machines/{id}", updateMachine)
	httpx.Handle(mux, "POST /v1/machines/{id}/stop", stopMachine)
	httpx.Handle(mux, "POST /v1/machines/{id}/start", startMachine)
	httpx.Handle(mux, "POST /v1/machines/{id}/terminate", terminateMachine)
	httpx.Handle(mux, "DELETE /v1/machines/{id}", forgetMachine)
	httpx.Handle(mux, "POST /v1/machines/{id}/link", linkMachine)
	httpx.Handle(mux, "GET /v1/machines/{id}/logs", machineLogs)
	for _, m := range relayMethods {
		httpx.Handle(mux, m+" /v1/machines/{id}/relay/{path...}", relayMachine)
	}
}

// ---- helpers ----

// DetailedError is an error that answers with details next to its code and
// message (v1api.FailDetails), e.g. a launch the image check blocked:
// 422 image_blocked with the checks in error.details.
type DetailedError interface {
	error
	V1Details() (code, message string, details any)
}

// respond writes err: a DetailedError with its details, an adapter error
// through provision.V1 (with Retry-After on 429), else a v1api.Problem or 500.
func respond(w http.ResponseWriter, r *http.Request, err error) {
	var d DetailedError
	if errors.As(err, &d) {
		code, msg, details := d.V1Details()
		v1api.FailDetails(w, code, msg, details)
		return
	}
	if code, msg, after, ok := provision.V1(err); ok {
		v1api.FailAfter(w, code, msg, after)
		return
	}
	v1api.Respond(w, r, err)
}

// kindOf reads {kind}; a kind that is unknown or disabled answers 404.
func kindOf(w http.ResponseWriter, r *http.Request) (provision.Kind, bool) {
	k, ok := provision.ParseKind(r.PathValue("kind"))
	if ok {
		_, ok = provision.Lookup(k)
	}
	if !ok {
		v1api.Fail(w, v1api.CodeNotFound, "unknown provider")
	}
	return k, ok
}

// loopbackPeer is true for a direct connection from this machine: a loopback
// RemoteAddr and no proxy header saying the request came from elsewhere.
func loopbackPeer(r *http.Request) bool {
	for _, h := range []string{"X-Forwarded-For", "X-Real-Ip", "Forwarded"} {
		if r.Header.Get(h) != "" {
			return false
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// localCaller is a loopback peer that also named this machine as its Host
// (localhost, a loopback IP, AITK_PUBLIC_URL's host or AITK_ALLOWED_HOSTS).
// A DNS-rebound page reaches the server from 127.0.0.1 too, but its Host is
// still the attacker's name, so it is not a local caller. This holds in
// either mode; studio API mode's Gate checks the same Host on every route.
func localCaller(r *http.Request) bool {
	return loopbackPeer(r) && studioapi.HostAllowed(r.Host)
}

// guard is the open-server guard: a server without AI_TOOLKIT_AUTH or API
// keys rents GPUs only for requests from this machine (localCaller).
func guard(w http.ResponseWriter, r *http.Request) bool {
	if !httpx.IsOpen() || localCaller(r) {
		return true
	}
	v1api.Fail(w, v1api.CodeForbidden, "Set AI_TOOLKIT_AUTH on this server before it can rent GPUs for you.")
	return false
}

// authenticated reports whether r carried this server's own credentials
// (AI_TOOLKIT_AUTH or an API key), or came from this machine to an open
// server. Only then does the relay add the machine's token; a request that
// reached the relay without them (a public media GET, a signed link) is
// forwarded with no credentials and the machine decides.
func authenticated(r *http.Request) bool {
	if httpx.IsOpen() {
		return localCaller(r)
	}
	scheme, token, ok := strings.Cut(r.Header.Get("Authorization"), " ")
	if !ok || !strings.EqualFold(scheme, "Bearer") || token == "" {
		return false
	}
	if want := os.Getenv("AI_TOOLKIT_AUTH"); want != "" && subtle.ConstantTimeCompare([]byte(token), []byte(want)) == 1 {
		return true
	}
	return state.VerifyKey(r.Context(), token)
}

// boolQ reads a boolean query param: 1 or true.
func boolQ(r *http.Request, name string) bool {
	v := strings.TrimSpace(r.URL.Query().Get(name))
	return v == "1" || strings.EqualFold(v, "true")
}

// queryReader collects the first bad number of a query.
type queryReader struct {
	r   *http.Request
	err error
}

func (q *queryReader) int(name string, lo, hi int) int {
	v := strings.TrimSpace(q.r.URL.Query().Get(name))
	if v == "" || q.err != nil {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < lo || n > hi {
		q.err = v1api.Errorf(v1api.CodeInvalidRequest, name+" must be a whole number from "+strconv.Itoa(lo)+" to "+strconv.Itoa(hi))
		return 0
	}
	return n
}

func (q *queryReader) price(name string) float64 {
	v := strings.TrimSpace(q.r.URL.Query().Get(name))
	if v == "" || q.err != nil {
		return 0
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || !(f > 0) || f > 1e6 {
		q.err = v1api.Errorf(v1api.CodeInvalidRequest, name+" must be a price above 0 ($/h)")
		return 0
	}
	return f
}

// decodeOptional reads an optional JSON body: an empty body leaves v as is.
func decodeOptional(w http.ResponseWriter, r *http.Request, v any) bool {
	body, err := io.ReadAll(io.LimitReader(r.Body, v1api.MaxJSONBody))
	if err != nil {
		v1api.Fail(w, v1api.CodeInvalidRequest, "could not read the request body")
		return false
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return true
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	return v1api.Decode(w, r, v)
}

// ---- providers ----

func listProviders(w http.ResponseWriter, r *http.Request) {
	items, err := provision.Default.Providers(r.Context())
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.Items(w, items)
}

func setKey(w http.ResponseWriter, r *http.Request) {
	k, ok := kindOf(w, r)
	if !ok || !guard(w, r) {
		return
	}
	var in provision.ProviderUpdate
	if !v1api.Decode(w, r, &in) {
		return
	}
	v, err := provision.Default.SetKey(r.Context(), k, in)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.OK(w, v)
}

func removeKey(w http.ResponseWriter, r *http.Request) {
	k, ok := kindOf(w, r)
	if !ok {
		return
	}
	if err := provision.Default.RemoveKey(r.Context(), k); err != nil {
		respond(w, r, err)
		return
	}
	v1api.NoContent(w)
}

func testKey(w http.ResponseWriter, r *http.Request) {
	k, ok := kindOf(w, r)
	if !ok {
		return
	}
	res, err := provision.Default.Test(r.Context(), k)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.OK(w, res)
}

// offerQuery parses every OfferQuery param of GET /v1/providers/{kind}/offers.
func offerQuery(r *http.Request) (provision.OfferQuery, error) {
	get := func(name string) string { return strings.TrimSpace(r.URL.Query().Get(name)) }
	qr := &queryReader{r: r}
	q := provision.OfferQuery{
		GPU:       get("gpu"),
		MinVRAMGB: qr.int("min_vram_gb", 0, 1024),
		MaxPrice:  qr.price("max_price"),
		GPUCount:  qr.int("gpu_count", 1, 16),
		Cloud:     get("cloud"),
		DiskGB:    qr.int("disk_gb", 20, 2000),
		Limit:     qr.int("limit", 1, 200),
		Image:     get("image"),
		ShowAll:   boolQ(r, "show_all"),
		Fast:      boolQ(r, "fast"),
		MaxVRAMGB: qr.int("max_vram_gb", 0, 1024),
	}
	if qr.err != nil {
		return q, qr.err
	}
	if q.Cloud != "" && q.Cloud != "secure" && q.Cloud != "community" {
		return q, v1api.Errorf(v1api.CodeInvalidRequest, "cloud must be secure or community")
	}
	if len(q.Image) > 1024 || strings.ContainsAny(q.Image, " \t\r\n") {
		return q, v1api.Errorf(v1api.CodeInvalidRequest, "image must be an image reference without spaces")
	}
	if a := get("model"); a != "" {
		q.Model = &hostfit.ModelRef{Arch: a}
	}
	return q, nil
}

func offers(w http.ResponseWriter, r *http.Request) {
	k, ok := kindOf(w, r)
	if !ok {
		return
	}
	q, err := offerQuery(r)
	if err != nil {
		respond(w, r, err)
		return
	}
	items, err := provision.Default.Offers(r.Context(), k, q)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.Items(w, items)
}

func templates(w http.ResponseWriter, r *http.Request) {
	k, ok := kindOf(w, r)
	if !ok {
		return
	}
	items, err := provision.Default.Templates(r.Context(), k)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.Items(w, items)
}

// setup is the manual setup guide. It needs no key and never accepts a
// token: generate_token=1 makes a fresh one for this guide only.
func setup(w http.ResponseWriter, r *http.Request) {
	k, ok := kindOf(w, r)
	if !ok {
		return
	}
	qr := &queryReader{r: r}
	o := guide.Options{
		Image:      strings.TrimSpace(r.URL.Query().Get("image")),
		DiskGB:     qr.int("disk_gb", 20, 2000),
		CORSOrigin: strings.TrimSpace(r.URL.Query().Get("cors_origin")),
	}
	if qr.err != nil {
		respond(w, r, qr.err)
		return
	}
	if boolQ(r, "generate_token") {
		o.Token = provision.NewToken()
	}
	g, err := guide.For(k, o)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.OK(w, g)
}

// ---- machines ----

func listMachines(w http.ResponseWriter, r *http.Request) {
	items, err := provision.Default.List(r.Context(), boolQ(r, "active"))
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.Items(w, items)
}

func launch(w http.ResponseWriter, r *http.Request) {
	if !guard(w, r) {
		return
	}
	var req provision.LaunchRequest
	if !v1api.Decode(w, r, &req) {
		return
	}
	m, created, err := provision.Default.Launch(r.Context(), req, r.Header.Get("Origin"))
	if err != nil {
		respond(w, r, err)
		return
	}
	if created {
		v1api.Created(w, m)
		return
	}
	v1api.OK(w, m)
}

func getMachine(w http.ResponseWriter, r *http.Request) {
	m, err := provision.Default.Get(r.Context(), r.PathValue("id"), boolQ(r, "refresh"))
	machineOrError(w, r, m, err)
}

func updateMachine(w http.ResponseWriter, r *http.Request) {
	var p provision.MachinePatch
	if !v1api.Decode(w, r, &p) {
		return
	}
	m, err := provision.Default.Update(r.Context(), r.PathValue("id"), p)
	machineOrError(w, r, m, err)
}

func stopMachine(w http.ResponseWriter, r *http.Request) {
	m, err := provision.Default.Stop(r.Context(), r.PathValue("id"))
	machineOrError(w, r, m, err)
}

func startMachine(w http.ResponseWriter, r *http.Request) {
	if !guard(w, r) {
		return
	}
	m, err := provision.Default.Start(r.Context(), r.PathValue("id"))
	machineOrError(w, r, m, err)
}

func terminateMachine(w http.ResponseWriter, r *http.Request) {
	var body struct {
		ConfirmSelf bool `json:"confirm_self"`
	}
	if !decodeOptional(w, r, &body) {
		return
	}
	m, err := provision.Default.Terminate(r.Context(), r.PathValue("id"), body.ConfirmSelf)
	machineOrError(w, r, m, err)
}

func machineOrError(w http.ResponseWriter, r *http.Request, m provision.Machine, err error) {
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.OK(w, m)
}

func forgetMachine(w http.ResponseWriter, r *http.Request) {
	if err := provision.Default.Forget(r.Context(), r.PathValue("id"), boolQ(r, "force")); err != nil {
		respond(w, r, err)
		return
	}
	v1api.NoContent(w)
}

// linkMachine is the only response that carries a machine's token.
func linkMachine(w http.ResponseWriter, r *http.Request) {
	if !guard(w, r) {
		return
	}
	l, err := provision.Default.Link(r.Context(), r.PathValue("id"))
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.OK(w, l)
}

func machineLogs(w http.ResponseWriter, r *http.Request) {
	qr := &queryReader{r: r}
	tail := qr.int("tail", 1, 100000)
	if qr.err != nil {
		respond(w, r, qr.err)
		return
	}
	l, err := provision.Default.Logs(r.Context(), r.PathValue("id"), tail)
	if err != nil {
		respond(w, r, err)
		return
	}
	v1api.OK(w, l)
}

// relayMachine forwards to a ready machine (internal/provision/relay).
func relayMachine(w http.ResponseWriter, r *http.Request) {
	if !guard(w, r) {
		return
	}
	t, err := provision.Default.RelayTarget(r.Context(), r.PathValue("id"))
	if err != nil {
		respond(w, r, err)
		return
	}
	relay.Serve(w, r, t, r.PathValue("path"), authenticated(r))
}

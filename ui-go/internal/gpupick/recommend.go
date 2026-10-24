package gpupick

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/geocine/aitoolkit/internal/hostfit"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// Note codes gpupick sets itself (the others are /v1 error codes and
// imagecheck codes).
const (
	NoteNotConfigured = "not_configured"
	NoteNotDecorated  = "not_decorated"
	NoteModelUnknown  = "model_unknown"
)

// providerTimeout is PerProviderTimeout; tests shorten it.
var providerTimeout = PerProviderTimeout

// searched is one provider in a run.
type searched struct {
	kind        provision.Kind
	caps        provision.Caps
	ctx         context.Context
	cancel      context.CancelFunc
	err         error // set when a query failed; the provider is skipped from then on
	seen        int   // offers returned so far
	undecorated int   // offers of a walked tier without a fit
}

// Recommend walks the ladder from StartTier. For each tier L it asks every
// provider concurrently (each bounded by PerProviderTimeout for the whole
// run) for
//
//	OfferQuery{GPUCount: 1, MinVRAMGB: floor(L x TierSlack), MaxVRAMGB: ceil(next x TierSlack)
//	           (0 for the last tier), Fast: true, Image, Model, DiskGB, Cloud, MaxPrice, Limit: TierLimit}
//
// through provision.Default.Offers (so the 30 s offers cache applies),
// classifies the offers with TierOf, qualifies and ranks those of tier L,
// and stops at the first tier with a pick. Fast makes Vast filter inet_down
// and disk_bw server-side, so slow cheap hosts cannot crowd a page. A
// provider error becomes a Note (provision.V1 code) and that provider is
// skipped for later tiers.
//
// A provider that does not answer within PerProviderTimeout fails with a
// ClassUnavailable *provision.Error. When every live provider fails in a
// tier the walk stops there, that tier is not listed in Tiers, and Message
// says where it stopped and why.
//
// It fails only when no provider can be searched: no listed provider has a
// key (a *v1api.Problem), or every provider failed before returning any
// offer (the first provider's error, mappable with provision.V1). A run on
// one listed provider also fails with that provider's error when the walk
// stopped on it, since the higher tiers were never searched. The
// Recommendation it returns then still carries the notes, so a
// cross-provider caller may answer with them instead.
func Recommend(ctx context.Context, q Query) (Recommendation, error) {
	th := DefaultThresholds()
	rec := Recommendation{Providers: []provision.Kind{}, Model: q.Model, StartTierGB: StartTier(q.Model),
		Tiers: []TierResult{}, Notes: []Note{}, Thresholds: th}
	m := provision.Default
	if m == nil {
		return rec, errors.New("gpupick: provisioning is not set up on this server")
	}
	provs, err := providersFor(ctx, m, q, &rec)
	if err != nil {
		return rec, err
	}

	ref := strings.TrimSpace(q.Image)
	if ref == "" {
		ref = provision.DefaultImage
	}
	img, cudaMin, imageNote := readImage(ctx, ref)
	rec.Need = hostfit.NeedFor(img, q.Model, hostfit.DefaultSizer)
	if imageNote != nil {
		rec.Notes = append(rec.Notes, *imageNote)
	}
	if q.Model != nil && !hostfit.KnownArch(q.Model.Arch) {
		rec.Notes = append(rec.Notes, Note{Code: NoteModelUnknown,
			Message: fmt.Sprintf("There's no VRAM data for %q, so the search starts at %d GB.", q.Model.Arch, rec.StartTierGB)})
	}

	for _, p := range provs {
		p.ctx, p.cancel = context.WithTimeout(ctx, providerTimeout)
		defer p.cancel()
	}
	cross := len(q.Providers) != 1
	disk := diskAsk(rec.Need, q)
	// stopped is the tier where every live provider failed, and failed
	// those providers; that tier is not a result (nobody answered it).
	stopped, failed := 0, []*searched(nil)
	for i, l := range th.Ladder {
		if l < rec.StartTierGB {
			continue
		}
		live := slices.DeleteFunc(slices.Clone(provs), func(p *searched) bool { return p.err != nil })
		if len(live) == 0 {
			break
		}
		oq := provision.OfferQuery{GPUCount: 1, MinVRAMGB: int(math.Floor(float64(l)*TierSlack + eps)), Fast: true,
			Image: strings.TrimSpace(q.Image), Model: q.Model, DiskGB: disk, Cloud: q.Cloud, MaxPrice: q.MaxPrice, Limit: TierLimit}
		if i+1 < len(th.Ladder) {
			oq.MaxVRAMGB = int(math.Ceil(float64(th.Ladder[i+1])*TierSlack - eps))
		}
		cands := fetchTier(m, live, oq, l, &rec)
		if !slices.ContainsFunc(live, func(p *searched) bool { return p.err == nil }) {
			stopped, failed = l, live
			break
		}
		tr, v, s := rankTier(cands, l, rec.Need, q, th)
		rec.Tiers = append(rec.Tiers, tr)
		if v != nil {
			v.Reason, s.Reason = ReasonLine(*v, cross, cudaMin), ReasonLine(*s, cross, cudaMin)
			rec.TierGB, rec.Default, rec.Value, rec.Speed = l, "value", v, s
			break
		}
	}
	for _, p := range provs {
		if p.undecorated > 0 {
			rec.Notes = append(rec.Notes, Note{Provider: p.kind, Code: NoteNotDecorated, Offers: p.undecorated,
				Message: provision.Label(p.kind) + " offers were not checked against the image."})
		}
	}

	var firstErr error
	seen := 0
	for _, p := range provs {
		seen += p.seen
		if firstErr == nil && p.err != nil {
			firstErr = p.err
		}
	}
	if seen == 0 && !slices.ContainsFunc(provs, func(p *searched) bool { return p.err == nil }) {
		return rec, firstErr
	}
	rec.Message = message(rec, q, imageNote, cudaMin, disk, stopped, failed)
	if stopped > 0 && !cross {
		// The tiers above were never searched, so "nothing qualifies"
		// would hide the real reason: the provider's own error.
		return rec, failed[0].err
	}
	return rec, nil
}

// providersFor resolves which providers a run searches (with their caps),
// in provider order, and notes the listed ones it cannot. A single listed
// provider without a key, or none at all, is the 400 the API answers.
func providersFor(ctx context.Context, m provision.Manager, q Query, rec *Recommendation) ([]*searched, error) {
	views, err := m.Providers(ctx)
	if err != nil {
		return nil, err
	}
	byKind := map[provision.Kind]provision.ProviderView{}
	for _, v := range views {
		byKind[v.Kind] = v
	}
	want := q.Providers
	if len(want) == 0 {
		for _, v := range views {
			want = append(want, v.Kind)
		}
	}
	want = slices.Clone(want)
	slices.SortStableFunc(want, func(a, b provision.Kind) int { return providerOrder[a] - providerOrder[b] })
	want = slices.Compact(want)

	var out []*searched
	var missing []string
	for _, k := range want {
		v, ok := byKind[k]
		switch {
		case !ok && len(q.Providers) == 1:
			return nil, v1api.Errorf(v1api.CodeNotFound, "unknown provider")
		case !ok:
			name := provision.Label(k)
			if name == "" {
				name = string(k)
			}
			rec.Notes = append(rec.Notes, Note{Provider: k, Code: NoteNotConfigured, Message: name + " isn't enabled on this server."})
		case !v.Configured:
			missing = append(missing, provision.Label(k))
			if len(q.Providers) > 0 {
				rec.Notes = append(rec.Notes, Note{Provider: k, Code: NoteNotConfigured, Message: "Add a " + provision.Label(k) + " API key first."})
			}
		default:
			out = append(out, &searched{kind: k, caps: v.Caps})
			rec.Providers = append(rec.Providers, k)
		}
	}
	if len(out) > 0 {
		return out, nil
	}
	if len(q.Providers) == 1 && len(missing) == 1 {
		return nil, v1api.Errorf(v1api.CodeInvalidRequest, "Add a "+missing[0]+" API key first")
	}
	if len(missing) == 0 {
		missing = []string{provision.Label(provision.RunPod), provision.Label(provision.Vast), provision.Label(provision.QuickPod)}
	}
	return nil, v1api.Errorf(v1api.CodeInvalidRequest, "Add an API key for "+orList(missing)+" first")
}

// orList is "A", "A or B", "A, B or C".
func orList(s []string) string { return joinList(s, " or ") }

// andList is "A", "A and B", "A, B and C".
func andList(s []string) string { return joinList(s, " and ") }

func joinList(s []string, last string) string {
	if len(s) < 2 {
		return strings.Join(s, "")
	}
	return strings.Join(s[:len(s)-1], ", ") + last + s[len(s)-1]
}

// readImage reads the image the picks are judged against, through the
// provisioning image check. When it cannot be read the failure becomes one
// note, and the default image's built-in requirements stand in (MASTER
// §3.10.3); other images are then not judged.
func readImage(ctx context.Context, ref string) (*imagecheck.Image, string, *Note) {
	var img *imagecheck.Image
	var err error
	if ic := provision.ImageChecker(); ic != nil {
		img, err = ic.Inspect(ctx, ref, "", false)
	} else {
		err = imagecheck.Errorf(imagecheck.KindUnreachable, "no image check is configured on this server")
	}
	if err == nil {
		return img, img.Requires.CUDAMin, nil
	}
	c := imagecheck.ErrorCheck(err)
	n := &Note{Code: c.Code, Message: c.Message}
	if fb, ok := hostfit.FallbackImage(ref); ok {
		return fb, fb.Requires.CUDAMin, n
	}
	return nil, "", n
}

// fetchTier asks every live provider for one tier at once and returns the
// offers of that tier, in provider order. A failure marks the provider and
// adds its note. A provider that has not answered when its run deadline
// passes fails with ClassUnavailable; its call is left running, since the
// offers cache's shared flight is detached from p.ctx and still fills the
// cache for the next request.
func fetchTier(m provision.Manager, live []*searched, oq provision.OfferQuery, tier int, rec *Recommendation) []Candidate {
	type result struct {
		offers []provision.Offer
		err    error
	}
	res := make([]result, len(live))
	var wg sync.WaitGroup
	for i, p := range live {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ch := make(chan result, 1)
			go func() {
				o, err := m.Offers(p.ctx, p.kind, oq)
				ch <- result{o, err}
			}()
			var r result
			select {
			case r = <-ch:
			case <-p.ctx.Done():
				r.err = p.ctx.Err()
			}
			if r.err != nil && errors.Is(p.ctx.Err(), context.DeadlineExceeded) {
				r = result{err: timeoutError(p.kind)}
			}
			res[i] = r
		}()
	}
	wg.Wait()

	var out []Candidate
	for i, p := range live {
		if err := res[i].err; err != nil {
			p.err = err
			rec.Notes = append(rec.Notes, errorNote(p.kind, err, p.seen))
			continue
		}
		p.seen += len(res[i].offers)
		for _, o := range res[i].offers {
			if TierOf(offerVRAM(o)) != tier {
				continue
			}
			if o.Fit == nil {
				p.undecorated++
			}
			out = append(out, Candidate{Offer: o, Caps: p.caps})
		}
	}
	return out
}

// timeoutError is the error of a provider that did not answer within
// providerTimeout; provision.V1 maps it to remote_failed (502).
func timeoutError(k provision.Kind) *provision.Error {
	return provision.Errorf(k, provision.ClassUnavailable, 0, "no answer within %s", providerTimeout.Round(time.Millisecond))
}

// errorNote is the note for a provider whose offers call failed.
func errorNote(k provision.Kind, err error, offers int) Note {
	n := Note{Provider: k, Offers: offers}
	var p *v1api.Problem
	switch code, msg, _, ok := provision.V1(err); {
	case ok:
		n.Code, n.Message = code, msg
	case errors.As(err, &p):
		n.Code, n.Message = p.Code, p.Message
	case errors.Is(err, context.DeadlineExceeded):
		n.Code, n.Message = v1api.CodeRemoteFailed, provision.Label(k)+" did not answer in time."
	default:
		n.Code, n.Message = v1api.CodeRemoteFailed, provision.Label(k)+" did not answer: "+provision.Clip(err.Error(), 200)
	}
	return n
}

// message is the run's sentence: an image the registry says is missing or
// private first, then, when nothing qualifies, why, from the counts, and
// where the walk stopped when the providers failed partway.
func message(rec Recommendation, q Query, imageNote *Note, cudaMin string, disk int, stopped int, failed []*searched) string {
	var parts []string
	if imageNote != nil && (imageNote.Code == "image_denied" || imageNote.Code == "image_not_found") {
		parts = append(parts, imageNote.Message)
	}
	if rec.Value == nil {
		parts = append(parts, nothingQualifies(rec, q, cudaMin, disk))
	}
	if stopped > 0 {
		var names []string
		for _, p := range failed {
			names = append(names, provision.Label(p.kind))
		}
		parts = append(parts, fmt.Sprintf("The search stopped at %d GB because %s failed.", stopped, andList(names)))
	}
	return strings.Join(parts, " ")
}

func nothingQualifies(rec Recommendation, q Query, cudaMin string, disk int) string {
	seen := 0
	counts := map[string]int{}
	for _, t := range rec.Tiers {
		seen += t.Offers
		for code, n := range t.Rejected {
			counts[code] += n
		}
	}
	if seen == 0 {
		price := ""
		if q.MaxPrice > 0 {
			price = fmt.Sprintf(", max $%.2f/h", q.MaxPrice)
		}
		return fmt.Sprintf("No offers match these filters (disk %d GB%s). Change the filters or pick a provider with more stock.", disk, price)
	}
	top, n := "", 0
	for _, code := range rejectOrder {
		if counts[code] > n {
			top, n = code, counts[code]
		}
	}
	driver := "a driver that runs the image"
	if cudaMin != "" {
		driver = "a driver for CUDA " + cudaMin
	}
	last := rec.Tiers[len(rec.Tiers)-1].TierGB
	return fmt.Sprintf("No offer from %d to %d GB has fast internet, enough disk (%d GB) and %s right now. Most were left out for %s (%d). Pick one yourself from the list.",
		rec.StartTierGB, last, disk, driver, RejectLabels[top], n)
}

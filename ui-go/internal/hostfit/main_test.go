package hostfit

import (
	"encoding/json"
	"net/http"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/imagecheck/registrymock"
)

// TestMain is the network tripwire: any request through the default
// transport fails, so a test that forgets a mock cannot reach a registry or
// Hugging Face.
func TestMain(m *testing.M) {
	http.DefaultTransport = tripwire{}
	os.Exit(m.Run())
}

type tripwire struct{}

func (tripwire) RoundTrip(r *http.Request) (*http.Response, error) {
	panic("hostfit test reached the network: " + r.URL.Host)
}

// codesOf is the check codes, in order.
func codesOf(checks []imagecheck.Check) []string {
	out := make([]string, 0, len(checks))
	for _, c := range checks {
		out = append(out, c.Code)
	}
	return out
}

func hasCode(checks []imagecheck.Check, code string) bool {
	return slices.Contains(codesOf(checks), code)
}

func checkOf(t *testing.T, checks []imagecheck.Check, code string) imagecheck.Check {
	t.Helper()
	for _, c := range checks {
		if c.Code == code {
			return c
		}
	}
	t.Fatalf("no %s check in %v", code, codesOf(checks))
	return imagecheck.Check{}
}

func readJSON(t *testing.T, name string, v any) {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

// vastOffers is the Vast fixture: 61 public offers then the synthetic rows.
func vastOffers(t *testing.T) []map[string]any {
	t.Helper()
	var f struct {
		Offers []map[string]any `json:"offers"`
	}
	readJSON(t, "vast_offers.json", &f)
	return f.Offers
}

func vastRow(t *testing.T, id float64) map[string]any {
	t.Helper()
	for _, o := range vastOffers(t) {
		if o["id"] == id {
			return o
		}
	}
	t.Fatalf("no vast row %v", id)
	return nil
}

func quickPodOffers(t *testing.T) []map[string]any {
	t.Helper()
	var rows []map[string]any
	readJSON(t, "quickpod_rentable.json", &rows)
	return rows
}

// newMock starts a registrymock serving images (Fixtures() when nil) and
// returns a client wired to it.
func newMock(t *testing.T, images []registrymock.Image, f registrymock.Faults) (*registrymock.Server, *imagecheck.Client) {
	t.Helper()
	if images == nil {
		images = registrymock.Fixtures()
	}
	srv := registrymock.New(registrymock.Options{Images: images, Faults: f})
	srv.StartTLS()
	t.Cleanup(srv.Close)
	return srv, imagecheck.New(imagecheck.Options{HTTP: srv.Client(), CheckTimeout: 5 * time.Second, Logf: t.Logf})
}

// goImage is the labelled cu130 Go-server image the fixtures publish.
func goImage(t *testing.T, c *imagecheck.Client) *imagecheck.Image {
	t.Helper()
	img, err := c.Inspect(t.Context(), DefaultRepo+":"+registrymock.Revision, "", false)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// retag copies the fixture repo:from as repo:to, with the labels changed
// by set (nil keeps them) and a new creation time.
func retag(t *testing.T, images []registrymock.Image, repo, from, to string, set map[string]string, created string) []registrymock.Image {
	t.Helper()
	for _, img := range images {
		if img.Repo != repo || img.Tag != from {
			continue
		}
		img.Tag = to
		if set != nil || created != "" {
			var cfg map[string]any
			if err := json.Unmarshal(img.Config, &cfg); err != nil {
				t.Fatal(err)
			}
			labels := cfg["config"].(map[string]any)["Labels"].(map[string]any)
			for k, v := range set {
				labels[k] = v
			}
			if created != "" {
				cfg["created"] = created
			}
			b, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			img.Config = b
		}
		out := slices.DeleteFunc(slices.Clone(images), func(o registrymock.Image) bool { return o.Repo == repo && o.Tag == to })
		return append(out, img)
	}
	t.Fatalf("no fixture %s:%s", repo, from)
	return nil
}

// goRequirements is what today's cu130 build needs.
func goRequirements() imagecheck.Requirements {
	req, _ := imagecheck.FallbackRequirements(DefaultRepo)
	return req
}

// fallbackImage is FallbackImage for the default repo.
func fallbackImage(t *testing.T) *imagecheck.Image {
	t.Helper()
	img, ok := FallbackImage(DefaultRepo + ":latest")
	if !ok {
		t.Fatal("no fallback image")
	}
	return img
}

package hfmock

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"testing"
	"time"
)

func get(t *testing.T, s *Server, path string) (*http.Response, []map[string]any) {
	t.Helper()
	resp, err := http.Get(s.URL() + path)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var files []map[string]any
	if resp.StatusCode == http.StatusOK {
		if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
			t.Fatal(err)
		}
	}
	return resp, files
}

func TestServe(t *testing.T) {
	s := New(Options{Faults: Faults{Status: map[string]int{"Qwen/Qwen-Image": 429}}})
	s.Start()
	defer s.Close()
	if !slices.Contains(Repos(), "black-forest-labs/FLUX.2-dev") {
		t.Errorf("repos %v", Repos())
	}
	resp, files := get(t, s, "/api/models/black-forest-labs/FLUX.1-dev/tree/main?recursive=true")
	if resp.StatusCode != 200 || len(files) < 20 {
		t.Fatalf("FLUX.1-dev: %d, %d files", resp.StatusCode, len(files))
	}
	_, root := get(t, s, "/api/models/black-forest-labs/FLUX.1-dev/tree/main")
	if len(root) >= len(files) {
		t.Errorf("non-recursive listing has %d entries", len(root))
	}
	if resp, _ := get(t, s, "/api/models/nobody/nothing/tree/main?recursive=true"); resp.StatusCode != 404 {
		t.Errorf("unknown repo: %d", resp.StatusCode)
	}
	if resp, _ := get(t, s, "/api/models/Qwen/Qwen-Image/tree/main?recursive=true"); resp.StatusCode != 429 {
		t.Errorf("forced status: %d", resp.StatusCode)
	}
	if n := len(s.Requests()); n != 4 {
		t.Errorf("%d requests recorded", n)
	}
}

func TestSlowAndPages(t *testing.T) {
	s := New(Options{Faults: Faults{SlowMs: 2000, PageSize: 3}, Extra: map[string]string{"me/tiny": `[{"type":"file","path":"a.safetensors","size":1}]`}})
	s.Start()
	defer s.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, s.URL()+"/api/models/me/tiny/tree/main?recursive=true", nil)
	if _, err := http.DefaultClient.Do(req); err == nil {
		t.Error("slow response arrived before the deadline")
	}
	fast := New(Options{Faults: Faults{PageSize: 3}})
	fast.Start()
	defer fast.Close()
	resp, files := get(t, fast, "/api/models/black-forest-labs/FLUX.1-dev/tree/main?recursive=true")
	if len(files) != 3 || resp.Header.Get("Link") == "" {
		t.Errorf("page: %d files, Link %q", len(files), resp.Header.Get("Link"))
	}
}

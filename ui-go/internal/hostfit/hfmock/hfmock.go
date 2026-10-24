// Package hfmock is a fake Hugging Face tree API
// (GET /api/models/{owner}/{name}/tree/main?recursive=true) serving trimmed
// tree listings, so the model sizer and simulate mode never reach
// huggingface.co. It is a non-test package so simulate mode can mount it.
package hfmock

import (
	"embed"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"time"
)

// testdata/hf holds one listing per repo (owner__name.json), trimmed to
// path, size and type. Qwen/Qwen-Image is the real listing; the others are
// reconstructed from the measured totals in IMAGE-PLAN §3.5 (file names as
// published, sizes summing to the measured repo and component totals).
//
//go:embed testdata/hf/*.json
var testdata embed.FS

// Faults bends the mock away from the happy path.
type Faults struct {
	Status   map[string]int // repo -> forced status (429, 500); unknown repos always answer 404
	SlowMs   int            // delay every response (cut short when the request is cancelled)
	PageSize int            // > 0: split listings into pages linked with Link rel="next"
}

// Options configures New.
type Options struct {
	Faults Faults
	Extra  map[string]string // repo -> listing JSON, served beside the embedded fixtures
}

// Server is the fake tree API.
type Server struct {
	faults   Faults
	trees    map[string][]map[string]any
	mu       sync.Mutex
	requests []string // request paths, in order
	ts       *httptest.Server
}

// Repos are the repos the embedded fixtures list.
func Repos() []string {
	entries, err := testdata.ReadDir("testdata/hf")
	if err != nil {
		panic(err)
	}
	var out []string
	for _, e := range entries {
		out = append(out, strings.Replace(strings.TrimSuffix(e.Name(), ".json"), "__", "/", 1))
	}
	return out
}

// New builds the mock. Start it with Start.
func New(o Options) *Server {
	s := &Server{faults: o.Faults, trees: map[string][]map[string]any{}}
	for _, repo := range Repos() {
		b, err := testdata.ReadFile("testdata/hf/" + strings.Replace(repo, "/", "__", 1) + ".json")
		if err != nil {
			panic(err)
		}
		s.add(repo, b)
	}
	for repo, listing := range o.Extra {
		s.add(repo, []byte(listing))
	}
	return s
}

func (s *Server) add(repo string, b []byte) {
	var files []map[string]any
	if err := json.Unmarshal(b, &files); err != nil {
		panic(fmt.Sprintf("hfmock: %s: %v", repo, err))
	}
	s.trees[repo] = files
}

// Start serves on a loopback port (plain http).
func (s *Server) Start() { s.ts = httptest.NewServer(s) }

// URL is the base to give the sizer ("http://127.0.0.1:port").
func (s *Server) URL() string { return s.ts.URL }

// Close stops the server.
func (s *Server) Close() {
	if s.ts != nil {
		s.ts.Close()
	}
}

// Requests are the paths seen so far.
func (s *Server) Requests() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.requests...)
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.requests = append(s.requests, r.URL.Path)
	s.mu.Unlock()
	if s.faults.SlowMs > 0 {
		select {
		case <-time.After(time.Duration(s.faults.SlowMs) * time.Millisecond):
		case <-r.Context().Done():
			return
		}
	}
	rest, ok := strings.CutPrefix(r.URL.Path, "/api/models/")
	repo, tail, _ := strings.Cut(rest, "/tree/")
	if r.Method != http.MethodGet || !ok || tail != "main" {
		http.Error(w, `{"error":"Not Found"}`, http.StatusNotFound)
		return
	}
	if code := s.faults.Status[repo]; code != 0 {
		w.WriteHeader(code)
		fmt.Fprintf(w, `{"error":"forced %d"}`, code)
		return
	}
	files, ok := s.trees[repo]
	if !ok {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error":"Repository not found"}`)
		return
	}
	if r.URL.Query().Get("recursive") != "true" {
		files = topLevel(files)
	}
	if n := s.faults.PageSize; n > 0 {
		start, _ := strconv.Atoi(r.URL.Query().Get("cursor"))
		end := min(start+n, len(files))
		if end < len(files) {
			w.Header().Set("Link", fmt.Sprintf(`<%s?recursive=true&cursor=%d>; rel="next"`, r.URL.Path, end))
		}
		files = files[start:end]
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(files)
}

// topLevel is what the API answers without recursive=true: root entries only.
func topLevel(files []map[string]any) []map[string]any {
	var out []map[string]any
	for _, f := range files {
		if p, _ := f["path"].(string); !strings.Contains(p, "/") {
			out = append(out, f)
		}
	}
	return out
}

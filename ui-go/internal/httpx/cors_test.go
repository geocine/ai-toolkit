package httpx

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCORS(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusTeapot) })
	h := CORS(ok)
	do := func(method, path, origin string, preflight bool) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, nil)
		if origin != "" {
			req.Header.Set("Origin", origin)
		}
		if preflight {
			req.Header.Set("Access-Control-Request-Method", "POST")
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec
	}

	t.Setenv("AITK_CORS_ORIGINS", "")
	if rec := do("GET", "/v1/jobs", "http://studio.local", false); rec.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("CORS must be off by default")
	}

	t.Setenv("AITK_CORS_ORIGINS", "http://localhost:5174, https://studio.example.com/")
	rec := do("OPTIONS", "/v1/jobs", "https://studio.example.com", true)
	if rec.Code != http.StatusNoContent || rec.Header().Get("Access-Control-Allow-Origin") != "https://studio.example.com" ||
		rec.Header().Get("Access-Control-Allow-Headers") == "" {
		t.Fatalf("preflight: %d %v", rec.Code, rec.Header())
	}
	rec = do("GET", "/v1/jobs", "http://localhost:5174", false)
	if rec.Code != http.StatusTeapot || rec.Header().Get("Access-Control-Allow-Origin") != "http://localhost:5174" {
		t.Fatalf("simple request: %d %v", rec.Code, rec.Header())
	}
	if rec := do("GET", "/v1/jobs", "https://evil.example", false); rec.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("unlisted origin must not be allowed")
	}
	if rec := do("GET", "/index.html", "http://localhost:5174", false); rec.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("the frontend itself is not a CORS path")
	}

	t.Setenv("AITK_CORS_ORIGINS", "*")
	if rec := do("OPTIONS", "/mcp", "https://anything.example", true); rec.Code != http.StatusNoContent {
		t.Fatalf("wildcard preflight %d", rec.Code)
	}
}

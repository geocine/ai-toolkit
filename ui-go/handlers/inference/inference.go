// Package inference ports ui/src/app/api/inference/[...path]/route.ts: a
// reverse proxy to the resident inference engine (a python job that binds an
// ephemeral loopback port and publishes it in its job folder's engine.json).
// Everything under /api/inference/* is forwarded verbatim and a streaming
// upstream body is passed through as it arrives. `?job=<id>` picks an
// engine when several run. /api/inference/status and POST
// /api/inference/outputs/delete are answered here.
package inference

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/engines"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

func Register(mux *http.ServeMux) {
	httpx.Handle(mux, "GET /api/inference/{path...}", proxy)
	httpx.Handle(mux, "POST /api/inference/{path...}", proxy)
	httpx.Handle(mux, "DELETE /api/inference/{path...}", proxy)
}

var hopByHop = map[string]bool{
	"connection": true, "keep-alive": true, "transfer-encoding": true,
	"host": true, "content-length": true, "authorization": true,
}

type engineSummary struct {
	JobID   string `json:"jobId"`
	JobName string `json:"jobName"`
	GpuIDs  string `json:"gpuIds"`
	Status  string `json:"status"`
	Ready   bool   `json:"ready"`
}

func summarize(h engines.Handle) engineSummary {
	return engineSummary{JobID: h.JobID, JobName: h.JobName, GpuIDs: h.GpuIDs, Status: h.Status, Ready: h.Endpoint != nil}
}

func proxy(w http.ResponseWriter, r *http.Request) {
	upstreamPath := r.PathValue("path")
	if upstreamPath == "" {
		// [...path] needs at least one segment
		http.NotFound(w, r)
		return
	}
	query := parseSearchParams(r.URL.RawQuery)
	jobID, _ := query.get("job")

	if upstreamPath == "status" {
		status(w, r, jobID)
		return
	}
	if upstreamPath == "outputs/delete" && r.Method == http.MethodPost {
		deleteOutputs(w, r)
		return
	}

	engine, err := engines.Resolve(r.Context(), jobID)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	if engine == nil || engine.Endpoint == nil {
		var info any
		if engine != nil {
			info = map[string]string{"status": engine.Status}
		}
		httpx.JSON(w, http.StatusServiceUnavailable, struct {
			Error  string `json:"error"`
			Engine any    `json:"engine"`
		}{"no inference engine is running", info})
		return
	}
	query.del("job")
	Forward(w, r, engine.Endpoint, upstreamPath, query.String())
}

func status(w http.ResponseWriter, r *http.Request, jobID string) {
	handles, err := engines.List(r.Context(), true)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	var active *engines.Handle
	if jobID != "" {
		for i := range handles {
			if handles[i].JobID == jobID {
				active = &handles[i]
				break
			}
		}
	} else {
		for i := range handles {
			if handles[i].Endpoint != nil {
				active = &handles[i]
				break
			}
		}
		if active == nil && len(handles) > 0 {
			active = &handles[0]
		}
	}
	var engine any
	if active != nil {
		engine = summarize(*active)
	}
	list := make([]engineSummary, 0, len(handles))
	for _, h := range handles {
		list = append(list, summarize(h))
	}
	httpx.OK(w, struct {
		Running bool            `json:"running"`
		Engine  any             `json:"engine"`
		Engines []engineSummary `json:"engines"`
	}{active != nil && active.Endpoint != nil, engine, list})
}

// deleteOutputs deletes generated files from disk, so history can be cleaned
// up while no engine is running. Only files under an inference job's outputs
// folder are touched.
func deleteOutputs(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	var requested []any
	if raw, err := io.ReadAll(r.Body); err == nil {
		if body, err := jsonx.Parse(raw); err == nil {
			if o, ok := body.(*jsonx.Object); ok {
				requested = o.Array("paths")
			}
		}
	}
	trainingRoot, err := settings.TrainingFolder(ctx)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	trainingFolder, _ := filepath.Abs(trainingRoot)
	jobs, err := store.FindJobs(ctx, store.JobQuery{Where: `"job_type" = 'inference'`})
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	roots := make([]string, 0, len(jobs))
	for _, j := range jobs {
		roots = append(roots, filepath.Join(trainingFolder, j.Name, "outputs"))
	}

	deleted, refused := []string{}, []string{}
	for _, p := range requested {
		s := jsString(p)
		resolved, _ := filepath.Abs(s)
		root := ""
		for _, rt := range roots {
			if strings.HasPrefix(resolved, rt+string(filepath.Separator)) {
				root = rt
				break
			}
		}
		if root == "" {
			refused = append(refused, s)
			continue
		}
		if err := rmForce(resolved); err != nil {
			refused = append(refused, s+": "+err.Error())
			continue
		}
		deleted = append(deleted, s)
		// request folders hold one generation each; drop the folder once empty
		dir := filepath.Dir(resolved)
		if dir != root {
			entries, err := os.ReadDir(dir)
			if err != nil {
				refused = append(refused, s+": "+nodeFSError("scandir", dir, err))
				continue
			}
			if len(entries) == 0 {
				if err := os.Remove(dir); err != nil {
					refused = append(refused, s+": "+nodeFSError("rmdir", dir, err))
				}
			}
		}
	}
	httpx.OK(w, map[string][]string{"deleted": deleted, "refused": refused})
}

// rmForce is fs.promises.rm(p, {force: true}): a missing path is fine, a
// directory is refused (no recursive).
func rmForce(p string) error {
	st, err := os.Lstat(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if st.IsDir() {
		return fmt.Errorf("Path is a directory: rm returned EISDIR (is a directory) %s", p)
	}
	if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func nodeFSError(syscall, path string, err error) string {
	if errors.Is(err, os.ErrNotExist) {
		return fmt.Sprintf("ENOENT: no such file or directory, %s '%s'", syscall, path)
	}
	return err.Error()
}

// jsString is String(v) for a decoded JSON value.
func jsString(v any) string {
	switch x := v.(type) {
	case nil:
		return "null"
	case string:
		return x
	case bool:
		return strconv.FormatBool(x)
	case json.Number:
		if f, err := strconv.ParseFloat(x.String(), 64); err == nil {
			return strconv.FormatFloat(f, 'f', -1, 64)
		}
		return x.String()
	case []any:
		parts := make([]string, len(x))
		for i, e := range x {
			if e != nil {
				parts[i] = jsString(e)
			}
		}
		return strings.Join(parts, ",")
	}
	return "[object Object]"
}

// Forward proxies r to the engine endpoint and streams the response back.
func Forward(w http.ResponseWriter, r *http.Request, ep *engines.Endpoint, upstreamPath, rawQuery string) {
	target := "http://" + ep.Host + ":" + strconv.Itoa(ep.Port) + "/" + encodePath(upstreamPath)
	if rawQuery != "" {
		target += "?" + rawQuery
	}

	var body io.Reader
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		// buffer the request body (JSON or a control-image upload); the
		// response side is what needs to stream
		buf, err := io.ReadAll(r.Body)
		if err != nil {
			httpx.Internal(w, r, err)
			return
		}
		body = strings.NewReader(string(buf))
	}
	req, err := http.NewRequestWithContext(r.Context(), r.Method, target, body)
	if err != nil {
		httpx.JSON(w, http.StatusBadGateway, map[string]string{"error": "engine unreachable: " + err.Error()})
		return
	}
	for k, vs := range r.Header {
		if hopByHop[strings.ToLower(k)] {
			continue
		}
		req.Header[k] = append([]string(nil), vs...)
	}
	req.Header.Set("x-engine-token", ep.Token)

	res, err := client.Do(req)
	if err != nil {
		httpx.JSON(w, http.StatusBadGateway, map[string]string{"error": "engine unreachable: " + err.Error()})
		return
	}
	defer res.Body.Close()

	h := w.Header()
	for k, vs := range res.Header {
		if hopByHop[strings.ToLower(k)] {
			continue
		}
		h[k] = append([]string(nil), vs...)
	}
	h.Set("Cache-Control", "no-store")
	// keep intermediaries from buffering the frame stream
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(res.StatusCode)

	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, err := res.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

// client never follows redirects (fetch redirect: 'manual'), never proxies
// the loopback engine, and leaves compression to the two ends.
var client = &http.Client{
	CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	Transport: &http.Transport{
		Proxy:               nil,
		DisableCompression:  true,
		MaxIdleConnsPerHost: 8,
	},
}

// encodePath percent-encodes the decoded path like the WHATWG URL parser
// does for the path of `http://host/${path}`. `?` and `#` are encoded too
// rather than starting a query or fragment.
func encodePath(p string) string {
	const hex = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(p); i++ {
		c := p[i]
		if c <= 0x20 || c >= 0x7F || strings.IndexByte("\"#<>?`{}", c) >= 0 {
			b.WriteByte('%')
			b.WriteByte(hex[c>>4])
			b.WriteByte(hex[c&15])
			continue
		}
		b.WriteByte(c)
	}
	return b.String()
}

// Command server is the Go backend for ai-toolkit: the HTTP API of the
// upstream Next.js UI (ui/src/app/api) plus its job queue worker (ui/cron),
// against the same database, job folders and Python entry point (run.py).
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/geocine/aitoolkit/handlers/media"
	"github.com/geocine/aitoolkit/internal/bg"
	"github.com/geocine/aitoolkit/internal/bootstrap"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/mcpserver"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/server"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/supervisor"
	"github.com/geocine/aitoolkit/internal/v1api"
	"github.com/geocine/aitoolkit/internal/worker"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == worker.RelayArg {
		os.Exit(worker.RunRelay(os.Args[2:]))
	}

	defaultPort := 8675
	if v, err := strconv.Atoi(os.Getenv("PORT")); err == nil {
		defaultPort = v
	}
	port := flag.Int("port", defaultPort, "port to listen on (env PORT)")
	host := flag.String("host", "0.0.0.0", "address to listen on")
	uiDir := flag.String("ui-dir", os.Getenv("AI_TOOLKIT_UI_DIR"), "static frontend to serve at / (env AI_TOOLKIT_UI_DIR; default ui-go/web/dist when present)")
	noWorker := flag.Bool("no-worker", false, "do not run the job queue worker (e.g. while the Next worker runs)")
	flag.Parse()

	log.Printf("TOOLKIT_ROOT: %s", paths.ToolkitRoot)
	if err := store.Open(paths.DBPath); err != nil {
		log.Fatal(err)
	}
	if err := state.Open(state.Path()); err != nil {
		log.Fatal(err)
	}
	// env configuration (remotes, defaults) before anything uses it
	if err := bootstrap.Apply(context.Background()); err != nil {
		log.Fatalf("bootstrap: %v", err)
	}

	mux := server.NewMux()
	if dir := resolveUIDir(*uiDir); dir != "" {
		log.Printf("Serving frontend from %s", dir)
		mux.Handle("/", spa(dir))
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	// the supervisor runs even with --no-worker: recovery and events matter
	// whichever process launched the job
	go supervisor.Run(ctx)
	go events.Prune(ctx)
	bg.Start(ctx)
	if !*noWorker {
		go worker.Run(ctx)
	}
	go bootstrap.ImportOnBoot(ctx)

	// media.Intercept serves the media GETs before the mux, which would
	// otherwise redirect a proxy-decoded `/api/img//abs/path` (see media).
	handler := httpx.CORS(httpx.Auth(media.Intercept(mux)))
	mcpserver.SetAPI(handler)
	srv := &http.Server{
		Addr:              net.JoinHostPort(*host, strconv.Itoa(*port)),
		Handler:           handler,
		ReadHeaderTimeout: 30 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdown, done := context.WithTimeout(context.Background(), 5*time.Second)
		defer done()
		srv.Shutdown(shutdown)
	}()
	log.Printf("Listening on http://%s", srv.Addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func resolveUIDir(flagValue string) string {
	if flagValue != "" {
		return flagValue
	}
	def := filepath.Join(paths.ToolkitRoot, "ui-go", "web", "dist")
	if st, err := os.Stat(filepath.Join(def, "index.html")); err == nil && !st.IsDir() {
		return def
	}
	return ""
}

// spa serves dir, falling back to index.html for client-side routes.
func spa(dir string) http.Handler {
	files := http.FileServer(http.Dir(dir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") || r.URL.Path == "/mcp" {
			http.NotFound(w, r)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/v1/") || r.URL.Path == "/v1" {
			v1api.Fail(w, v1api.CodeNotFound, "no such endpoint: "+r.Method+" "+r.URL.Path)
			return
		}
		p := filepath.Join(dir, filepath.FromSlash(filepath.Clean("/"+r.URL.Path)))
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			if strings.HasPrefix(r.URL.Path, "/assets/") {
				// Vite puts a content hash in these names
				w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
			} else {
				w.Header().Set("Cache-Control", "no-cache")
			}
			files.ServeHTTP(w, r)
			return
		}
		// revalidate the shell so a new image's UI shows up on the next load
		w.Header().Set("Cache-Control", "no-cache")
		http.ServeFile(w, r, filepath.Join(dir, "index.html"))
	})
}

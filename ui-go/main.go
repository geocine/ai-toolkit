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
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/server"
	"github.com/geocine/aitoolkit/internal/store"
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

	mux := server.NewMux()
	if dir := resolveUIDir(*uiDir); dir != "" {
		log.Printf("Serving frontend from %s", dir)
		mux.Handle("/", spa(dir))
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if !*noWorker {
		go worker.Run(ctx)
	}

	srv := &http.Server{
		Addr: net.JoinHostPort(*host, strconv.Itoa(*port)),
		// media.Intercept serves the media GETs before the mux, which would
		// otherwise redirect a proxy-decoded `/api/img//abs/path` (see media).
		Handler:           httpx.Auth(media.Intercept(mux)),
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
		if strings.HasPrefix(r.URL.Path, "/api/") {
			http.NotFound(w, r)
			return
		}
		p := filepath.Join(dir, filepath.FromSlash(filepath.Clean("/"+r.URL.Path)))
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			files.ServeHTTP(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(dir, "index.html"))
	})
}

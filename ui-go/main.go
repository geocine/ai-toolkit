// Command server is the Go backend for ai-toolkit: the HTTP API of the
// upstream Next.js UI (ui/src/app/api) plus its job queue worker (ui/cron),
// against the same database, job folders and Python entry point (run.py).
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
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
	"github.com/geocine/aitoolkit/internal/controller"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/imagecheck"
	"github.com/geocine/aitoolkit/internal/lifecycle"
	"github.com/geocine/aitoolkit/internal/mcpserver"
	"github.com/geocine/aitoolkit/internal/notify"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/provision/sim"
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
	if len(os.Args) > 1 && os.Args[1] == "image-check" {
		os.Exit(imagecheck.RunCLI(os.Args[2:]))
	}

	defaultPort := 8675
	if v, err := strconv.Atoi(os.Getenv("PORT")); err == nil {
		defaultPort = v
	}
	port := flag.Int("port", defaultPort, "port to listen on (env PORT)")
	host := flag.String("host", "0.0.0.0", "address to listen on")
	uiDir := flag.String("ui-dir", os.Getenv("AI_TOOLKIT_UI_DIR"), "static frontend to serve at / (env AI_TOOLKIT_UI_DIR; default ui-go/web/dist when present)")
	noWorker := flag.Bool("no-worker", false, "do not run the job queue worker (e.g. while the Next worker runs)")
	controllerFlag := flag.Bool("controller", false, "controller mode: no GPU or trainer; creates and tracks GPU servers (env AITK_CONTROLLER=1)")
	dataDir := flag.String("data-dir", "", "controller mode: where the databases live (env AITK_CONTROLLER_DIR; default <user config dir>/aitk-controller)")
	flag.Parse()
	if *controllerFlag || controller.FromEnv() {
		controller.Enable()
	}
	hostSet := false
	flag.Visit(func(f *flag.Flag) { hostSet = hostSet || f.Name == "host" })

	// the checkout this binary was started in, before controller mode moves
	// the root: its studio build is still worth serving
	checkoutRoot := paths.ToolkitRoot
	var controllerDir string
	if controller.Enabled() {
		if !hostSet {
			*host = "127.0.0.1"
		}
		if err := controller.CheckBind(*host, os.Getenv("AI_TOOLKIT_AUTH")); err != nil {
			log.Fatal(err)
		}
		dir, err := openController(*dataDir)
		if err != nil {
			log.Fatalf("controller: %v", err)
		}
		controllerDir = dir
	} else {
		log.Printf("TOOLKIT_ROOT: %s", paths.ToolkitRoot)
		if err := store.Open(paths.DBPath); err != nil {
			log.Fatal(err)
		}
		if err := state.Open(state.Path()); err != nil {
			log.Fatal(err)
		}
	}
	// env configuration (remotes, defaults) before anything uses it
	if err := bootstrap.Apply(context.Background()); err != nil {
		log.Fatalf("bootstrap: %v", err)
	}

	mux := server.NewMux()
	if controller.Enabled() {
		// no console: it drives the trainer
		mux.Handle("/", controllerRoot())
	} else if dir := resolveUIDir(*uiDir); dir != "" {
		log.Printf("Serving frontend from %s", dir)
		mux.Handle("/", spa(dir))
	}
	// the studio at /studio/, so one server URL works from a phone
	studioDir := resolveStudioDir(checkoutRoot)
	if studioDir != "" {
		log.Printf("Serving studio from %s at /studio/", studioDir)
		mux.Handle("/studio/", http.StripPrefix("/studio", spa(studioDir)))
		mux.Handle("/studio", http.RedirectHandler("/studio/", http.StatusMovedPermanently))
		notify.SetLocalStudio(true)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	inner, handler := newHandler(mux)
	// simulate before the watcher's first tick, so simulated machines are
	// refreshed and real ones are left alone
	if sim.Enabled() {
		stopSim, err := sim.Start(ctx, inner)
		if err != nil {
			log.Fatalf("simulate: %v", err)
		}
		defer stopSim()
		sim.WarnIfReachable(*host)
	}

	// when done (AITK_WHEN_DONE) on a machine without a provider API: exit cleanly
	lifecycle.SetExit(cancel)
	if controller.Enabled() {
		// no trainer, job folders or datasets here
		bg.Skip(controller.SkippedLoops...)
	} else {
		// the supervisor runs even with --no-worker: recovery and events matter
		// whichever process launched the job
		go supervisor.Run(ctx)
	}
	go events.Prune(ctx)
	bg.Start(ctx)
	if !controller.Enabled() {
		if !*noWorker {
			go worker.Run(ctx)
		}
		go bootstrap.ImportOnBoot(ctx)
	}

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
	if controller.Enabled() {
		studio := "No studio build found (cd studio && npm run build, or set AITK_STUDIO_DIR)."
		if studioDir != "" {
			studio = "Studio at http://" + net.JoinHostPort(browseHost(*host), strconv.Itoa(*port)) + "/studio/"
		}
		log.Printf("Controller mode: no trainer. Databases in %s. %s", controllerDir, studio)
	}
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

// resolveStudioDir is the first of AITK_STUDIO_DIR (when set, the only
// candidate), studio/dist next to the executable (`go build` in ui-go/) and
// <root>/ui-go/studio/dist that holds a build.
func resolveStudioDir(root string) string {
	var dirs []string
	if dir := os.Getenv("AITK_STUDIO_DIR"); dir != "" {
		dirs = []string{dir}
	} else {
		if exe, err := os.Executable(); err == nil {
			dirs = append(dirs, filepath.Join(filepath.Dir(exe), "studio", "dist"))
		}
		dirs = append(dirs, filepath.Join(root, "ui-go", "studio", "dist"))
	}
	for _, dir := range dirs {
		if st, err := os.Stat(filepath.Join(dir, "index.html")); err == nil && !st.IsDir() {
			return dir
		}
	}
	return ""
}

// newHandler is the request chain. media.Intercept serves the media GETs
// before the mux, which would otherwise redirect a proxy-decoded
// `/api/img//abs/path` (see media). controller.Gate refuses trainer routes on
// a controller (a no-op otherwise). inner is the chain below Auth: simulated
// GPU servers serve it behind their own token, so a simulated machine on a
// controller is gated too.
func newHandler(mux http.Handler) (inner, handler http.Handler) {
	inner = controller.Gate(media.Intercept(mux))
	return inner, httpx.CORS(httpx.Auth(inner))
}

// openController opens controller mode's databases in its data dir: it moves
// the root there (so datasets, output and files are its own empty folders),
// creates the upstream database from the embedded schema, replaces one an
// older version made, and keeps the files private to this user.
func openController(dataDir string) (string, error) {
	dir, err := controller.DataDir(dataDir)
	if err != nil {
		return "", err
	}
	paths.SetRoot(dir)
	if _, err := os.Stat(paths.DBPath); errors.Is(err, os.ErrNotExist) {
		if err := store.Create(paths.DBPath); err != nil {
			return "", err
		}
	}
	if err := store.Open(paths.DBPath); err != nil {
		if !store.IsOutOfDate(err) {
			return "", err
		}
		// an older binary made it; a controller has no checkout to migrate with
		backup, err := store.Recreate(paths.DBPath)
		if err != nil {
			return "", fmt.Errorf("recreate %s: %w", paths.DBPath, err)
		}
		log.Printf("Controller database was made by an older version: moved it to %s and created a new one with its settings.", backup)
		if err := store.Open(paths.DBPath); err != nil {
			return "", err
		}
	}
	// AI_TOOLKIT_GO_DB, else aitk_go.db in the data dir (the root moved)
	goDB := state.Path()
	if err := state.Open(goDB); err != nil {
		return "", err
	}
	// the fork database holds the provider keys; the 0700 dir covers files
	// SQLite creates later (a no-op on Windows)
	for _, db := range []string{goDB, paths.DBPath} {
		for _, side := range []string{"", "-wal", "-shm"} {
			if _, err := os.Stat(db + side); err == nil {
				if err := os.Chmod(db+side, 0o600); err != nil {
					return "", err
				}
			}
		}
	}
	return dir, nil
}

// controllerRoot stands in for the console at / on a controller: the exact
// path / goes to the studio, anything else is a miss.
func controllerRoot() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.Redirect(w, r, "/studio/", http.StatusFound)
			return
		}
		notFound(w, r)
	})
}

// browseHost is the address a browser on this machine opens for a listen
// address (every interface → 127.0.0.1).
func browseHost(host string) string {
	if host == "" || host == "0.0.0.0" || host == "::" {
		return "127.0.0.1"
	}
	return host
}

// notFound answers a path nothing serves: the /v1 error under /v1, a plain
// 404 otherwise.
func notFound(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/v1/") || r.URL.Path == "/v1" {
		v1api.Fail(w, v1api.CodeNotFound, "no such endpoint: "+r.Method+" "+r.URL.Path)
		return
	}
	http.NotFound(w, r)
}

// spa serves dir, falling back to index.html for client-side routes.
func spa(dir string) http.Handler {
	files := http.FileServer(http.Dir(dir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if p := r.URL.Path; strings.HasPrefix(p, "/api/") || p == "/mcp" || strings.HasPrefix(p, "/v1/") || p == "/v1" {
			notFound(w, r)
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

package notify

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/lifecycle"
	"github.com/geocine/aitoolkit/internal/paths"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
)

func init() { backoffBase = 10 * time.Millisecond }

func TestParseSink(t *testing.T) {
	cases := []struct{ url, kind, secret string }{
		{"https://ntfy.sh/aitk-secret-topic", SinkNtfy, "secret-topic"},
		{"ntfy+https://ntfy.example.com/mytopic", SinkNtfy, "topic"},
		{"https://discord.com/api/webhooks/123456789/tokenpart", SinkDiscord, "tokenpart"},
		{"telegram://123456:AAH-secret@-1001234", SinkTelegram, "AAH-secret"},
		{"pushover://azGDORePK8gMaC0QOYAMyEEuzJnyUi@uQiRzpo4DXghDmr9QzzfQu27cmVRsG", SinkPushover, "azGDORe"},
		{"pushover://azGDORePK8gMaC0QOYAMyEEuzJnyUi@uQiRzpo4DXghDmr9QzzfQu27cmVRsG/phone", SinkPushover, "uQiRzpo"},
		{"https://hooks.example.com/notify?key=s3cr3t", SinkWebhook, "s3cr3t"},
		{"webhook+https://ntfy.sh/forced", SinkWebhook, "forced"},
	}
	for _, c := range cases {
		kind, target := SinkInfo(c.url)
		if kind != c.kind {
			t.Errorf("%s: kind %s, want %s", c.url, kind, c.kind)
		}
		if strings.Contains(target, c.secret) {
			t.Errorf("%s: target %q leaks %q", c.url, target, c.secret)
		}
	}
	for _, bad := range []string{"", "ftp://x/y", "telegram://nocolon@1", "telegram://1:2@", "https://ntfy.sh/", "notaurl",
		"pushover://short@uQiRzpo4DXghDmr9QzzfQu27cmVRsG", "pushover://azGDORePK8gMaC0QOYAMyEEuzJnyUi", "pushover://@"} {
		if _, err := NewSink(bad, "api"); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	if SinkID("a") != SinkID("a") || SinkID("a") == SinkID("b") {
		t.Fatal("sink ids not stable")
	}
}

type got struct {
	method, path, query, ctype string
	body                       []byte
}

func receiver(t *testing.T, status func(n int) int) (*httptest.Server, func() []got) {
	t.Helper()
	var mu sync.Mutex
	var all []got
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		all = append(all, got{r.Method, r.URL.Path, r.URL.RawQuery, r.Header.Get("Content-Type"), b})
		n := len(all)
		mu.Unlock()
		code := http.StatusOK
		if status != nil {
			code = status(n)
		}
		w.WriteHeader(code)
		io.WriteString(w, `{"ok":true}`)
	}))
	t.Cleanup(srv.Close)
	return srv, func() []got {
		mu.Lock()
		defer mu.Unlock()
		return append([]got(nil), all...)
	}
}

func jpegBytes(t *testing.T) []byte {
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	var buf bytes.Buffer
	jpeg.Encode(&buf, img, nil)
	return buf.Bytes()
}

func parts(t *testing.T, g got) map[string][]byte {
	t.Helper()
	_, params, err := mime.ParseMediaType(g.ctype)
	if err != nil {
		t.Fatalf("content type %q: %v", g.ctype, err)
	}
	out := map[string][]byte{}
	mr := multipart.NewReader(bytes.NewReader(g.body), params["boundary"])
	for {
		p, err := mr.NextPart()
		if err != nil {
			break
		}
		b, _ := io.ReadAll(p)
		out[p.FormName()] = b
	}
	return out
}

func TestSinksFormat(t *testing.T) {
	srv, all := receiver(t, nil)
	telegramAPI = srv.URL
	defer func() { telegramAPI = "https://api.telegram.org" }()
	pushoverAPI = srv.URL + "/1/messages.json"
	defer func() { pushoverAPI = "https://api.pushover.net/1/messages.json" }()
	img := jpegBytes(t)
	m := Message{Kind: KindCompleted, Level: LevelOK, Title: "lora: completed", Body: "Step 20/20.\nKept on s3.",
		URL: "https://pod-8675.proxy.runpod.net/studio/#/runs/j1", JobID: "j1", JobName: "lora", At: time.Now()}
	sinks := []string{
		"ntfy+" + srv.URL + "/topic1",
		"discord+" + srv.URL + "/api/webhooks/1/tok",
		"telegram://123:secret@-42",
		"webhook+" + srv.URL + "/hook",
		"pushover://azGDORePK8gMaC0QOYAMyEEuzJnyUi@uQiRzpo4DXghDmr9QzzfQu27cmVRsG/phone",
	}
	ctx := context.Background()
	for _, u := range sinks {
		for _, withImage := range []bool{false, true} {
			mm := m
			if withImage {
				mm.Image = img
			}
			if err := deliverOnce(ctx, Sink{ID: SinkID(u), URL: u}, mm); err != nil {
				t.Fatalf("%s: %v", u, err)
			}
		}
	}
	g := all()
	if len(g) != 10 {
		t.Fatalf("%d requests", len(g))
	}
	// ntfy: text body, then PUT with the attachment and the text as ?message=
	if g[0].method != "POST" || g[0].path != "/topic1" || !strings.Contains(string(g[0].body), "Kept on s3.") ||
		!strings.Contains(g[0].query, "title=lora%3A+completed") || !strings.Contains(g[0].query, "click=https") {
		t.Fatalf("ntfy text: %+v", g[0])
	}
	if g[1].method != "PUT" || !bytes.Equal(g[1].body, img) || !strings.Contains(g[1].query, "filename=sample.jpg") ||
		!strings.Contains(g[1].query, "message=Step+20%2F20.%0AKept") {
		t.Fatalf("ntfy image: %s %s", g[1].method, g[1].query)
	}
	// discord: JSON embed, then multipart with the file referenced by the embed
	var d struct {
		Embeds []struct {
			Title, URL string
			Image      *struct{ URL string }
		}
	}
	if json.Unmarshal(g[2].body, &d) != nil || d.Embeds[0].Title != "lora: completed" || d.Embeds[0].URL != m.URL {
		t.Fatalf("discord: %s", g[2].body)
	}
	p := parts(t, g[3])
	if json.Unmarshal(p["payload_json"], &d) != nil || d.Embeds[0].Image == nil || d.Embeds[0].Image.URL != "attachment://sample.jpg" ||
		!bytes.Equal(p["files[0]"], img) {
		t.Fatalf("discord multipart: %v", p)
	}
	// telegram: sendMessage, then sendPhoto with a caption
	if g[4].path != "/bot123:secret/sendMessage" || !strings.Contains(string(g[4].body), `"chat_id":"-42"`) ||
		!strings.Contains(string(g[4].body), "lora: completed") {
		t.Fatalf("telegram: %s %s", g[4].path, g[4].body)
	}
	p = parts(t, g[5])
	if g[5].path != "/bot123:secret/sendPhoto" || string(p["chat_id"]) != "-42" || !bytes.Equal(p["photo"], img) ||
		!strings.Contains(string(p["caption"]), m.URL) {
		t.Fatalf("telegram photo: %s %v", g[5].path, p)
	}
	// webhook: JSON with the image as base64
	var w map[string]any
	if json.Unmarshal(g[7].body, &w) != nil || w["kind"] != "completed" || w["job_id"] != "j1" || w["url"] != m.URL {
		t.Fatalf("webhook: %s", g[7].body)
	}
	if b, _ := base64.StdEncoding.DecodeString(w["image_jpeg_base64"].(string)); !bytes.Equal(b, img) {
		t.Fatal("webhook image")
	}
	// pushover: the link in its url field, the image as the attachment
	p = parts(t, g[8])
	if g[8].path != "/1/messages.json" || string(p["token"]) != "azGDORePK8gMaC0QOYAMyEEuzJnyUi" || string(p["device"]) != "phone" ||
		string(p["title"]) != "lora: completed" || string(p["message"]) != m.Body || string(p["url"]) != m.URL || p["attachment"] != nil {
		t.Fatalf("pushover: %s %v", g[8].path, p)
	}
	if p = parts(t, g[9]); !bytes.Equal(p["attachment"], img) {
		t.Fatalf("pushover image: %v", p)
	}
}

func TestRetryAndFilter(t *testing.T) {
	// 503, 429, then OK: delivered on the third attempt
	srv, all := receiver(t, func(n int) int {
		switch n {
		case 1:
			return http.StatusServiceUnavailable
		case 2:
			return http.StatusTooManyRequests
		}
		return http.StatusOK
	})
	cfgMu.Lock()
	cfg = Config{Sinks: []Sink{{ID: "s1", URL: "webhook+" + srv.URL + "/a"}}, Events: []string{KindFailed}}
	cfgMu.Unlock()
	Send(Message{Kind: KindStarted, Title: "filtered out"})
	Send(Message{Kind: KindFailed, Title: "kept"})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	Flush(ctx)
	g := all()
	if len(g) != 3 {
		t.Fatalf("%d requests", len(g))
	}
	for _, r := range g {
		if !strings.Contains(string(r.body), `"title":"kept"`) {
			t.Fatalf("%s", r.body)
		}
	}
	// a 400 is final: one attempt, nothing left in flight
	srv2, all2 := receiver(t, func(int) int { return http.StatusBadRequest })
	cfgMu.Lock()
	cfg = Config{Sinks: []Sink{{ID: "s2", URL: "webhook+" + srv2.URL + "/b"}}}
	cfgMu.Unlock()
	Send(Message{Kind: KindFailed, Title: "rejected"})
	Flush(ctx)
	if n := len(all2()); n != 1 || inflight.Load() != 0 {
		t.Fatalf("%d requests, %d in flight", n, inflight.Load())
	}
	// SendNow reports what the sink said
	res := SendNow(ctx, Message{Kind: KindTest, Title: "t"}, nil)
	if len(res) != 1 || res[0].OK || !strings.Contains(res[0].Detail, "400") || res[0].Kind != SinkWebhook {
		t.Fatalf("%+v", res)
	}
}

const ddl = `
CREATE TABLE "Settings" ("id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "key" TEXT NOT NULL, "value" TEXT NOT NULL);
CREATE UNIQUE INDEX "Settings_key_key" ON "Settings"("key");
CREATE TABLE "Queue" ("id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "gpu_ids" TEXT NOT NULL, "is_running" BOOLEAN NOT NULL DEFAULT false);
CREATE TABLE "Job" ("id" TEXT NOT NULL PRIMARY KEY, "name" TEXT NOT NULL, "gpu_ids" TEXT NOT NULL, "job_config" TEXT NOT NULL,
  "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "updated_at" DATETIME NOT NULL, "status" TEXT NOT NULL DEFAULT 'stopped',
  "stop" BOOLEAN NOT NULL DEFAULT false, "return_to_queue" BOOLEAN NOT NULL DEFAULT false, "step" INTEGER NOT NULL DEFAULT 0,
  "total_steps" INTEGER, "info" TEXT NOT NULL DEFAULT '', "speed_string" TEXT NOT NULL DEFAULT '', "queue_position" INTEGER NOT NULL DEFAULT 0,
  "pid" INTEGER, "job_type" TEXT NOT NULL DEFAULT 'train', "job_ref" TEXT, "save_now" BOOLEAN NOT NULL DEFAULT false,
  "sample_now" BOOLEAN NOT NULL DEFAULT false);
CREATE UNIQUE INDEX "Job_name_key" ON "Job"("name");
`

func setupDB(t *testing.T) string {
	t.Helper()
	root := testutil.TempDir(t)
	dbPath := filepath.Join(root, "aitk_db.db")
	db, err := sql.Open("sqlite", store.FileURI(dbPath))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(ddl); err != nil {
		t.Fatal(err)
	}
	db.Close()
	paths.ToolkitRoot, paths.DBPath = root, dbPath
	paths.DefaultTrainFolder = filepath.Join(root, "output")
	paths.DefaultDatasetsFolder = filepath.Join(root, "datasets")
	paths.DefaultDataRoot = filepath.Join(root, "data")
	settings.Flush()
	if err := store.Open(dbPath); err != nil {
		t.Fatal(err)
	}
	if err := state.Open(filepath.Join(root, "aitk_go.db")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.DB.Close(); state.DB.Close() })
	return root
}

func writePNG(t *testing.T, p string, w, h int, c color.Color) {
	t.Helper()
	os.MkdirAll(filepath.Dir(p), 0o755)
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	png.Encode(f, img)
	f.Close()
}

func TestEnvConfig(t *testing.T) {
	setupDB(t)
	ctx := context.Background()
	t.Setenv("AITK_NOTIFY", "https://ntfy.sh/one, telegram://1:x@2, bogus://nope")
	t.Setenv("AITK_NOTIFY_EVENTS", "completed,failed")
	t.Setenv("AITK_PUBLIC_URL", "https://pod-8675.proxy.runpod.net/")
	t.Setenv("AITK_NOTIFY_SAMPLE_EVERY", "5m")
	if err := applyEnv(ctx); err != nil {
		t.Fatal(err)
	}
	c := Current()
	if len(c.Sinks) != 2 || c.SampleEvery != 300 || len(c.Events) != 2 {
		t.Fatalf("%+v", c)
	}
	// an API sink survives a reboot; env sinks are replaced
	api, _ := NewSink("https://hooks.example.com/x", "api")
	Update(ctx, func(c *Config) error { c.Sinks = append(c.Sinks, api); return nil })
	t.Setenv("AITK_NOTIFY", "https://ntfy.sh/two")
	t.Setenv("AITK_NOTIFY_EVENTS", "all")
	applyEnv(ctx)
	c = Current()
	if len(c.Sinks) != 2 || c.Sinks[0].ID != api.ID || c.Sinks[1].ID != SinkID("https://ntfy.sh/two") || c.Events != nil {
		t.Fatalf("%+v", c)
	}
	SetLocalStudio(true)
	defer SetLocalStudio(false)
	if l := JobLink(c, "j 1"); l != "https://pod-8675.proxy.runpod.net/studio/#/runs/j%201" {
		t.Fatal(l)
	}
	c.StudioURL = "https://studio.example.com"
	if l := LifecycleLink(c); l != "https://studio.example.com/#/lifecycle" {
		t.Fatal(l)
	}
}

func TestComposeJobEvents(t *testing.T) {
	root := setupDB(t)
	ctx := context.Background()
	srv, all := receiver(t, nil)
	cfgMu.Lock()
	cfg, loaded = Config{Sinks: []Sink{{ID: "w", URL: "webhook+" + srv.URL + "/w"}}, PublicURL: "https://srv.example"}, true
	cfgMu.Unlock()
	runCtx = ctx
	j, err := store.CreateJob(ctx, store.NewJob{Name: "lora", GpuIDs: "0", JobConfig: "{}"})
	if err != nil {
		t.Fatal(err)
	}
	total := 20
	store.UpdateJob(ctx, j.ID, store.Fields{"status": "completed", "step": 20, "total_steps": total})
	folder := filepath.Join(root, "output", "lora")
	writePNG(t, filepath.Join(folder, "samples", "1700000000000__000000020_0.png"), 64, 32, color.RGBA{200, 0, 0, 255})
	writePNG(t, filepath.Join(folder, "samples", "1700000000000__000000020_1.png"), 32, 64, color.RGBA{0, 200, 0, 255})
	writePNG(t, filepath.Join(folder, "samples", "1700000000000__000000010_0.png"), 32, 32, color.RGBA{0, 0, 200, 255})
	os.WriteFile(filepath.Join(folder, "lora.safetensors"), []byte("weights"), 0o644)
	os.WriteFile(filepath.Join(folder, "log.txt"), []byte("loading\nstep 1\r step 2\r step 3\nTraceback (most recent call last):\nValueError: boom\n"), 0o644)
	state.StartAttempt(ctx, j.ID, 0, false)
	state.EndAttempt(ctx, j.ID, "completed", "", 20)

	c := newComposer()
	ev := func(typ string, data any) events.Event {
		b, _ := json.Marshal(data)
		id := j.ID
		return events.Event{Type: typ, JobID: &id, Data: b, At: time.Now()}
	}
	c.handle(ctx, ev(events.JobState, map[string]any{"to": jobstate.Completed}))
	c.handle(ctx, ev(events.JobState, map[string]any{"to": jobstate.Failed, "info": "Error: boom"}))
	c.handle(ctx, ev("job.sync", map[string]any{"remote": "s3", "action": "checkpoint", "step": 20,
		"files": []string{"lora_000000020.safetensors", "optimizer.pt"}}))
	c.handle(ctx, ev("job.sync", map[string]any{"remote": "s3", "action": "checkpoint", "files": []string{}, "error": "network down"}))
	c.handle(ctx, ev("job.sync", map[string]any{"remote": "s3", "action": "checkpoint", "files": []string{}, "error": "again"})) // rate limited
	dl := time.Now().Add(10 * time.Minute)
	name := "lora"
	life := func(s lifecycle.Status) {
		b, _ := json.Marshal(s)
		c.handle(ctx, events.Event{Type: lifecycle.Event, Data: b})
	}
	life(lifecycle.Status{Mode: "destroy", State: lifecycle.StateCountdown, Deadline: &dl, JobName: &name, Provider: "runpod", Reason: "job lora completed"})
	life(lifecycle.Status{Mode: "destroy", State: lifecycle.StateCountdown, Deadline: &dl, Provider: "runpod"}) // same: not repeated
	life(lifecycle.Status{Mode: "destroy", State: lifecycle.StateHeld, Provider: "runpod", Reason: "results only on this server"})

	fctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	Flush(fctx)
	var msgs []map[string]any
	for _, g := range all() {
		var m map[string]any
		json.Unmarshal(g.body, &m)
		msgs = append(msgs, m)
	}
	if len(msgs) != 6 {
		for _, m := range msgs {
			t.Log(m["title"])
		}
		t.Fatalf("%d messages", len(msgs))
	}
	done := msgs[0]
	if done["title"] != "lora: completed" || done["url"] != "https://srv.example/#/jobs/"+j.ID ||
		!strings.Contains(done["body"].(string), "Step 20/20.") || !strings.Contains(done["body"].(string), "only on this server") {
		t.Fatalf("%v", done)
	}
	// the completion carries the newest step's samples as a 2-up grid
	b, _ := base64.StdEncoding.DecodeString(done["image_jpeg_base64"].(string))
	img, err := jpeg.Decode(bytes.NewReader(b))
	if err != nil || img.Bounds().Dx() != 2*512+4 || img.Bounds().Dy() != 512 {
		t.Fatalf("grid: %v %v", err, img)
	}
	if f := msgs[1]["body"].(string); !strings.Contains(f, "Error: boom") || !strings.Contains(f, "ValueError: boom") ||
		!strings.Contains(f, "loading\n step 3\nTraceback") {
		t.Fatalf("failed body: %q", f)
	}
	if msgs[2]["kind"] != KindCheckpoint || !strings.Contains(msgs[2]["body"].(string), "safe on s3") {
		t.Fatalf("%v", msgs[2])
	}
	if msgs[3]["kind"] != KindSyncFailed {
		t.Fatalf("%v", msgs[3])
	}
	if !strings.HasPrefix(msgs[4]["title"].(string), "Destroying this server in 10 min") || msgs[4]["kind"] != KindLifecycle {
		t.Fatalf("%v", msgs[4])
	}
	// the repeated countdown was dropped; held follows it
	if !strings.HasPrefix(msgs[5]["title"].(string), "Results only on this server, not destroying") || msgs[5]["level"] != LevelError {
		t.Fatalf("%v", msgs[5])
	}
}

func TestGrid(t *testing.T) {
	dir := t.TempDir()
	var ps []string
	for i := 0; i < 4; i++ {
		p := filepath.Join(dir, string(rune('a'+i))+".png")
		writePNG(t, p, 100, 50, color.White)
		ps = append(ps, p)
	}
	b, err := Grid(ps)
	if err != nil {
		t.Fatal(err)
	}
	img, _ := jpeg.Decode(bytes.NewReader(b))
	if img.Bounds().Dx() != 804 || img.Bounds().Dy() != 804 {
		t.Fatal(img.Bounds())
	}
	b, _ = Grid(ps[:1])
	img, _ = jpeg.Decode(bytes.NewReader(b))
	if img.Bounds().Dx() != 100 || img.Bounds().Dy() != 50 {
		t.Fatal(img.Bounds()) // small images are not scaled up
	}
	if _, err := Grid([]string{filepath.Join(dir, "missing.png")}); err == nil {
		t.Fatal("no images accepted")
	}
}

package notify

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/image/draw"

	"github.com/geocine/aitoolkit/internal/artifacts"
	"github.com/geocine/aitoolkit/internal/events"
	"github.com/geocine/aitoolkit/internal/jobstate"
	"github.com/geocine/aitoolkit/internal/lifecycle"
	"github.com/geocine/aitoolkit/internal/portable"
	"github.com/geocine/aitoolkit/internal/remote"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/state"
	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/thumbs"
	"github.com/geocine/aitoolkit/internal/uploads"
)

// PublicURL is the server's public base URL: AITK_PUBLIC_URL (or the
// configured public_url), else derived from the provider (RunPod's HTTP
// proxy, Vast.ai's mapped port, QuickPod's mapped port), else "".
func PublicURL(c Config) string {
	if c.PublicURL != "" {
		return strings.TrimRight(c.PublicURL, "/")
	}
	port := env("PORT")
	if port == "" {
		port = "8675"
	}
	if id := env("RUNPOD_POD_ID"); id != "" {
		return "https://" + id + "-" + port + ".proxy.runpod.net"
	}
	if ip, p := env("PUBLIC_IPADDR"), env("VAST_TCP_PORT_"+port); ip != "" && p != "" {
		return "http://" + ip + ":" + p
	}
	if ip, p := env("PUBLIC_IPADDR"), env("QUICKPOD_PORT_"+port); ip != "" && p != "" {
		return "http://" + ip + ":" + p
	}
	return ""
}

// StudioURL is where tap-through links open: AITK_STUDIO_URL (or the
// configured studio_url), else <public url>/studio/ when this server serves
// the studio; "" when neither is known. Always ends in "/".
func StudioURL(c Config) string {
	if c.StudioURL != "" {
		return strings.TrimRight(c.StudioURL, "/") + "/"
	}
	cfgMu.Lock()
	local := localStudio
	cfgMu.Unlock()
	if pub := PublicURL(c); pub != "" && local {
		return pub + "/studio/"
	}
	return ""
}

// JobLink opens the run in the studio (#/runs/<id>), else in the console
// (#/jobs/<id>).
func JobLink(c Config, jobID string) string {
	if s := StudioURL(c); s != "" {
		return s + "#/runs/" + url.PathEscape(jobID)
	}
	if p := PublicURL(c); p != "" {
		return p + "/#/jobs/" + url.PathEscape(jobID)
	}
	return ""
}

// LifecycleLink opens the when-done controls (studio #/lifecycle; the
// console's settings when only it is known).
func LifecycleLink(c Config) string {
	if s := StudioURL(c); s != "" {
		return s + "#/lifecycle"
	}
	if p := PublicURL(c); p != "" {
		return p + "/#/settings"
	}
	return ""
}

// Run sends notifications for events until ctx ends.
func Run(ctx context.Context) {
	runCtx = ctx
	if err := applyEnv(ctx); err != nil {
		log.Printf("notify: %v", err)
	}
	lifecycle.OnBeforeAct(Flush)
	ch, cancel := events.Subscribe(events.Filter{Types: map[string]bool{
		events.JobAttempt: true, events.JobState: true, events.JobSample: true, portable.JobSync: true,
		events.UploadState: true, lifecycle.Event: true,
	}})
	defer cancel()
	c := newComposer()
	for {
		select {
		case <-ctx.Done():
			return
		case e := <-ch:
			c.handle(ctx, e)
		}
	}
}

// composer turns events into messages; it remembers just enough to avoid
// repeats (rate limits, the last lifecycle state).
type composer struct {
	mu          sync.Mutex
	lastSample  map[string]time.Time
	pending     map[string]*time.Timer // per job: a sample burst being collected
	lastSyncErr map[string]time.Time
	lastLife    string // state + deadline of the last lifecycle message
}

func newComposer() *composer {
	return &composer{lastSample: map[string]time.Time{}, pending: map[string]*time.Timer{},
		lastSyncErr: map[string]time.Time{}}
}

// sampleSettle waits for the rest of a sampling burst (one image per prompt)
// before sending its grid; syncErrEvery spaces sync failure messages.
var (
	sampleSettle = 20 * time.Second
	syncErrEvery = 30 * time.Minute
)

func (c *composer) handle(ctx context.Context, e events.Event) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("notify: panic handling %s: %v", e.Type, r)
		}
	}()
	if e.Type == lifecycle.Event {
		c.lifecycle(e)
		return
	}
	if e.JobID == nil {
		return
	}
	j, err := store.GetJob(ctx, *e.JobID)
	if err != nil || j == nil {
		return
	}
	cfg := Current()
	if len(cfg.Sinks) == 0 {
		return
	}
	base := Message{JobID: j.ID, JobName: j.Name, URL: JobLink(cfg, j.ID), Level: LevelInfo}
	switch e.Type {
	case events.JobAttempt:
		var a state.Attempt
		if json.Unmarshal(e.Data, &a) != nil || a.EndedAt != nil {
			return
		}
		m := base
		switch {
		case a.N <= 1:
			m.Kind, m.Title = KindStarted, j.Name+": started"
			m.Body = fmt.Sprintf("Training on GPU %s.%s", j.GpuIDs, stepsText(" Steps: ", 0, j.TotalSteps))
		case a.Auto:
			m.Kind, m.Title, m.Level = KindResumed, j.Name+": auto-resumed", LevelWarn
			m.Body = fmt.Sprintf("Resumed by itself after an interruption (attempt %d), from step %d.", a.N, a.StartStep)
		default:
			m.Kind, m.Title = KindResumed, j.Name+": resumed"
			m.Body = fmt.Sprintf("Attempt %d, from step %d.", a.N, a.StartStep)
		}
		Send(m)
	case events.JobState:
		var d struct {
			To   string `json:"to"`
			Info string `json:"info"`
		}
		if json.Unmarshal(e.Data, &d) != nil {
			return
		}
		switch d.To {
		case jobstate.Completed:
			m := base
			m.Kind, m.Title, m.Level = KindCompleted, j.Name+": completed", LevelOK
			m.Body = completedBody(ctx, j)
			m.Image = latestGrid(ctx, j)
			Send(m)
		case jobstate.Failed:
			m := base
			m.Kind, m.Title, m.Level = KindFailed, j.Name+": failed", LevelError
			m.Body = strings.TrimSpace(d.Info + stepsText("\nAt step ", j.Step, j.TotalSteps))
			if t := logTail(ctx, j, 12, 1200); t != "" {
				m.Body += "\n\nLast log lines:\n" + t
			}
			Send(m)
		case jobstate.Interrupted:
			m := base
			m.Kind, m.Title, m.Level = KindInterrupted, j.Name+": interrupted", LevelWarn
			m.Body = strings.TrimSpace(d.Info + stepsText("\nAt step ", j.Step, j.TotalSteps))
			if auto, _, _, err := state.EffectivePolicy(ctx, j.ID); err == nil && auto {
				m.Body += "\nAuto-resume is on: it continues from the last checkpoint."
			} else {
				m.Body += "\nAuto-resume is off: start it again to continue from the last checkpoint."
			}
			Send(m)
		}
	case events.JobSample:
		c.sample(j)
	case portable.JobSync:
		var d struct {
			Remote string   `json:"remote"`
			Action string   `json:"action"`
			Step   int      `json:"step"`
			Files  []string `json:"files"`
			Error  *string  `json:"error"`
		}
		if json.Unmarshal(e.Data, &d) != nil {
			return
		}
		m := base
		switch {
		case d.Action == "lease_lost":
			m.Kind, m.Title, m.Level = KindLeaseLost, j.Name+": taken over by another server", LevelWarn
			m.Body = "Another server holds this job's lease, so it was stopped here. " + deref(d.Error)
			Send(m)
		case d.Error != nil:
			c.mu.Lock()
			last := c.lastSyncErr[j.ID]
			if time.Since(last) < syncErrEvery {
				c.mu.Unlock()
				return
			}
			c.lastSyncErr[j.ID] = time.Now()
			c.mu.Unlock()
			m.Kind, m.Title, m.Level = KindSyncFailed, j.Name+": sync failed", LevelWarn
			m.Body = fmt.Sprintf("Could not push to %s (it retries): %s", d.Remote, *d.Error)
			Send(m)
		case (d.Action == portable.ReasonCheckpoint || d.Action == portable.ReasonFinal) && hasCheckpoint(d.Files):
			c.mu.Lock()
			delete(c.lastSyncErr, j.ID)
			c.mu.Unlock()
			m.Kind, m.Title, m.Level = KindCheckpoint, j.Name+": checkpoint synced", LevelOK
			m.Body = fmt.Sprintf("Step %d is safe on %s (%s).", d.Step, d.Remote, strings.Join(d.Files, ", "))
			if d.Action == portable.ReasonFinal {
				m.Title = j.Name + ": final state synced"
			}
			Send(m)
		}
	case events.UploadState:
		var u uploads.Upload
		if json.Unmarshal(e.Data, &u) != nil || u.State != uploads.Done {
			return
		}
		name := u.RemoteID
		if r, err := remote.Get(ctx, u.RemoteID); err == nil && r != nil {
			name = r.Name
		}
		m := base
		m.Kind, m.Title, m.Level = KindCheckpoint, j.Name+": uploaded", LevelOK
		m.Body = fmt.Sprintf("%s (%s) is on %s.", filepath.Base(u.Path), humanBytes(u.Bytes), name)
		Send(m)
	}
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func hasCheckpoint(files []string) bool {
	for _, f := range files {
		switch strings.ToLower(filepath.Ext(f)) {
		case ".safetensors", ".ckpt", ".bin":
			return true
		case ".pt":
			if f != "optimizer.pt" {
				return true
			}
		}
	}
	return false
}

func stepsText(prefix string, step int, total *int) string {
	switch {
	case total != nil && step > 0:
		return fmt.Sprintf("%s%d/%d.", prefix, step, *total)
	case total != nil:
		return fmt.Sprintf("%s%d.", prefix, *total)
	case step > 0:
		return fmt.Sprintf("%s%d.", prefix, step)
	}
	return ""
}

func humanBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.0f KB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

func humanDuration(d time.Duration) string {
	d = d.Round(time.Minute)
	if d < time.Minute {
		return "under a minute"
	}
	h, m := int(d.Hours()), int(d.Minutes())%60
	if h == 0 {
		return fmt.Sprintf("%d min", m)
	}
	return fmt.Sprintf("%dh %02dm", h, m)
}

func jobFolder(ctx context.Context, j *store.Job) string {
	root, err := settings.TrainingFolder(ctx)
	if err != nil {
		return ""
	}
	return filepath.Join(root, j.Name)
}

// completedBody: steps, time, the result file and where it is kept.
func completedBody(ctx context.Context, j *store.Job) string {
	var b strings.Builder
	b.WriteString(strings.TrimSpace(stepsText("Step ", j.Step, j.TotalSteps)))
	if as, err := state.Attempts(ctx, j.ID); err == nil && len(as) > 0 {
		end := time.Now()
		if last := as[len(as)-1]; last.EndedAt != nil {
			end = *last.EndedAt
		}
		fmt.Fprintf(&b, " Took %s", humanDuration(end.Sub(as[0].StartedAt)))
		if len(as) > 1 {
			fmt.Fprintf(&b, " over %d attempts", len(as))
		}
		b.WriteString(".")
	}
	if folder := jobFolder(ctx, j); folder != "" {
		if cps, err := artifacts.Checkpoints(folder); err == nil && len(cps) > 0 {
			sort.SliceStable(cps, func(a, c int) bool { return cps[a].Modified.Before(cps[c].Modified) })
			cp := cps[len(cps)-1]
			fmt.Fprintf(&b, "\nResult: %s (%s).", cp.Name, humanBytes(cp.Size))
		}
	}
	pol, _ := state.GetPolicy(ctx, j.ID)
	_, _, up, _ := state.EffectivePolicy(ctx, j.ID)
	var where []string
	if pol.SyncRemote != nil && *pol.SyncRemote != "" {
		where = append(where, "synced to "+remoteName(ctx, *pol.SyncRemote))
	}
	if up != "" {
		where = append(where, "uploaded to "+remoteName(ctx, up))
	}
	if len(where) > 0 {
		b.WriteString("\nKept off this server: " + strings.Join(where, ", ") + ".")
	} else {
		b.WriteString("\nThe results are only on this server (no sync or upload remote).")
	}
	return strings.TrimSpace(b.String())
}

func remoteName(ctx context.Context, ref string) string {
	if r, err := remote.Get(ctx, ref); err == nil && r != nil {
		return r.Name
	}
	return ref
}

// logTail is the last n non-empty lines of the job's log.txt, at most max chars.
func logTail(ctx context.Context, j *store.Job, n, max int) string {
	folder := jobFolder(ctx, j)
	if folder == "" {
		return ""
	}
	f, err := os.Open(filepath.Join(folder, "log.txt"))
	if err != nil {
		return ""
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return ""
	}
	size := min(st.Size(), 16<<10)
	buf := make([]byte, size)
	if _, err := f.ReadAt(buf, st.Size()-size); err != nil {
		return ""
	}
	// progress bars rewrite their line with \r: keep the last state of each
	var lines []string
	for _, l := range strings.Split(strings.ReplaceAll(string(buf), "\r\n", "\n"), "\n") {
		if i := strings.LastIndex(l, "\r"); i >= 0 {
			l = l[i+1:]
		}
		if strings.TrimSpace(l) != "" {
			lines = append(lines, strings.TrimRight(l, " \t"))
		}
	}
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	out := strings.Join(lines, "\n")
	if r := []rune(out); len(r) > max {
		out = "…" + string(r[len(r)-max+1:])
	}
	return out
}

// sample collects a sampling burst and sends one grid, at most once per
// sample interval per job (the first burst of a job always goes).
func (c *composer) sample(j *store.Job) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, waiting := c.pending[j.ID]; waiting {
		return
	}
	if last, ok := c.lastSample[j.ID]; ok && time.Since(last) < sampleEvery(Current()) {
		return
	}
	id := j.ID
	c.pending[id] = time.AfterFunc(sampleSettle, func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("notify: panic composing samples: %v", r)
			}
		}()
		c.mu.Lock()
		delete(c.pending, id)
		c.lastSample[id] = time.Now()
		c.mu.Unlock()
		ctx := runCtx
		job, err := store.GetJob(ctx, id)
		if err != nil || job == nil {
			return
		}
		img, step := gridFor(ctx, job)
		if img == nil {
			return
		}
		cfg := Current()
		Send(Message{Kind: KindSample, Level: LevelInfo, JobID: job.ID, JobName: job.Name, URL: JobLink(cfg, job.ID),
			Title: job.Name + ": new samples", Body: fmt.Sprintf("Samples at step %d%s", step, stepsText(" of ", 0, job.TotalSteps)),
			Image: img})
	})
}

func latestGrid(ctx context.Context, j *store.Job) []byte {
	img, _ := gridFor(ctx, j)
	return img
}

var imageExt = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true}

// gridFor renders the newest step's samples (up to 4) as one JPEG.
func gridFor(ctx context.Context, j *store.Job) ([]byte, int) {
	folder := jobFolder(ctx, j)
	if folder == "" {
		return nil, 0
	}
	smp, err := artifacts.Samples(folder)
	if err != nil || len(smp) == 0 {
		return nil, 0
	}
	newest := -1
	for _, s := range smp {
		if s.Step != nil && *s.Step > newest && imageExt[strings.ToLower(filepath.Ext(s.Name))] {
			newest = *s.Step
		}
	}
	var pick []string
	for _, s := range smp {
		if s.Step != nil && *s.Step == newest && imageExt[strings.ToLower(filepath.Ext(s.Name))] {
			pick = append(pick, s.Path)
		}
	}
	sort.Strings(pick)
	if len(pick) > 4 {
		pick = pick[:4]
	}
	b, err := Grid(pick)
	if err != nil {
		log.Printf("notify: sample grid for %s: %v", j.Name, err)
		return nil, 0
	}
	return b, newest
}

// Grid draws up to 4 images into one JPEG (1: fitted into 768 px like the
// MCP preview, thumbs.Fit; 2: side by side; 3-4: 2x2), each fitted into its
// cell on a neutral background. Decoders come with internal/thumbs.
func Grid(paths []string) ([]byte, error) {
	var imgs []image.Image
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			continue
		}
		img, err := thumbs.Decode(f)
		f.Close()
		if err == nil {
			imgs = append(imgs, img)
		}
	}
	if len(imgs) == 0 {
		return nil, fmt.Errorf("no decodable images among %d sample(s)", len(paths))
	}
	if len(imgs) == 1 {
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, thumbs.Fit(imgs[0], 768), &jpeg.Options{Quality: 82}); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil
	}
	cols, rows, cell := 2, 1, 512
	switch {
	case len(imgs) == 2:
		cols, cell = 2, 512
	case len(imgs) > 2:
		cols, rows, cell = 2, 2, 400
	}
	const gap = 4
	w, h := cols*cell+(cols-1)*gap, rows*cell+(rows-1)*gap
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(dst, dst.Bounds(), &image.Uniform{color.RGBA{0x18, 0x18, 0x1b, 0xff}}, image.Point{}, draw.Src)
	for i, img := range imgs {
		x0, y0 := (i%cols)*(cell+gap), (i/cols)*(cell+gap)
		cellRect := image.Rect(x0, y0, min(x0+cell, w), min(y0+cell, h))
		draw.CatmullRom.Scale(dst, fitRect(img.Bounds(), cellRect), img, img.Bounds(), draw.Over, nil)
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 82}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// fitRect is the largest rect of src's aspect centred in cell.
func fitRect(src, cell image.Rectangle) image.Rectangle {
	sw, sh := float64(src.Dx()), float64(src.Dy())
	cw, ch := float64(cell.Dx()), float64(cell.Dy())
	s := min(cw/sw, ch/sh)
	w, h := int(sw*s), int(sh*s)
	x := cell.Min.X + (cell.Dx()-w)/2
	y := cell.Min.Y + (cell.Dy()-h)/2
	return image.Rect(x, y, x+w, y+h)
}

// lifecycle turns when-done state changes into messages: the countdown
// (with the link to cancel it), held (results not safe), cancelled, acting,
// done and failed.
func (c *composer) lifecycle(e events.Event) {
	var s lifecycle.Status
	if json.Unmarshal(e.Data, &s) != nil {
		return
	}
	key := s.State
	if s.Deadline != nil {
		key += "@" + s.Deadline.Format(time.RFC3339)
	}
	c.mu.Lock()
	if key == c.lastLife {
		c.mu.Unlock()
		return
	}
	prev := c.lastLife
	c.lastLife = key
	c.mu.Unlock()
	cfg := Current()
	verb, done := "Destroying", "destroyed"
	if s.Mode == lifecycle.ModeStop {
		verb, done = "Stopping", "stopped"
	}
	dry := ""
	if s.DryRun {
		dry = " (dry run)"
	}
	m := Message{Kind: KindLifecycle, Level: LevelWarn, URL: LifecycleLink(cfg)}
	if s.JobID != nil {
		m.JobID = *s.JobID
	}
	if s.JobName != nil {
		m.JobName = *s.JobName
	}
	switch s.State {
	case lifecycle.StateCountdown:
		if s.Deadline == nil {
			return
		}
		left := time.Until(*s.Deadline)
		in := humanDuration(left)
		if left < 2*time.Minute {
			in = fmt.Sprintf("%ds", max(0, int(left.Round(time.Second).Seconds())))
		}
		m.Title = fmt.Sprintf("%s this server in %s%s", verb, in, dry)
		if strings.HasPrefix(prev, lifecycle.StateCountdown) {
			m.Title = fmt.Sprintf("%s this server at %s UTC%s (extended)", verb, s.Deadline.UTC().Format("15:04"), dry)
		}
		m.Body = s.Reason + "\nOpen to keep it."
	case lifecycle.StateHeld:
		m.Level = LevelError
		m.Title = "Results only on this server, not " + strings.ToLower(verb)
		m.Body = s.Reason + "\nThe server stays up: save the results, then stop it yourself."
	case lifecycle.StateCancelled:
		if !strings.HasPrefix(prev, lifecycle.StateCountdown) && !strings.HasPrefix(prev, lifecycle.StateVerifying) &&
			!strings.HasPrefix(prev, lifecycle.StateActing) {
			return
		}
		m.Level, m.Title, m.Body = LevelInfo, "Shutdown cancelled", s.Reason
	case lifecycle.StateWaiting, lifecycle.StateOff:
		// a queued job (or mode keep) cancelling the countdown
		if !strings.HasPrefix(prev, lifecycle.StateCountdown) && !strings.HasPrefix(prev, lifecycle.StateVerifying) {
			return
		}
		m.Level, m.Title, m.Body = LevelInfo, "Shutdown cancelled", s.Reason
	case lifecycle.StateActing:
		m.Title, m.Body = verb+" this server now"+dry, s.Reason
	case lifecycle.StateDone:
		m.Level, m.Title, m.Body = LevelOK, "Server "+done+dry, s.Reason
	case lifecycle.StateFailed:
		m.Level, m.Title, m.Body = LevelError, "Could not end the server", s.Reason
	default:
		return
	}
	if s.Provider != "" {
		m.Body += "\nProvider: " + s.Provider + "."
	}
	Send(m)
}

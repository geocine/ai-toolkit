package photocheck

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"io/fs"
	"log"
	"os"
	"runtime"
	"strconv"
	"sync"

	"github.com/geocine/aitoolkit/internal/state"
)

// bigPixels is where a decode counts as big: at most bigDecodes of them run
// at once, because the server shares its memory with the training run.
const (
	bigPixels  = 24_000_000
	bigDecodes = 2
)

// task is one photo being analysed. done closes when m, sha and err are set.
type task struct {
	done   chan struct{}
	path   string
	rel    string
	info   fs.FileInfo
	urgent bool
	m      Metrics
	sha    string
	err    error
}

// pool analyses photos in the background: min(4, max(1, NumCPU/4)) workers
// (AITK_PHOTOCHECK_WORKERS overrides), at most two big decodes at once, the
// same file (path, size, mtime) analysed once however often it is asked
// for, and photos a request waits on ahead of warm-up work.
type pool struct {
	once    sync.Once
	mu      sync.Mutex
	wake    *sync.Cond
	urgent  []*task
	warm    []*task
	byKey   map[string]*task
	big     chan struct{}
	workers int
}

var analysis = &pool{}

// poolWorkers is the worker count for this machine.
func poolWorkers() int {
	if v, err := strconv.Atoi(os.Getenv("AITK_PHOTOCHECK_WORKERS")); err == nil && v > 0 {
		return min(v, 64)
	}
	return min(4, max(1, runtime.NumCPU()/4))
}

func (p *pool) start() {
	p.once.Do(func() {
		p.wake = sync.NewCond(&p.mu)
		p.byKey = map[string]*task{}
		p.big = make(chan struct{}, bigDecodes)
		p.workers = poolWorkers()
		for range p.workers {
			go p.work()
		}
	})
}

func taskKey(path string, info fs.FileInfo) string {
	return fmt.Sprintf("%s|%d|%d", path, info.Size(), info.ModTime().UnixNano())
}

// analyse queues one photo and returns its task. A photo already queued or
// running returns the same task; asking urgently for one queued for warm-up
// moves it to the front.
func (p *pool) analyse(path, rel string, info fs.FileInfo, urgent bool) *task {
	p.start()
	key := taskKey(path, info)
	p.mu.Lock()
	defer p.mu.Unlock()
	if t := p.byKey[key]; t != nil {
		if urgent && !t.urgent {
			for i, w := range p.warm {
				if w == t {
					p.warm = append(p.warm[:i], p.warm[i+1:]...)
					t.urgent = true
					p.urgent = append(p.urgent, t)
					break
				}
			}
		}
		return t
	}
	t := &task{done: make(chan struct{}), path: path, rel: rel, info: info, urgent: urgent}
	p.byKey[key] = t
	if urgent {
		p.urgent = append(p.urgent, t)
	} else {
		p.warm = append(p.warm, t)
	}
	p.wake.Signal()
	return t
}

func (p *pool) next() *task {
	p.mu.Lock()
	defer p.mu.Unlock()
	for len(p.urgent) == 0 && len(p.warm) == 0 {
		p.wake.Wait()
	}
	var t *task
	if len(p.urgent) > 0 {
		t, p.urgent = p.urgent[0], p.urgent[1:]
	} else {
		t, p.warm = p.warm[0], p.warm[1:]
	}
	return t
}

func (p *pool) work() {
	for {
		t := p.next()
		p.run(t)
		p.mu.Lock()
		delete(p.byKey, taskKey(t.path, t.info))
		p.mu.Unlock()
		close(t.done)
	}
}

// run analyses one photo: its sha256 (through the file_hashes cache), the
// cached metrics for that content when another path already measured it,
// else a fresh Measure, which is then cached by content.
func (p *pool) run(t *task) {
	defer func() {
		if r := recover(); r != nil {
			t.err = fmt.Errorf("analyse %s: %v", t.rel, r)
		}
	}()
	ctx := context.Background()
	sha, err := state.FileSHA256(ctx, t.path, t.info)
	if err != nil {
		t.err = err
		return
	}
	t.sha = sha
	if rows, err := state.GetPhotoMetrics(ctx, []string{sha}, MetricsVersion); err == nil {
		if m, ok := decodeMetrics(rows[sha], t.rel); ok {
			t.m = m
			return
		}
	}
	if pixels(t.path) > bigPixels {
		p.big <- struct{}{}
		defer func() { <-p.big }()
	}
	m, err := Measure(t.path, t.rel)
	if err != nil {
		t.err = err
		return
	}
	if m.SHA256 == "" {
		m.SHA256 = sha
	}
	t.sha, t.m = m.SHA256, m
	if err := storeMetrics(ctx, m); err != nil {
		log.Printf("photocheck: cache %s: %v", t.rel, err)
	}
}

// settle waits until the analysis of the photo at path, queued or running,
// is done, so a fix can move or remove the file: Windows refuses that while
// a worker has it open. A photo queued for warm-up moves to the front.
func (p *pool) settle(ctx context.Context, path string, info fs.FileInfo) {
	if info == nil {
		return
	}
	p.start()
	p.mu.Lock()
	t := p.byKey[taskKey(path, info)]
	if t != nil && !t.urgent {
		for i, w := range p.warm {
			if w == t {
				p.warm = append(p.warm[:i], p.warm[i+1:]...)
				t.urgent = true
				p.urgent = append(p.urgent, t)
				break
			}
		}
	}
	p.mu.Unlock()
	if t == nil {
		return
	}
	select {
	case <-t.done:
	case <-ctx.Done():
	}
}

// holdBig takes one of the big-decode slots while a fix decodes the photo
// at path, when its header says it is big, so fixes and the analysis
// together stay within bigDecodes. release gives the slot back.
func (p *pool) holdBig(path string) (release func()) {
	if pixels(path) <= bigPixels {
		return func() {}
	}
	p.start()
	p.big <- struct{}{}
	return func() { <-p.big }
}

// pixels is the pixel count a file's header declares, 0 when unknown.
func pixels(path string) int64 {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	cfg, _, err := image.DecodeConfig(bufio.NewReader(f))
	if err != nil {
		return 0
	}
	return int64(cfg.Width) * int64(cfg.Height)
}

// storeMetrics caches m by content, without its file name.
func storeMetrics(ctx context.Context, m Metrics) error {
	if m.SHA256 == "" {
		return nil
	}
	m.File = ""
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return state.PutPhotoMetrics(ctx, m.SHA256, MetricsVersion, data)
}

// decodeMetrics reads a cached row for the file rel.
func decodeMetrics(data []byte, rel string) (Metrics, bool) {
	if len(data) == 0 {
		return Metrics{}, false
	}
	var m Metrics
	if json.Unmarshal(data, &m) != nil || m.Version != MetricsVersion {
		return Metrics{}, false
	}
	m.File = rel
	return m, true
}

// measureNow analyses one photo on the caller's goroutine: a fix that has
// just written it needs its metrics before the next op.
func measureNow(abs, rel string) (Metrics, string, error) {
	st, err := os.Stat(abs)
	if err != nil {
		return Metrics{}, "", err
	}
	analysis.start()
	t := &task{path: abs, rel: rel, info: st}
	analysis.run(t)
	t.m.File = rel
	return t.m, t.sha, t.err
}

package system

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/paths"
)

// Port of ui/src/app/api/scripts/route.ts: run a ui_scripts/*.py script with
// the toolkit's python and return (or stream as NDJSON) its output.

const scriptTimeout = 20 * time.Minute

var (
	// Only allow flat script names (no path separators, no traversal).
	scriptNameRe = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_.-]*\.py$`)
	argKeyRe     = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
)

func uiScriptsRoot() string { return filepath.Join(paths.ToolkitRoot, "ui_scripts") }

func resolveScriptPath(raw any) string {
	name, ok := raw.(string)
	if !ok {
		return ""
	}
	name = strings.TrimSpace(name)
	if !scriptNameRe.MatchString(name) {
		return ""
	}
	root := uiScriptsRoot()
	target, err := filepath.Abs(filepath.Join(root, name))
	if err != nil {
		return ""
	}
	if !strings.HasPrefix(target, strings.TrimSuffix(root, string(filepath.Separator))+string(filepath.Separator)) {
		return ""
	}
	st, err := os.Stat(target)
	if err != nil || !st.Mode().IsRegular() {
		return ""
	}
	return target
}

// jsString is String(v) for the JSON scalars args may hold.
func jsString(v any) (string, bool) {
	switch x := v.(type) {
	case string:
		return x, true
	case bool:
		return strconv.FormatBool(x), true
	case json.Number:
		f, err := strconv.ParseFloat(x.String(), 64)
		if err != nil {
			return x.String(), true
		}
		return jsNumberString(f), true
	}
	return "", false
}

// jsNumberString is Number.prototype.toString().
func jsNumberString(f float64) string {
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case f == 0:
		return "0"
	}
	if a := math.Abs(f); a >= 1e-6 && a < 1e21 {
		return strconv.FormatFloat(f, 'f', -1, 64)
	}
	mant, exp, _ := strings.Cut(strconv.FormatFloat(f, 'e', -1, 64), "e")
	sign, digits := exp[:1], strings.TrimLeft(exp[1:], "0")
	return mant + "e" + sign + digits
}

// jsKeyOrder is Object.entries order: array-index keys ascending first, then
// the rest in insertion order.
func jsKeyOrder(keys []string) []string {
	var idx, rest []string
	for _, k := range keys {
		if n, err := strconv.ParseUint(k, 10, 32); err == nil && n < math.MaxUint32 && strconv.FormatUint(n, 10) == k {
			idx = append(idx, k)
		} else {
			rest = append(rest, k)
		}
	}
	sort.Slice(idx, func(i, j int) bool {
		a, _ := strconv.ParseUint(idx[i], 10, 32)
		b, _ := strconv.ParseUint(idx[j], 10, 32)
		return a < b
	})
	return append(idx, rest...)
}

// normalizeArgs: a positional list, or an object that becomes --key value
// pairs. Every value is stringified before being passed to the process.
func normalizeArgs(raw any) ([]string, string) {
	switch x := raw.(type) {
	case nil:
		return []string{}, ""
	case []any:
		out := []string{}
		for _, v := range x {
			if v == nil {
				continue
			}
			s, ok := jsString(v)
			if !ok {
				return nil, "args entries must be string|number|boolean"
			}
			out = append(out, s)
		}
		return out, ""
	case *jsonx.Object:
		out := []string{}
		for _, key := range jsKeyOrder(x.Keys()) {
			if !argKeyRe.MatchString(key) {
				return nil, "invalid arg key: " + key
			}
			v, _ := x.Get(key)
			flag := "--" + key
			switch val := v.(type) {
			case bool:
				if val {
					out = append(out, flag)
				}
			case nil:
			case string, json.Number:
				s, _ := jsString(val)
				out = append(out, flag, s)
			default:
				return nil, "args." + key + " must be string|number|boolean"
			}
		}
		return out, ""
	}
	return nil, "args must be an array or object"
}

// parseResult parses the last non-empty stdout line as JSON when it looks
// like an object or array, so scripts can return structured data.
func parseResult(stdout string) any {
	lines := regexp.MustCompile(`\r?\n`).Split(strings.TrimRightFunc(stdout, isJSSpace), -1)
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimFunc(lines[i], isJSSpace)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "{") || strings.HasPrefix(line, "[") {
			v, err := jsonx.Parse([]byte(line))
			if err != nil {
				return nil
			}
			return v
		}
		return nil
	}
	return nil
}

func isJSSpace(r rune) bool {
	// JS whitespace and line terminators (U+0085 is the one unicode.IsSpace
	// adds that JS does not)
	return r != 0x85 && (unicode.IsSpace(r) || r == 0xFEFF)
}

func scriptCommand(ctx context.Context, scriptPath string, args []string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, paths.ResolvePython(), append([]string{"-u", scriptPath}, args...)...)
	cmd.Dir = paths.ToolkitRoot
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1", "PYTHONIOENCODING=utf-8")
	hideWindow(cmd)
	// CommandContext kills with SIGKILL (TerminateProcess on Windows)
	return cmd
}

// exitInfo is (exitCode, signal) as Node's 'close' event reports them.
func exitInfo(err error, timedOut bool) (*int, *string) {
	if err == nil {
		zero := 0
		return &zero, nil
	}
	if timedOut {
		sig := "SIGKILL"
		return nil, &sig
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		if sig := signalOf(exitErr); sig != "" {
			return nil, &sig
		}
		code := exitErr.ExitCode()
		return &code, nil
	}
	return nil, nil
}

type runResult struct {
	OK       bool    `json:"ok"`
	ExitCode *int    `json:"exitCode"`
	Signal   *string `json:"signal"`
	Stdout   string  `json:"stdout"`
	Stderr   string  `json:"stderr"`
	Result   any     `json:"result"`
	TimedOut bool    `json:"timedOut"`
	Error    *string `json:"error,omitempty"`
}

func runBuffered(ctx context.Context, scriptPath string, args []string) runResult {
	ctx, cancel := context.WithTimeout(ctx, scriptTimeout)
	defer cancel()
	cmd := scriptCommand(ctx, scriptPath, args)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Start(); err != nil {
		msg := err.Error()
		return runResult{OK: false, Stdout: "", Stderr: "", TimedOut: false, Error: &msg}
	}
	err := cmd.Wait()
	timedOut := errors.Is(ctx.Err(), context.DeadlineExceeded)
	code, sig := exitInfo(err, timedOut)
	res := runResult{
		OK:       !timedOut && code != nil && *code == 0,
		ExitCode: code,
		Signal:   sig,
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Result:   parseResult(stdout.String()),
		TimedOut: timedOut,
	}
	if timedOut {
		msg := "Script timed out after 20 minutes"
		res.Error = &msg
	}
	return res
}

type streamChunk struct {
	Type string `json:"type"`
	Data string `json:"data"`
}

type streamError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type streamExit struct {
	Type     string  `json:"type"`
	ExitCode *int    `json:"exitCode"`
	Signal   *string `json:"signal"`
	OK       bool    `json:"ok"`
	TimedOut bool    `json:"timedOut"`
	Result   any     `json:"result"`
	Stderr   string  `json:"stderr"`
}

// utf8Stream turns byte chunks into text without splitting a multi-byte
// character across two events.
type utf8Stream struct{ pending []byte }

func (u *utf8Stream) decode(p []byte) string {
	b := append(u.pending, p...)
	cut := len(b)
	for i := len(b) - 1; i >= 0 && i >= len(b)-utf8.UTFMax; i-- {
		if utf8.RuneStart(b[i]) {
			if !utf8.FullRune(b[i:]) {
				cut = i
			}
			break
		}
	}
	u.pending = append([]byte(nil), b[cut:]...)
	return string(b[:cut])
}

// runStreaming streams NDJSON: one JSON object per line.
func runStreaming(w http.ResponseWriter, r *http.Request, scriptPath string, args []string) {
	h := w.Header()
	h.Set("Content-Type", "application/x-ndjson; charset=utf-8")
	h.Set("Cache-Control", "no-cache, no-transform")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	var mu sync.Mutex
	send := func(v any) {
		b, _ := json.Marshal(v)
		mu.Lock()
		defer mu.Unlock()
		w.Write(append(b, '\n'))
		if flusher != nil {
			flusher.Flush()
		}
	}

	// the client going away kills the script, like the stream's cancel()
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	cmd := scriptCommand(ctx, scriptPath, args)
	stdoutPipe, err1 := cmd.StdoutPipe()
	stderrPipe, err2 := cmd.StderrPipe()
	if err := errors.Join(err1, err2); err != nil {
		send(streamError{Type: "error", Message: err.Error()})
		return
	}
	if err := cmd.Start(); err != nil {
		send(streamError{Type: "error", Message: err.Error()})
		return
	}

	var timedOut bool
	var timeMu sync.Mutex
	timer := time.AfterFunc(scriptTimeout, func() {
		timeMu.Lock()
		timedOut = true
		timeMu.Unlock()
		send(streamError{Type: "error", Message: "Script timed out after 20 minutes"})
		cmd.Process.Kill()
	})
	defer timer.Stop()

	var stdoutAll, stderrAll strings.Builder
	var bufMu sync.Mutex
	pump := func(rd io.Reader, kind string, all *strings.Builder, wg *sync.WaitGroup) {
		defer wg.Done()
		var dec utf8Stream
		buf := make([]byte, 64*1024)
		for {
			n, err := rd.Read(buf)
			if n > 0 {
				text := dec.decode(buf[:n])
				if text != "" {
					bufMu.Lock()
					all.WriteString(text)
					bufMu.Unlock()
					send(streamChunk{Type: kind, Data: text})
				}
			}
			if err != nil {
				if rest := string(dec.pending); rest != "" {
					bufMu.Lock()
					all.WriteString(rest)
					bufMu.Unlock()
					send(streamChunk{Type: kind, Data: rest})
				}
				return
			}
		}
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go pump(stdoutPipe, "stdout", &stdoutAll, &wg)
	go pump(stderrPipe, "stderr", &stderrAll, &wg)
	wg.Wait()
	err := cmd.Wait()
	timer.Stop()

	timeMu.Lock()
	to := timedOut
	timeMu.Unlock()
	if r.Context().Err() != nil {
		return // client is gone
	}
	code, sig := exitInfo(err, to)
	send(streamExit{
		Type:     "exit",
		ExitCode: code,
		Signal:   sig,
		OK:       !to && code != nil && *code == 0,
		TimedOut: to,
		Result:   parseResult(stdoutAll.String()),
		Stderr:   stderrAll.String(),
	})
}

func scripts(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		httpx.Error(w, http.StatusBadRequest, "Invalid JSON body")
		return
	}
	body, err := jsonx.Parse(raw)
	if err != nil {
		httpx.Error(w, http.StatusBadRequest, "Invalid JSON body")
		return
	}
	// body?.script / body?.args / body?.stream
	get := func(key string) any {
		if o, ok := body.(*jsonx.Object); ok {
			v, _ := o.Get(key)
			return v
		}
		return nil
	}

	scriptPath := resolveScriptPath(get("script"))
	if scriptPath == "" {
		httpx.Error(w, http.StatusBadRequest, "Invalid or unknown script. Must be a *.py file inside ui_scripts/.")
		return
	}
	args, argErr := normalizeArgs(get("args"))
	if argErr != "" {
		httpx.Error(w, http.StatusBadRequest, argErr)
		return
	}
	if stream, ok := get("stream").(bool); ok && stream {
		runStreaming(w, r, scriptPath, args)
		return
	}
	// a buffered run is not tied to the client, like upstream's
	result := runBuffered(context.WithoutCancel(r.Context()), scriptPath, args)
	status := http.StatusInternalServerError
	if result.OK {
		status = http.StatusOK
	} else if result.TimedOut {
		status = http.StatusGatewayTimeout
	}
	httpx.JSON(w, status, result)
}

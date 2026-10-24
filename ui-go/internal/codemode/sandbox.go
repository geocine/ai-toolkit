// Package codemode runs agent-written TypeScript against the /v1 API in a
// sandbox ("code mode", docs/headless.md): the code is transpiled with esbuild
// and executed in goja with nothing but the `aitk` client, console and sleep.
// API calls are synchronous Go functions, so `await` on them settles in
// goja's microtask queue and the whole script finishes inside one run.
package codemode

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/dop251/goja"
	"github.com/evanw/esbuild/pkg/api"
)

// Limits (docs/headless.md, MCP).
const (
	DefaultTimeout = 120 * time.Second
	MaxTimeout     = 600 * time.Second
	MaxCalls       = 256
	MaxLogBytes    = 1 << 20
	MaxResultBytes = 4 << 20
	MaxTextBytes   = 1 << 20
	MaxBinaryBytes = 16 << 20 // decoded base64 per call (ModeBase64, each ModeMultipart file)
)

// Response is what an in-process /v1 call returned.
type Response struct {
	Status int
	Header http.Header
	Body   []byte
	// Size is the full body length; more than len(Body) when the caller kept
	// only the first MaxResponseBytes (0 means len(Body))
	Size int64
}

// Caller performs one /v1 request in-process (with the caller's credentials).
type Caller func(ctx context.Context, method, path string, query url.Values, body []byte, contentType string) (Response, error)

// Options configure one run.
type Options struct {
	Timeout time.Duration // 0 = DefaultTimeout, capped at MaxTimeout
	Call    Caller
}

// ErrorInfo describes why a run failed.
type ErrorInfo struct {
	Code    string `json:"code"` // syntax_error, runtime_error, timeout, api_error, call_limit, result_too_large
	Message string `json:"message"`
	Status  int    `json:"status,omitempty"`
}

// Result is {result, logs, error}.
type Result struct {
	Result     json.RawMessage `json:"result"`
	Logs       string          `json:"logs"`
	Error      *ErrorInfo      `json:"error"`
	Calls      int             `json:"calls"`
	DurationMs int64           `json:"duration_ms"`
}

const wrapperPrefix = "(async () => {\n"

// Transpile turns the agent's TypeScript body into the script goja runs: an
// async IIFE, so top-level await and `return` work. Lowered to ES2017 (goja
// runs async functions natively; newer syntax is down-leveled).
func Transpile(code string) (string, error) {
	res := api.Transform(wrapperPrefix+code+"\n})()", api.TransformOptions{
		Loader:     api.LoaderTS,
		Target:     api.ES2017,
		Sourcefile: "aitk_run.ts",
		LogLevel:   api.LogLevelSilent,
	})
	if len(res.Errors) > 0 {
		msgs := make([]string, 0, len(res.Errors))
		for _, m := range res.Errors {
			msg := m.Text
			if m.Location != nil {
				msg = fmt.Sprintf("line %d:%d: %s", m.Location.Line-1, m.Location.Column+1, m.Text)
			}
			msgs = append(msgs, msg)
		}
		return "", errors.New(strings.Join(msgs, "; "))
	}
	return string(res.Code), nil
}

type run struct {
	vm       *goja.Runtime
	ctx      context.Context
	call     Caller
	calls    int
	logs     bytes.Buffer
	logsFull bool
	errCtor  goja.Value
}

// prelude defines AitkError, jobs.wait (on top of jobs.get) and a JSON
// structuredClone (goja has none; agents copy configs with it).
const prelude = `
if (typeof structuredClone === 'undefined') {
  globalThis.structuredClone = v => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));
}
class AitkError extends Error {
  constructor(code, status, message) { super(message); this.name = 'AitkError'; this.code = code; this.status = status; }
}
globalThis.AitkError = AitkError;
aitk.jobs.wait = async function (id, options) {
  const o = options || {};
  const until = o.until || ['completed', 'failed', 'stopped', 'interrupted'];
  const poll = Math.max(500, o.poll_ms || 5000);
  const deadline = o.timeout_ms ? Date.now() + o.timeout_ms : Infinity;
  for (;;) {
    const job = await aitk.jobs.get(id);
    if (until.includes(job.state)) return job;
    if (Date.now() + poll > deadline) throw new AitkError('timeout', 0, 'job ' + id + ' still ' + job.state + ' after ' + o.timeout_ms + ' ms');
    await sleep(poll);
  }
};
aitk.bundles.wait = async function (id, options) {
  const o = options || {};
  const poll = Math.max(500, o.poll_ms || 3000);
  const deadline = o.timeout_ms ? Date.now() + o.timeout_ms : Infinity;
  const get = String(id).startsWith('bpsh_') ? aitk.bundles.pushStatus : aitk.bundles.importStatus;
  for (;;) {
    const t = await get(id);
    if (t.state === 'done' || t.state === 'failed') return t;
    if (Date.now() + poll > deadline) throw new AitkError('timeout', 0, 'bundle task ' + id + ' still ' + t.state + ' after ' + o.timeout_ms + ' ms');
    await sleep(poll);
  }
};
`

// preludes are feature JS (e.g. a group's wait helper) run right after
// prelude, in registration order; feature files add them from init().
var preludes []string

// RegisterPrelude adds JS that every run executes after the built-in prelude,
// with aitk and AitkError defined. An error in it fails the run.
func RegisterPrelude(js string) { preludes = append(preludes, js) }

// Run executes code and never panics; problems come back in Result.Error.
func Run(ctx context.Context, code string, opt Options) (res Result) {
	start := time.Now()
	timeout := opt.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	timeout = min(timeout, MaxTimeout)
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	r := &run{vm: goja.New(), ctx: ctx, call: opt.Call}
	defer func() {
		res.Logs = r.logs.String()
		res.Calls = r.calls
		res.DurationMs = time.Since(start).Milliseconds()
		if res.Result == nil {
			res.Result = json.RawMessage("null")
		}
		if p := recover(); p != nil {
			res.Error = &ErrorInfo{Code: "runtime_error", Message: fmt.Sprint(p)}
		}
	}()

	js, err := Transpile(code)
	if err != nil {
		res.Error = &ErrorInfo{Code: "syntax_error", Message: err.Error()}
		return
	}
	if err := r.install(); err != nil {
		res.Error = &ErrorInfo{Code: "runtime_error", Message: err.Error()}
		return
	}

	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-ctx.Done():
			r.vm.Interrupt("timeout")
		case <-stop:
		}
	}()

	v, err := r.vm.RunScript("aitk_run.js", js)
	if err != nil {
		res.Error = r.errorInfo(err)
		return
	}
	promise, ok := v.Export().(*goja.Promise)
	if !ok {
		res.Error = &ErrorInfo{Code: "runtime_error", Message: "script did not produce a promise"}
		return
	}
	switch promise.State() {
	case goja.PromiseStateFulfilled:
		raw, err := r.toJSON(promise.Result())
		if err != nil {
			res.Error = &ErrorInfo{Code: "runtime_error", Message: "result is not JSON-serializable: " + err.Error()}
			return
		}
		if len(raw) > MaxResultBytes {
			res.Error = &ErrorInfo{Code: "result_too_large", Message: fmt.Sprintf("result is %d bytes (max %d); return less", len(raw), MaxResultBytes)}
			return
		}
		res.Result = raw
	case goja.PromiseStateRejected:
		res.Error = r.rejection(promise.Result())
	default:
		if ctx.Err() != nil {
			res.Error = &ErrorInfo{Code: "timeout", Message: fmt.Sprintf("run exceeded %s", timeout)}
		} else {
			res.Error = &ErrorInfo{Code: "runtime_error", Message: "the script finished while awaiting a promise that never settles"}
		}
	}
	return
}

func (r *run) errorInfo(err error) *ErrorInfo {
	var intr *goja.InterruptedError
	if errors.As(err, &intr) || r.ctx.Err() != nil {
		return &ErrorInfo{Code: "timeout", Message: "run exceeded its time limit"}
	}
	var ex *goja.Exception
	if errors.As(err, &ex) {
		return r.rejection(ex.Value())
	}
	return &ErrorInfo{Code: "runtime_error", Message: err.Error()}
}

func (r *run) rejection(v goja.Value) *ErrorInfo {
	if r.ctx.Err() != nil {
		return &ErrorInfo{Code: "timeout", Message: "run exceeded its time limit"}
	}
	if obj, ok := v.(*goja.Object); ok {
		if name := obj.Get("name"); name != nil && name.String() == "AitkError" {
			info := &ErrorInfo{Code: obj.Get("code").String(), Message: obj.Get("message").String()}
			if s := obj.Get("status"); s != nil {
				info.Status = int(s.ToInteger())
			}
			if info.Code == "call_limit" || info.Code == "timeout" {
				return info
			}
			info.Message = "aitk call failed (" + info.Code + "): " + info.Message
			info.Code = "api_error"
			return info
		}
		if stack := obj.Get("stack"); stack != nil && !goja.IsUndefined(stack) {
			return &ErrorInfo{Code: "runtime_error", Message: stack.String()}
		}
	}
	return &ErrorInfo{Code: "runtime_error", Message: v.String()}
}

func (r *run) toJSON(v goja.Value) (json.RawMessage, error) {
	if v == nil || goja.IsUndefined(v) {
		return json.RawMessage("null"), nil
	}
	stringify, _ := goja.AssertFunction(r.vm.Get("JSON").ToObject(r.vm).Get("stringify"))
	out, err := stringify(goja.Undefined(), v)
	if err != nil {
		return nil, err
	}
	if goja.IsUndefined(out) {
		return json.RawMessage("null"), nil
	}
	return json.RawMessage(out.String()), nil
}

func (r *run) throw(code string, status int, msg string) {
	ctor, _ := goja.AssertConstructor(r.errCtor)
	if ctor == nil {
		panic(r.vm.NewGoError(errors.New(msg)))
	}
	obj, err := ctor(nil, r.vm.ToValue(code), r.vm.ToValue(status), r.vm.ToValue(msg))
	if err != nil {
		panic(r.vm.NewGoError(errors.New(msg)))
	}
	panic(obj)
}

func (r *run) install() error {
	vm := r.vm
	vm.SetFieldNameMapper(goja.TagFieldNameMapper("json", true))

	logFn := func(level string) func(goja.FunctionCall) goja.Value {
		return func(c goja.FunctionCall) goja.Value {
			r.log(level, c.Arguments)
			return goja.Undefined()
		}
	}
	console := vm.NewObject()
	console.Set("log", logFn(""))
	console.Set("info", logFn(""))
	console.Set("debug", logFn(""))
	console.Set("warn", logFn("warn: "))
	console.Set("error", logFn("error: "))
	vm.Set("console", console)

	vm.Set("sleep", func(c goja.FunctionCall) goja.Value {
		ms := c.Argument(0).ToInteger()
		if ms > 0 {
			t := time.NewTimer(time.Duration(ms) * time.Millisecond)
			defer t.Stop()
			select {
			case <-t.C:
			case <-r.ctx.Done():
				r.throw("timeout", 0, "run exceeded its time limit")
			}
		}
		p, resolve, _ := vm.NewPromise()
		resolve(goja.Undefined())
		return vm.ToValue(p)
	})

	client := vm.NewObject()
	groups := map[string]*goja.Object{}
	for i := range Ops {
		op := Ops[i]
		g, ok := groups[op.Group]
		if !ok {
			g = vm.NewObject()
			groups[op.Group] = g
			client.Set(op.Group, g)
		}
		g.Set(op.Name, func(c goja.FunctionCall) goja.Value { return r.invoke(op, c.Arguments) })
	}
	vm.Set("aitk", client)

	if _, err := vm.RunString(prelude); err != nil {
		return err
	}
	for _, js := range preludes {
		if _, err := vm.RunString(js); err != nil {
			return err
		}
	}
	r.errCtor = vm.Get("AitkError")
	// nothing else from the host: no require, fetch, process or timers
	for _, name := range []string{"require", "process", "fetch", "setTimeout", "setInterval", "queueMicrotask"} {
		vm.GlobalObject().Delete(name)
	}
	return nil
}

func (r *run) log(prefix string, args []goja.Value) {
	if r.logsFull {
		return
	}
	parts := make([]string, 0, len(args))
	for _, a := range args {
		if s, ok := a.Export().(string); ok {
			parts = append(parts, s)
			continue
		}
		if raw, err := r.toJSON(a); err == nil {
			parts = append(parts, string(raw))
		} else {
			parts = append(parts, a.String())
		}
	}
	line := prefix + strings.Join(parts, " ") + "\n"
	if r.logs.Len()+len(line) > MaxLogBytes {
		room := MaxLogBytes - r.logs.Len()
		for room > 0 && !utf8.ValidString(line[:room]) {
			room--
		}
		if room > 0 {
			r.logs.WriteString(line[:room])
		}
		r.logs.WriteString("\n[log output truncated at 1 MiB]\n")
		r.logsFull = true
		return
	}
	r.logs.WriteString(line)
}

// invoke maps a JS call onto the op: positional path params, then one object
// for query/body params. The first argument may instead be one object
// holding everything, path params included.
func (r *run) invoke(op Op, args []goja.Value) goja.Value {
	if err := r.ctx.Err(); err != nil {
		r.throw("timeout", 0, "run exceeded its time limit")
	}
	if r.calls >= MaxCalls {
		r.throw("call_limit", 0, fmt.Sprintf("more than %d aitk calls in one run", MaxCalls))
	}
	r.calls++

	pathParams := op.pathParams()
	values := map[string]any{}
	restIdx := len(pathParams)
	objectForm := len(args) > 0 && len(pathParams) > 0 && isPlainObject(args[0])
	if objectForm {
		restIdx = 0
	} else {
		for i, pp := range pathParams {
			if i >= len(args) || goja.IsUndefined(args[i]) || goja.IsNull(args[i]) {
				r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: missing argument %s", op.Group, op.Name, pp.Name))
			}
			values[pp.Name] = args[i].String()
		}
	}
	if restIdx < len(args) && !goja.IsUndefined(args[restIdx]) && !goja.IsNull(args[restIdx]) {
		raw, err := r.toJSON(args[restIdx])
		if err != nil {
			r.throw("invalid_request", 400, "params must be JSON-serializable")
		}
		var obj map[string]any
		if err := json.Unmarshal(raw, &obj); err != nil {
			r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: params must be an object", op.Group, op.Name))
		}
		for k, v := range obj {
			values[k] = v
		}
	}

	path := op.Path
	for _, pp := range pathParams {
		v, ok := values[pp.Name]
		if !ok {
			r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: missing %s", op.Group, op.Name, pp.Name))
		}
		s := fmt.Sprint(v)
		var enc string
		if strings.Contains(path, "{"+pp.Name+"...}") {
			segs := strings.Split(strings.ReplaceAll(s, `\`, "/"), "/")
			for i := range segs {
				segs[i] = url.PathEscape(segs[i])
			}
			enc = strings.Join(segs, "/")
			path = strings.Replace(path, "{"+pp.Name+"...}", enc, 1)
		} else {
			path = strings.Replace(path, "{"+pp.Name+"}", url.PathEscape(s), 1)
		}
		delete(values, pp.Name)
	}

	query := url.Values{}
	body := map[string]any{}
	hasBody := op.Method == "POST" || op.Method == "PUT" || op.Method == "PATCH"
	known := map[string]Param{}
	for _, p := range op.Params {
		known[p.Name] = p
	}
	for k, v := range values {
		in := "body"
		if p, ok := known[k]; ok {
			in = p.In
		} else if !hasBody {
			in = "query"
		}
		if in == "query" {
			query.Set(k, queryValue(v))
		} else {
			body[k] = v
		}
	}
	for _, p := range op.Params {
		if !p.Optional && p.In != "path" {
			if _, ok := values[p.Name]; !ok {
				r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: missing %s", op.Group, op.Name, p.Name))
			}
		}
	}

	method := op.Method
	var payload []byte
	contentType := ""
	switch {
	case op.Mode == ModeTextRequest:
		text, ok := body["text"].(string)
		if !ok {
			r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: text must be a string", op.Group, op.Name))
		}
		payload, contentType = []byte(text), "text/plain; charset=utf-8"
	case op.Mode == ModeBase64:
		payload, contentType = r.decodeBase64(op, body["base64"]), "application/octet-stream"
	case op.Mode == ModeMultipart:
		payload, contentType = r.multipartBody(op, body["files"])
	case op.Mode == ModeLink:
		// sign the download instead of buffering it
		ttl := query.Get("ttl_s")
		query.Del("ttl_s")
		target := path
		if len(query) > 0 {
			target += "?" + query.Encode()
		}
		link := map[string]any{"path": target}
		if ttl != "" {
			n, err := strconv.ParseInt(ttl, 10, 64)
			if err != nil {
				r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: ttl_s must be a number", op.Group, op.Name))
			}
			link["ttl_s"] = n
		}
		method, path, query = "POST", "/v1/links", url.Values{}
		payload, _ = json.Marshal(link)
		contentType = "application/json"
	case hasBody:
		payload, _ = json.Marshal(body)
		contentType = "application/json"
	}

	resp, err := r.call(r.ctx, method, path, query, payload, contentType)
	if err != nil {
		if r.ctx.Err() != nil {
			r.throw("timeout", 0, "run exceeded its time limit")
		}
		r.throw("internal", 0, err.Error())
	}
	if resp.Status < 200 || resp.Status > 299 {
		code, msg := "http_"+strconv.Itoa(resp.Status), strings.TrimSpace(string(resp.Body))
		var e struct {
			Error struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(resp.Body, &e) == nil && e.Error.Code != "" {
			code, msg = e.Error.Code, e.Error.Message
		}
		if len(msg) > 2000 {
			msg = msg[:2000] + "..."
		}
		r.throw(code, resp.Status, msg)
	}
	if resp.Status == http.StatusNoContent || len(resp.Body) == 0 {
		return goja.Null()
	}
	size := max(resp.Size, int64(len(resp.Body)))
	if op.Mode == ModeTextResponse {
		if size > MaxTextBytes {
			r.throw("too_large", 413, fmt.Sprintf("file is %d bytes; readText reads at most %d", size, MaxTextBytes))
		}
		return r.vm.ToValue(string(resp.Body))
	}
	if size > int64(len(resp.Body)) {
		r.throw("too_large", 413, fmt.Sprintf("aitk.%s.%s: the response is %d bytes, more than code mode keeps (%d)", op.Group, op.Name, size, MaxResponseBytes))
	}
	parse, _ := goja.AssertFunction(r.vm.Get("JSON").ToObject(r.vm).Get("parse"))
	v, err := parse(goja.Undefined(), r.vm.ToValue(string(resp.Body)))
	if err != nil {
		return r.vm.ToValue(string(resp.Body))
	}
	return v
}

func (r *run) decodeBase64(op Op, v any) []byte {
	s, ok := v.(string)
	if !ok {
		r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: base64 must be a string", op.Group, op.Name))
	}
	if base64.StdEncoding.DecodedLen(len(s)) > MaxBinaryBytes+3 {
		r.throw("too_large", 413, fmt.Sprintf("aitk.%s.%s: at most %d bytes per call; send larger files by URL", op.Group, op.Name, MaxBinaryBytes))
	}
	s = strings.TrimSpace(s)
	if i := strings.Index(s, ";base64,"); i >= 0 && strings.HasPrefix(s, "data:") {
		s = s[i+len(";base64,"):] // a data: URL
	}
	for _, enc := range []*base64.Encoding{base64.StdEncoding, base64.RawStdEncoding, base64.URLEncoding, base64.RawURLEncoding} {
		if b, err := enc.DecodeString(s); err == nil {
			if len(b) > MaxBinaryBytes {
				break
			}
			return b
		}
	}
	r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: base64 is not valid base64 (or decodes to more than %d bytes)", op.Group, op.Name, MaxBinaryBytes))
	return nil
}

// multipartBody builds files[] from [{name, base64}].
func (r *run) multipartBody(op Op, v any) ([]byte, string) {
	list, ok := v.([]any)
	if !ok || len(list) == 0 {
		r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: files must be a non-empty array of { name, base64 }", op.Group, op.Name))
	}
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for _, item := range list {
		f, _ := item.(map[string]any)
		name, _ := f["name"].(string)
		if name == "" {
			r.throw("invalid_request", 400, fmt.Sprintf("aitk.%s.%s: every file needs a name", op.Group, op.Name))
		}
		data := r.decodeBase64(op, f["base64"])
		w, err := mw.CreateFormFile("files", name)
		if err != nil {
			r.throw("internal", 0, err.Error())
		}
		w.Write(data)
		if buf.Len() > 4*MaxBinaryBytes {
			r.throw("too_large", 413, fmt.Sprintf("aitk.%s.%s: at most %d bytes per call", op.Group, op.Name, 4*MaxBinaryBytes))
		}
	}
	mw.Close()
	return buf.Bytes(), mw.FormDataContentType()
}

func isPlainObject(v goja.Value) bool {
	if v == nil || goja.IsUndefined(v) || goja.IsNull(v) {
		return false
	}
	obj, ok := v.(*goja.Object)
	if !ok {
		return false
	}
	switch obj.ClassName() {
	case "Object":
		return true
	}
	return false
}

func queryValue(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case bool:
		return strconv.FormatBool(x)
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	case []any:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = queryValue(e)
		}
		return strings.Join(parts, ",")
	case nil:
		return ""
	}
	b, _ := json.Marshal(v)
	return string(b)
}

package notify

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Sink kinds.
const (
	SinkNtfy     = "ntfy"
	SinkDiscord  = "discord"
	SinkTelegram = "telegram"
	SinkWebhook  = "webhook"
)

// parsed is a sink URL understood: where to send and how.
type parsed struct {
	kind   string
	target string // redacted, for display
	u      *url.URL
	// telegram
	token, chat string
}

// telegramAPI is the Bot API base (tests point it at an httptest server).
var telegramAPI = "https://api.telegram.org"

var discordHosts = map[string]bool{"discord.com": true, "discordapp.com": true, "canary.discord.com": true,
	"ptb.discord.com": true}

// parseSink recognises:
//
//	https://ntfy.sh/<topic>                      ntfy (ntfy+https://host/topic for a self-hosted server;
//	                                             https://:tk_token@host/topic or user:pass@ for auth)
//	https://discord.com/api/webhooks/<id>/<tok>  Discord webhook
//	telegram://<bot token>@<chat id>             Telegram bot (sendMessage / sendPhoto)
//	any other http(s) URL                        JSON webhook (webhook+https:// forces it)
//
// A "<kind>+" scheme prefix (ntfy+, discord+, webhook+) forces the kind.
func parseSink(raw string) (*parsed, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, errors.New("empty sink URL")
	}
	if rest, ok := strings.CutPrefix(raw, "telegram://"); ok {
		i := strings.LastIndex(rest, "@")
		if i <= 0 || i == len(rest)-1 {
			return nil, errors.New("telegram sink must be telegram://<bot token>@<chat id>")
		}
		token, chat := rest[:i], strings.TrimSuffix(rest[i+1:], "/")
		if !strings.Contains(token, ":") {
			return nil, errors.New("telegram bot token looks wrong (expected <digits>:<secret>)")
		}
		return &parsed{kind: SinkTelegram, target: "telegram chat " + chat, token: token, chat: chat}, nil
	}
	forced := ""
	for _, k := range []string{SinkNtfy, SinkDiscord, SinkWebhook} {
		if rest, ok := strings.CutPrefix(raw, k+"+"); ok {
			forced, raw = k, rest
			break
		}
	}
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("not a URL: %v", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" || u.Host == "" {
		return nil, errors.New("sink URL must be http(s)://..., or telegram://<bot token>@<chat id>")
	}
	host := strings.ToLower(u.Hostname())
	kind := forced
	if kind == "" {
		switch {
		case host == "ntfy.sh":
			kind = SinkNtfy
		case discordHosts[host] && strings.HasPrefix(u.Path, "/api/webhooks/"):
			kind = SinkDiscord
		default:
			kind = SinkWebhook
		}
	}
	p := &parsed{kind: kind, u: u}
	switch kind {
	case SinkNtfy:
		topic := strings.Trim(u.Path, "/")
		if topic == "" || strings.Contains(topic, "/") {
			return nil, errors.New("ntfy sink must be https://<server>/<topic>")
		}
		p.target = "ntfy " + u.Host + "/" + short(topic)
	case SinkDiscord:
		parts := strings.Split(strings.Trim(u.Path, "/"), "/")
		id := parts[len(parts)-1]
		if len(parts) >= 2 {
			id = parts[len(parts)-2]
		}
		p.target = "discord webhook …" + tail(id, 4)
	default:
		p.target = "webhook " + u.Scheme + "://" + u.Host + "/…"
	}
	return p, nil
}

func short(s string) string {
	if len(s) <= 3 {
		return s + "…"
	}
	return s[:3] + "…"
}

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}

// redact describes a sink URL without its secret (for logs and the API).
func redact(raw string) string {
	if p, err := parseSink(raw); err == nil {
		return p.target
	}
	if u, err := url.Parse(raw); err == nil && u.Host != "" {
		return u.Scheme + "://" + u.Host + "/…"
	}
	return "(unparseable sink)"
}

// SinkInfo is the kind and redacted target of a sink URL.
func SinkInfo(raw string) (kind, target string) {
	if p, err := parseSink(raw); err == nil {
		return p.kind, p.target
	}
	return "unknown", redact(raw)
}

// httpError is a non-2xx answer; retryable for 429 and 5xx.
type httpError struct {
	status     int
	body       string
	retryAfter time.Duration
}

func (e *httpError) Error() string {
	if e.body == "" {
		return fmt.Sprintf("HTTP %d", e.status)
	}
	return fmt.Sprintf("HTTP %d: %s", e.status, e.body)
}

func retryable(err error) (bool, time.Duration) {
	var he *httpError
	if errors.As(err, &he) {
		return he.status == http.StatusTooManyRequests || he.status >= 500, he.retryAfter
	}
	return !errors.Is(err, context.Canceled), 0 // network trouble
}

var client = &http.Client{Timeout: 30 * time.Second}

func do(req *http.Request) error {
	req.Header.Set("User-Agent", "aitk-server")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
	if resp.StatusCode/100 == 2 {
		return nil
	}
	e := &httpError{status: resp.StatusCode, body: strings.TrimSpace(string(body))}
	if s := resp.Header.Get("Retry-After"); s != "" {
		if n, err := strconv.ParseFloat(s, 64); err == nil {
			e.retryAfter = time.Duration(n * float64(time.Second))
		}
	}
	if e.retryAfter == 0 && resp.StatusCode == http.StatusTooManyRequests {
		// Discord and Telegram put it in the JSON body
		var b struct {
			RetryAfter float64 `json:"retry_after"`
			Parameters struct {
				RetryAfter float64 `json:"retry_after"`
			} `json:"parameters"`
		}
		if json.Unmarshal(body, &b) == nil {
			e.retryAfter = time.Duration(max(b.RetryAfter, b.Parameters.RetryAfter) * float64(time.Second))
		}
	}
	return e
}

func clip(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

func text(m Message) string {
	s := m.Body
	if m.URL != "" {
		s += "\n\n" + m.URL
	}
	return strings.TrimSpace(s)
}

// deliverOnce sends m to one sink, one attempt.
func deliverOnce(ctx context.Context, s Sink, m Message) error {
	p, err := parseSink(s.URL)
	if err != nil {
		return err
	}
	switch p.kind {
	case SinkNtfy:
		return sendNtfy(ctx, p, m)
	case SinkDiscord:
		return sendDiscord(ctx, p, m)
	case SinkTelegram:
		return sendTelegram(ctx, p, m)
	}
	return sendWebhook(ctx, p, m)
}

var ntfyPriority = map[string]string{LevelInfo: "3", LevelOK: "3", LevelWarn: "4", LevelError: "5"}
var ntfyTags = map[string]string{LevelInfo: "information_source", LevelOK: "white_check_mark", LevelWarn: "warning",
	LevelError: "rotating_light"}

// sendNtfy publishes to the topic URL. Title, click and the rest go as query
// parameters (UTF-8 and newlines are safe there, unlike in headers); with an
// image the body is the attachment (PUT, filename) and the text the message
// parameter, else the body is the text.
func sendNtfy(ctx context.Context, p *parsed, m Message) error {
	u := *p.u
	q := u.Query()
	q.Set("title", clip(m.Title, 250))
	if m.URL != "" {
		q.Set("click", m.URL)
	}
	q.Set("priority", ntfyPriority[m.Level])
	if t := ntfyTags[m.Level]; t != "" {
		q.Set("tags", t)
	}
	method, body := http.MethodPost, []byte(clip(text(m), 3900))
	if len(m.Image) > 0 {
		q.Set("message", clip(text(m), 1500))
		q.Set("filename", "sample.jpg")
		method, body = http.MethodPut, m.Image
	}
	u.RawQuery = q.Encode()
	req, err := http.NewRequestWithContext(ctx, method, u.String(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	return do(req)
}

var discordColor = map[string]int{LevelInfo: 0x5b5bd6, LevelOK: 0x15803d, LevelWarn: 0xb45309, LevelError: 0xdc2626}

// sendDiscord posts an embed; with an image, as multipart with the JPEG
// attached and shown in the embed.
func sendDiscord(ctx context.Context, p *parsed, m Message) error {
	embed := map[string]any{"title": clip(m.Title, 256), "description": clip(m.Body, 4000),
		"color": discordColor[m.Level], "timestamp": m.At.Format(time.RFC3339)}
	if strings.HasPrefix(m.URL, "http://") || strings.HasPrefix(m.URL, "https://") {
		embed["url"] = m.URL
	}
	if len(m.Image) > 0 {
		embed["image"] = map[string]string{"url": "attachment://sample.jpg"}
	}
	payload, _ := json.Marshal(map[string]any{"embeds": []any{embed}, "allowed_mentions": map[string]any{"parse": []string{}}})
	var req *http.Request
	var err error
	if len(m.Image) == 0 {
		req, err = http.NewRequestWithContext(ctx, http.MethodPost, p.u.String(), bytes.NewReader(payload))
		if err == nil {
			req.Header.Set("Content-Type", "application/json")
		}
	} else {
		var buf bytes.Buffer
		mw := multipart.NewWriter(&buf)
		mw.WriteField("payload_json", string(payload))
		fw, _ := mw.CreateFormFile("files[0]", "sample.jpg")
		fw.Write(m.Image)
		mw.Close()
		req, err = http.NewRequestWithContext(ctx, http.MethodPost, p.u.String(), &buf)
		if err == nil {
			req.Header.Set("Content-Type", mw.FormDataContentType())
		}
	}
	if err != nil {
		return err
	}
	return do(req)
}

// sendTelegram uses sendMessage, or sendPhoto with a caption when there is
// an image. Plain text (no parse_mode), so nothing needs escaping.
func sendTelegram(ctx context.Context, p *parsed, m Message) error {
	base := strings.TrimRight(telegramAPI, "/") + "/bot" + p.token
	msg := strings.TrimSpace(m.Title + "\n" + text(m))
	var req *http.Request
	var err error
	if len(m.Image) == 0 {
		body, _ := json.Marshal(map[string]any{"chat_id": p.chat, "text": clip(msg, 4000), "disable_web_page_preview": true})
		req, err = http.NewRequestWithContext(ctx, http.MethodPost, base+"/sendMessage", bytes.NewReader(body))
		if err == nil {
			req.Header.Set("Content-Type", "application/json")
		}
	} else {
		var buf bytes.Buffer
		mw := multipart.NewWriter(&buf)
		mw.WriteField("chat_id", p.chat)
		mw.WriteField("caption", clip(msg, 1000))
		fw, _ := mw.CreateFormFile("photo", "sample.jpg")
		fw.Write(m.Image)
		mw.Close()
		req, err = http.NewRequestWithContext(ctx, http.MethodPost, base+"/sendPhoto", &buf)
		if err == nil {
			req.Header.Set("Content-Type", mw.FormDataContentType())
		}
	}
	if err != nil {
		return err
	}
	err = do(req)
	var he *httpError
	if errors.As(err, &he) {
		// the URL holds the bot token: keep it out of error text
		he.body = strings.ReplaceAll(he.body, p.token, "<token>")
	}
	return err
}

// maxWebhookImage caps the base64 image a JSON webhook carries.
const maxWebhookImage = 1 << 20

// sendWebhook posts the message as JSON: {kind, level, title, body, url,
// job_id, job_name, at, server, image_jpeg_base64?}.
func sendWebhook(ctx context.Context, p *parsed, m Message) error {
	payload := map[string]any{"kind": m.Kind, "level": m.Level, "title": m.Title, "body": m.Body, "url": m.URL,
		"job_id": m.JobID, "job_name": m.JobName, "at": m.At.Format(time.RFC3339Nano), "server": PublicURL(Current())}
	if len(m.Image) > 0 && len(m.Image) <= maxWebhookImage {
		payload["image_jpeg_base64"] = base64.StdEncoding.EncodeToString(m.Image)
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.u.String(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	return do(req)
}

// Delivery: one queue and one worker per sink, so a slow or failing sink
// never holds the others back.

const (
	queueSize   = 64
	maxAttempts = 5
	maxBackoff  = 2 * time.Minute
)

// backoffBase is the first retry delay (doubles per attempt); tests shrink it.
var backoffBase = 2 * time.Second

type delivery struct {
	sink Sink
	msg  Message
}

var (
	queuesMu sync.Mutex
	queues   = map[string]chan delivery{}
	inflight atomic.Int64
	runCtx   = context.Background()
)

func enqueue(s Sink, m Message) {
	queuesMu.Lock()
	q, ok := queues[s.ID]
	if !ok {
		q = make(chan delivery, queueSize)
		queues[s.ID] = q
		go worker(q)
	}
	queuesMu.Unlock()
	inflight.Add(1)
	select {
	case q <- delivery{s, m}:
	default:
		inflight.Add(-1)
		log.Printf("notify: queue for %s full, dropped %q", redact(s.URL), m.Title)
	}
}

func worker(q chan delivery) {
	for d := range q {
		deliver(d)
		inflight.Add(-1)
	}
}

func deliver(d delivery) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("notify: panic sending to %s: %v", redact(d.sink.URL), r)
		}
	}()
	ctx := runCtx
	wait := backoffBase
	for attempt := 1; ; attempt++ {
		err := deliverOnce(ctx, d.sink, d.msg)
		if err == nil {
			return
		}
		again, after := retryable(err)
		if !again || attempt >= maxAttempts || ctx.Err() != nil {
			log.Printf("notify: %s: %q not delivered after %d attempt(s): %v", redact(d.sink.URL), d.msg.Title, attempt, err)
			return
		}
		if after > wait {
			wait = after
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(min(wait, maxBackoff)):
		}
		wait *= 2
	}
}

// Flush waits until every queued message was delivered or given up on, or
// ctx ends (lifecycle calls it before the server goes away).
func Flush(ctx context.Context) {
	for inflight.Load() > 0 {
		select {
		case <-ctx.Done():
			return
		case <-time.After(100 * time.Millisecond):
		}
	}
}

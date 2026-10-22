package codemode

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
)

// HandlerCaller calls h in-process, as the client with this Authorization
// header ("" for none), so auth and every handler behave exactly as over HTTP.
// origin (scheme://host the client reached the server at, "" for unknown) is
// what handlers building absolute URLs (signed links) see as the host.
func HandlerCaller(h http.Handler, authorization, origin string) Caller {
	var host, scheme string
	if u, err := url.Parse(origin); err == nil && u.Host != "" {
		host, scheme = u.Host, u.Scheme
	}
	return func(ctx context.Context, method, path string, query url.Values, body []byte, contentType string) (Response, error) {
		target := path
		if len(query) > 0 {
			target += "?" + query.Encode()
		}
		req := httptest.NewRequest(method, target, bytes.NewReader(body)).WithContext(ctx)
		req.RequestURI = ""
		if contentType != "" {
			req.Header.Set("Content-Type", contentType)
		}
		if authorization != "" {
			req.Header.Set("Authorization", authorization)
		}
		if host != "" {
			req.Host = host
			req.Header.Set("X-Forwarded-Host", host)
			req.Header.Set("X-Forwarded-Proto", scheme)
		}
		rec := &cappedRecorder{header: http.Header{}, limit: MaxResponseBytes}
		h.ServeHTTP(rec, req)
		if rec.code == 0 {
			rec.code = http.StatusOK
		}
		return Response{Status: rec.code, Header: rec.header, Body: rec.buf.Bytes(), Size: rec.size}, nil
	}
}

// MaxResponseBytes is how much of one in-process response is kept: a file
// read through the API (a checkpoint, a model) must not be buffered whole in
// the server, which shares its memory with the training run.
const MaxResponseBytes = 16 << 20

// cappedRecorder is an http.ResponseWriter that keeps the first limit bytes
// and counts the rest.
type cappedRecorder struct {
	header http.Header
	code   int
	buf    bytes.Buffer
	size   int64
	limit  int64
}

func (c *cappedRecorder) Header() http.Header { return c.header }

func (c *cappedRecorder) WriteHeader(code int) {
	if c.code == 0 {
		c.code = code
	}
}

func (c *cappedRecorder) Write(p []byte) (int, error) {
	c.WriteHeader(http.StatusOK)
	c.size += int64(len(p))
	if room := c.limit - int64(c.buf.Len()); room > 0 {
		c.buf.Write(p[:min(int64(len(p)), room)])
	}
	return len(p), nil
}

func (c *cappedRecorder) Flush() {}

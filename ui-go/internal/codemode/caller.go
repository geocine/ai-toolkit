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
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return Response{Status: rec.Code, Header: rec.Header(), Body: rec.Body.Bytes()}, nil
	}
}

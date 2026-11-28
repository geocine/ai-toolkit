package relay

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/geocine/aitoolkit/internal/provision"
	"github.com/geocine/aitoolkit/internal/tunnel"
)

// The relay reaches a worker through its tunnel: plain requests with the
// machine token, and a WebSocket upgrade (the shell rides on this) both
// ways through the stream.
func TestRelayThroughTunnel(t *testing.T) {
	const id = "mch_Tunnel01"
	old := tunnel.Default
	tunnel.Default = &tunnel.Hub{Lookup: func(_ context.Context, m string) (string, bool) { return machineToken, m == id }}
	t.Cleanup(func() { tunnel.Default = old })

	hubMux := http.NewServeMux()
	hubMux.Handle("GET "+tunnel.Path, tunnel.Default)
	studio := httptest.NewServer(hubMux)
	defer studio.Close()

	worker := http.NewServeMux()
	worker.HandleFunc("/v1/whoami", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, r.Header.Get("Authorization")+"|"+r.Header.Get("X-Forwarded-Prefix"))
	})
	worker.HandleFunc("/v1/echo", func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer c.CloseNow()
		c.SetReadLimit(1 << 20)
		for {
			typ, b, err := c.Read(r.Context())
			if err != nil {
				return
			}
			if c.Write(r.Context(), typ, append([]byte("echo:"), b...)) != nil {
				return
			}
		}
	})
	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	go (&tunnel.Client{URL: studio.URL, MachineID: id, Token: machineToken, Handler: worker}).Run(ctx)
	deadline := time.Now().Add(5 * time.Second)
	for !tunnel.Default.Connected(id) {
		if time.Now().After(deadline) {
			t.Fatal("the worker never connected")
		}
		time.Sleep(10 * time.Millisecond)
	}

	target := provision.RelayTarget{MachineID: id, BaseURL: tunnel.URL(id), Token: machineToken}
	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		inner := strings.TrimPrefix(r.URL.Path, Prefix(id)+"/")
		Serve(w, r, target, inner, true)
	}))
	defer front.Close()

	res, err := http.Get(front.URL + Prefix(id) + "/v1/whoami")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if got := string(body); got != "Bearer "+machineToken+"|"+Prefix(id) {
		t.Fatalf("worker saw %q", got)
	}

	wctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c, _, err := websocket.Dial(wctx, "ws"+strings.TrimPrefix(front.URL, "http")+Prefix(id)+"/v1/echo", nil)
	if err != nil {
		t.Fatalf("websocket through the relay and tunnel: %v", err)
	}
	defer c.CloseNow()
	for _, msg := range []string{"ls -la\r", strings.Repeat("x", 200<<10)} {
		if err := c.Write(wctx, websocket.MessageBinary, []byte(msg)); err != nil {
			t.Fatal(err)
		}
		c.SetReadLimit(1 << 20)
		_, b, err := c.Read(wctx)
		if err != nil || string(b) != "echo:"+msg {
			t.Fatalf("echo of %d bytes: %d bytes, %v", len(msg), len(b), err)
		}
	}

	// another machine's tunnel URL is refused, never dialled
	other := provision.RelayTarget{MachineID: "mch_other", BaseURL: tunnel.URL(id), Token: machineToken}
	rec := httptest.NewRecorder()
	Serve(rec, httptest.NewRequest("GET", Prefix("mch_other")+"/v1/whoami", nil), other, "v1/whoami", true)
	if rec.Code == http.StatusOK {
		t.Fatalf("relayed to another machine's tunnel: %d %s", rec.Code, rec.Body)
	}
}

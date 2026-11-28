package tunnel

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"log"
	"net"
	"net/http"
	"sync"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/hashicorp/yamux"

	"github.com/geocine/aitoolkit/internal/netguard"
)

// Hub is the studio API's end: it accepts workers' tunnels and opens streams
// to them.
type Hub struct {
	// Lookup is the token of a machine this studio API launched and still
	// tracks; ok=false refuses the tunnel. Nil refuses every tunnel.
	Lookup func(ctx context.Context, machineID string) (token string, ok bool)
	// OnChange runs (in its own goroutine) when a machine's tunnel comes up
	// or goes down, so its status is refreshed at once.
	OnChange func(machineID string, up bool)

	mu   sync.Mutex
	sess map[string]*yamux.Session
}

// Default is the hub the server registers and netguard dials through.
var Default = &Hub{}

func init() {
	netguard.SetTunnelDialer(func(ctx context.Context, host string) (net.Conn, error) {
		id, ok := MachineOf(host)
		if !ok {
			return nil, ErrNotConnected
		}
		return Default.Dial(ctx, id)
	})
}

// Connected reports a live tunnel for machineID.
func (h *Hub) Connected(machineID string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	s := h.sess[machineID]
	return s != nil && !s.IsClosed()
}

// Dial opens a stream to the worker: one HTTP connection.
func (h *Hub) Dial(ctx context.Context, machineID string) (net.Conn, error) {
	h.mu.Lock()
	s := h.sess[machineID]
	h.mu.Unlock()
	if s == nil || s.IsClosed() {
		return nil, ErrNotConnected
	}
	type result struct {
		c   net.Conn
		err error
	}
	ch := make(chan result, 1)
	go func() {
		c, err := s.OpenStream()
		ch <- result{c, err}
	}()
	select {
	case r := <-ch:
		if r.err != nil {
			return nil, r.err
		}
		return r.c, nil
	case <-ctx.Done():
		go func() {
			if r := <-ch; r.c != nil {
				r.c.Close()
			}
		}()
		return nil, ctx.Err()
	}
}

// Drop closes a machine's tunnel (it was terminated or forgotten).
func (h *Hub) Drop(machineID string) {
	h.mu.Lock()
	s := h.sess[machineID]
	delete(h.sess, machineID)
	h.mu.Unlock()
	if s != nil {
		s.Close()
	}
}

// ServeHTTP accepts one worker's tunnel. It is not behind the bearer auth:
// the handshake is the worker's credential.
func (h *Hub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	id := r.Header.Get(MachineHeader)
	if h.Lookup == nil || id == "" || r.Header.Get("Origin") != "" {
		// a browser page always sends Origin; a worker never does
		http.Error(w, "not a tunnel", http.StatusBadRequest)
		return
	}
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{CompressionMode: websocket.CompressionDisabled})
	if err != nil {
		return
	}
	hctx, cancel := context.WithTimeout(r.Context(), handshakeTimeout)
	token, ok := h.Lookup(hctx, id)
	if !ok {
		// the same exchange as a wrong token, so an unknown id is not told apart
		token = randomToken()
	}
	if !h.handshake(hctx, c, id, token) || !ok {
		cancel()
		c.Close(websocket.StatusPolicyViolation, "tunnel refused")
		log.Printf("tunnel %s: refused from %s", id, r.RemoteAddr)
		return
	}
	cancel()

	c.SetReadLimit(64 << 20)
	// the connection lives as long as the session, not the request: a
	// hijacked request's context is not cancelled by the server
	sctx, stop := context.WithCancel(context.WithoutCancel(r.Context()))
	defer stop()
	sess, err := yamux.Client(websocket.NetConn(sctx, c, websocket.MessageBinary), yamuxConfig())
	if err != nil {
		c.Close(websocket.StatusInternalError, "tunnel failed")
		return
	}
	h.mu.Lock()
	if h.sess == nil {
		h.sess = map[string]*yamux.Session{}
	}
	old := h.sess[id]
	h.sess[id] = sess
	h.mu.Unlock()
	if old != nil {
		old.Close() // a reconnect replaces the previous tunnel
	}
	log.Printf("tunnel %s: connected from %s", id, r.RemoteAddr)
	h.changed(id, true)

	<-sess.CloseChan()
	h.mu.Lock()
	current := h.sess[id] == sess
	if current {
		delete(h.sess, id)
	}
	h.mu.Unlock()
	c.CloseNow()
	if current {
		log.Printf("tunnel %s: disconnected", id)
		h.changed(id, false)
	}
}

func (h *Hub) changed(id string, up bool) {
	if f := h.OnChange; f != nil {
		go f(id, up)
	}
}

// handshake: challenge the worker, check its MAC, then prove the token back.
func (h *Hub) handshake(ctx context.Context, c *websocket.Conn, id, token string) bool {
	challenge := randomToken()
	if wsjson.Write(ctx, c, hello{V: 1, Challenge: challenge}) != nil {
		return false
	}
	var a answer
	if wsjson.Read(ctx, c, &a) != nil || a.Nonce == "" || len(a.Nonce) > 128 {
		return false
	}
	if !macEqual(a.MAC, mac(token, "worker", id, challenge, a.Nonce)) {
		return false
	}
	return wsjson.Write(ctx, c, done{OK: true, MAC: mac(token, "studio", id, a.Nonce, challenge)}) == nil
}

func randomToken() string {
	b := make([]byte, 32)
	_, _ = rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}

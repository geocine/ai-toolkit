package tunnel

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math/rand/v2"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/hashicorp/yamux"
)

// Client is the worker's end: it keeps a tunnel to the studio API up and
// serves every stream the studio API opens with Handler.
type Client struct {
	URL       string // the studio API's base URL (AITK_STUDIO_API_URL)
	MachineID string // AITK_MACHINE_ID
	Token     string // AI_TOOLKIT_AUTH: the machine token
	Handler   http.Handler
}

// CheckURL accepts an https studio API URL, or plain http only to this
// machine, the Docker host or a private address (a test on one machine or a
// LAN): anywhere else plain http would expose every byte on the way.
func CheckURL(raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimRight(strings.TrimSpace(raw), "/"))
	if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("AITK_STUDIO_API_URL must be the studio API's base URL, like https://studio.example")
	}
	if u.Scheme == "http" && !localHost(u.Hostname()) {
		return nil, fmt.Errorf("AITK_STUDIO_API_URL is plain http to %s: use https (plain http is only accepted to localhost, host.docker.internal or a private address)", u.Hostname())
	}
	return u, nil
}

func localHost(h string) bool {
	switch strings.ToLower(h) {
	case "localhost", "host.docker.internal":
		return true
	}
	a, err := netip.ParseAddr(h)
	return err == nil && (a.IsLoopback() || a.IsPrivate())
}

// Run keeps the tunnel up until ctx ends, reconnecting with jittered backoff
// (1 s doubling to 30 s; reset once a tunnel stayed up a minute). It returns
// only a configuration error, or nil when ctx ends.
func (c *Client) Run(ctx context.Context) error {
	base, err := CheckURL(c.URL)
	if err != nil {
		return err
	}
	if c.MachineID == "" || c.Token == "" {
		return errors.New("a worker's tunnel needs AITK_MACHINE_ID and AI_TOOLKIT_AUTH")
	}
	ws := *base
	ws.Scheme = map[string]string{"http": "ws", "https": "wss"}[base.Scheme]
	ws.Path = strings.TrimRight(base.Path, "/") + Path
	backoff := time.Second
	for ctx.Err() == nil {
		start := time.Now()
		err := c.once(ctx, ws.String())
		if ctx.Err() != nil {
			return nil
		}
		if time.Since(start) > time.Minute {
			backoff = time.Second
		}
		wait := backoff/2 + rand.N(backoff/2+1)
		log.Printf("tunnel: %v; reconnecting in %s", err, wait.Round(100*time.Millisecond))
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(wait):
		}
		backoff = min(backoff*2, 30*time.Second)
	}
	return nil
}

// once runs one tunnel until it drops.
func (c *Client) once(ctx context.Context, wsURL string) error {
	hctx, cancel := context.WithTimeout(ctx, handshakeTimeout)
	defer cancel()
	conn, res, err := websocket.Dial(hctx, wsURL, &websocket.DialOptions{
		HTTPHeader:      http.Header{MachineHeader: {c.MachineID}},
		CompressionMode: websocket.CompressionDisabled,
	})
	if err != nil {
		if res != nil {
			return fmt.Errorf("the studio API answered %s", res.Status)
		}
		return fmt.Errorf("couldn't reach the studio API: %w", err)
	}
	if err := c.handshake(hctx, conn); err != nil {
		conn.Close(websocket.StatusPolicyViolation, "handshake failed")
		return err
	}
	conn.SetReadLimit(64 << 20)
	sctx, stop := context.WithCancel(ctx)
	defer stop()
	sess, err := yamux.Server(websocket.NetConn(sctx, conn, websocket.MessageBinary), yamuxConfig())
	if err != nil {
		conn.CloseNow()
		return err
	}
	defer sess.Close()
	log.Printf("tunnel: connected to %s", wsURL)
	srv := &http.Server{Handler: fromTunnel(c.Handler), ReadHeaderTimeout: 30 * time.Second}
	go func() {
		select {
		case <-ctx.Done():
		case <-sess.CloseChan():
		}
		srv.Close()
	}()
	err = srv.Serve(sess)
	if ctx.Err() != nil {
		return nil
	}
	if err == nil || errors.Is(err, http.ErrServerClosed) || errors.Is(err, net.ErrClosed) {
		return errors.New("the tunnel closed")
	}
	return fmt.Errorf("the tunnel dropped: %w", err)
}

// handshake answers the studio API's challenge, then checks it knows the token.
func (c *Client) handshake(ctx context.Context, conn *websocket.Conn) error {
	var h hello
	if err := wsjson.Read(ctx, conn, &h); err != nil {
		return fmt.Errorf("tunnel handshake: %w", err)
	}
	if h.V != 1 || h.Challenge == "" || len(h.Challenge) > 128 {
		return errors.New("tunnel handshake: the studio API speaks another tunnel version")
	}
	nonce := randomToken()
	if err := wsjson.Write(ctx, conn, answer{MAC: mac(c.Token, "worker", c.MachineID, h.Challenge, nonce), Nonce: nonce}); err != nil {
		return fmt.Errorf("tunnel handshake: %w", err)
	}
	var d done
	if err := wsjson.Read(ctx, conn, &d); err != nil {
		return errors.New("the studio API refused this machine's tunnel (it doesn't track this machine, or the token differs)")
	}
	if !d.OK || !macEqual(d.MAC, mac(c.Token, "studio", c.MachineID, nonce, h.Challenge)) {
		return errors.New("tunnel handshake: the studio API couldn't prove it holds this machine's token")
	}
	return nil
}

// fromTunnel marks requests that came through the tunnel: never a loopback
// peer, so an open-server rule meant for this machine's own processes cannot
// apply to them.
func fromTunnel(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.RemoteAddr = "tunnel:0"
		next.ServeHTTP(w, r)
	})
}

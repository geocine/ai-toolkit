// Package tunnel connects a GPU server (a worker) to the studio API that
// launched it over one outbound WebSocket, so the studio API never dials the
// worker and the worker needs no public port (docs/gateway-design.md §1).
//
// The worker dials GET <studio API>/v1/fleet/tunnel with its machine id. Both
// sides prove they hold the machine token without sending it: each sends a
// fresh random challenge, and each answers the other's with an HMAC-SHA256
// keyed by the token. The connection then carries yamux, and every stream the
// studio API opens is one HTTP/1.1 connection the worker serves with its
// normal handler (auth included: the studio API still sends the token as a
// bearer inside the stream).
//
// On the studio API a worker is addressed as http://<hex machine id>.tunnel.aitk:
// netguard routes that host to Hub.Dial, so the probe, relay and every other
// netguard client with Options.Tunnel reach the worker through its tunnel.
package tunnel

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"io"
	"strings"
	"time"

	"github.com/hashicorp/yamux"

	"github.com/geocine/aitoolkit/internal/netguard"
)

// Path is the studio API route a worker dials.
const Path = "/v1/fleet/tunnel"

// MachineHeader carries the dialling worker's machine id.
const MachineHeader = "X-Aitk-Machine"

// Protocol version, part of every MAC so a later change cannot be confused
// with this one.
const proto = "aitk-tunnel-v1"

// handshakeTimeout bounds the whole handshake on both sides.
const handshakeTimeout = 15 * time.Second

// URL is the base URL a worker's tunnel answers on, on the studio API.
func URL(machineID string) string {
	return "http://" + hex.EncodeToString([]byte(machineID)) + netguard.TunnelSuffix
}

// MachineOf is the machine id a tunnel host (no port) names.
func MachineOf(host string) (string, bool) {
	host = strings.ToLower(strings.TrimSuffix(host, "."))
	label, ok := strings.CutSuffix(host, netguard.TunnelSuffix)
	if !ok || label == "" || strings.Contains(label, ".") {
		return "", false
	}
	b, err := hex.DecodeString(label)
	if err != nil || len(b) == 0 {
		return "", false
	}
	return string(b), true
}

// IsURL reports base is the tunnel URL of machineID.
func IsURL(base, machineID string) bool {
	return machineID != "" && strings.TrimRight(base, "/") == URL(machineID)
}

// hello is the studio API's first message; answer the worker's reply; done
// the studio API's last.
type hello struct {
	V         int    `json:"v"`
	Challenge string `json:"challenge"`
}

type answer struct {
	MAC   string `json:"mac"`
	Nonce string `json:"nonce"`
}

type done struct {
	OK  bool   `json:"ok"`
	MAC string `json:"mac"`
}

// mac is HMAC-SHA256(token, proto|role|machine|a|b), base64url.
func mac(token, role, machineID, a, b string) string {
	h := hmac.New(sha256.New, []byte(token))
	io.WriteString(h, proto+"|"+role+"|"+machineID+"|"+a+"|"+b)
	return base64.RawURLEncoding.EncodeToString(h.Sum(nil))
}

func macEqual(a, b string) bool { return hmac.Equal([]byte(a), []byte(b)) }

// yamuxConfig is shared by both ends. Keepalives every 20 s hold provider NAT
// and proxy idle timers open; larger stream windows keep bulk transfers fast
// across a long round trip.
func yamuxConfig() *yamux.Config {
	c := yamux.DefaultConfig()
	c.KeepAliveInterval = 20 * time.Second
	c.ConnectionWriteTimeout = 30 * time.Second
	c.MaxStreamWindowSize = 4 << 20
	c.LogOutput = io.Discard
	return c
}

// ErrNotConnected: the machine has no tunnel up.
var ErrNotConnected = errors.New("tunnel: the machine is not connected")

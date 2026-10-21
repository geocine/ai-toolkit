//go:build !linux

package supervisor

// Reading another process's command line needs elevated APIs off Linux, so
// liveness falls back to the pid alone there.
func commandLine(int) (string, bool) { return "", false }

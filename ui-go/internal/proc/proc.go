// Package proc inspects and signals job processes that are not (or no longer)
// children of this server.
package proc

import (
	"context"
	"os/exec"
	"strconv"
)

// TreeKill force-kills pid and its descendants (taskkill /T /F on Windows,
// SIGKILL on the pid elsewhere, like upstream's kill route).
func TreeKill(ctx context.Context, pid int) error {
	return treeKill(ctx, pid)
}

func taskkill(ctx context.Context, pid int) error {
	cmd := exec.CommandContext(ctx, "taskkill", "/PID", strconv.Itoa(pid), "/T", "/F")
	hideWindow(cmd)
	return cmd.Run()
}

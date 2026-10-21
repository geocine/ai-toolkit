//go:build linux

package sysstats

import (
	"os/exec"
	"syscall"
)

func hideWindow(*exec.Cmd) {}

// dieWithParent makes the resident nvidia-smi die with the server.
func dieWithParent(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
}

func adoptChild(*exec.Cmd) {}

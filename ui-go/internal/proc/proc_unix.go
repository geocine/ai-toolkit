//go:build !windows

package proc

import (
	"context"
	"errors"
	"os/exec"
	"syscall"
)

// Alive reports whether pid exists; EPERM (someone else's process) counts.
func Alive(pid int) bool {
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}

// Interrupt sends SIGINT, which the trainer handles as a graceful stop.
func Interrupt(pid int) error { return syscall.Kill(pid, syscall.SIGINT) }

func treeKill(_ context.Context, pid int) error { return syscall.Kill(pid, syscall.SIGKILL) }

func hideWindow(*exec.Cmd) {}

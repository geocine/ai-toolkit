//go:build windows

package proc

import (
	"context"
	"errors"
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

const stillActive = 259

// Alive reports whether pid exists; access denied (someone else's process) counts.
func Alive(pid int) bool {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return errors.Is(err, windows.ERROR_ACCESS_DENIED)
	}
	defer windows.CloseHandle(h)
	var code uint32
	if err := windows.GetExitCodeProcess(h, &code); err != nil {
		return true
	}
	return code == stillActive
}

// Interrupt is a no-op on Windows: a job under pythonw has no console to
// deliver a signal to. The `stop` flag in the database is the signal; the
// trainer's stop watcher raises SIGINT in-process.
func Interrupt(int) error { return nil }

func treeKill(ctx context.Context, pid int) error { return taskkill(ctx, pid) }

func hideWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
}

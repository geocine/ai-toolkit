//go:build windows

package remote

import (
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

// hfHideWindow keeps the helper's console from flashing on Windows.
func hfHideWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
}

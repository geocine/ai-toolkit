//go:build windows

package system

import (
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

func hideWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
}

// signalOf: Windows processes do not die from signals.
func signalOf(*exec.ExitError) string { return "" }

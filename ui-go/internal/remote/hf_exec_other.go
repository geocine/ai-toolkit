//go:build !windows

package remote

import "os/exec"

func hfHideWindow(*exec.Cmd) {}

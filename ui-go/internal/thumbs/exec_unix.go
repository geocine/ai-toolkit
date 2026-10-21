//go:build !windows

package thumbs

import "os/exec"

func hideWindow(*exec.Cmd) {}

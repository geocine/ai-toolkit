//go:build darwin

package sysstats

import "os/exec"

func dieWithParent(*exec.Cmd) {}

func adoptChild(*exec.Cmd) {}

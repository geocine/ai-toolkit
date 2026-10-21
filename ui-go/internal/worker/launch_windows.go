//go:build windows

package worker

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sys/windows"
)

type launchSpec struct {
	python, dir, logPath, trainingFolder, jobID string
	args, env                                   []string
}

const (
	relayErrorPrefix = "error:"
	relayPidTimeout  = 30 * time.Second
)

// launch starts the job through a relay: this executable re-run as
// `<exe> __aitk_launch_relay <pidfile> <python> <args...>`. The relay starts
// the job detached and exits, so the job is not a descendant of the server
// and survives `taskkill /T` (or any tree kill) of the UI. The relay's
// stdout/stderr are the job log, which the job inherits.
func launch(s launchSpec) (int, error) {
	exe, err := os.Executable()
	if err != nil {
		return 0, err
	}
	pidPath := filepath.Join(s.trainingFolder, ".job_pid")
	_ = os.Remove(pidPath)

	logFile, err := os.OpenFile(s.logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, err
	}
	defer logFile.Close()

	relay := exec.Command(exe, append([]string{RelayArg, pidPath, s.python}, s.args...)...)
	relay.Dir = s.dir
	relay.Env = s.env
	relay.Stdout = logFile
	relay.Stderr = logFile
	relay.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
	if err := relay.Start(); err != nil {
		return 0, err
	}

	done := make(chan struct{})
	go func() { relay.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(relayPidTimeout):
		return 0, fmt.Errorf("Timed out waiting for the job process to start")
	}

	raw, err := os.ReadFile(pidPath)
	if err != nil {
		return 0, fmt.Errorf("Job process did not report a pid")
	}
	contents := strings.TrimSpace(string(raw))
	if strings.HasPrefix(contents, relayErrorPrefix) {
		return 0, fmt.Errorf("%s", strings.TrimPrefix(contents, relayErrorPrefix))
	}
	pid, err := strconv.Atoi(contents)
	if err != nil || pid <= 0 {
		return 0, fmt.Errorf("Job process did not report a usable pid")
	}
	watchDetached(pid, s.jobID, s.logPath)
	return pid, nil
}

// RunRelay is the relay process body: start the job detached, report its pid
// (or the launch error) through the pid file, and exit.
func RunRelay(args []string) int {
	if len(args) < 2 {
		fmt.Fprintln(os.Stderr, "relay: missing arguments")
		return 2
	}
	pidPath, command, rest := args[0], args[1], args[2:]
	cmd := exec.Command(command, rest...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: windows.DETACHED_PROCESS | windows.CREATE_NEW_PROCESS_GROUP,
	}
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "Error launching job process: "+err.Error())
		if werr := os.WriteFile(pidPath, []byte(relayErrorPrefix+err.Error()), 0o644); werr != nil {
			fmt.Fprintln(os.Stderr, "Could not write job pid file: "+werr.Error())
		}
		return 1
	}
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(cmd.Process.Pid)), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "Could not write job pid file: "+err.Error())
	}
	cmd.Process.Release()
	return 0
}

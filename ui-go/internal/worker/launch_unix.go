//go:build !windows

package worker

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"syscall"
)

type launchSpec struct {
	python, dir, logPath, trainingFolder, jobID string
	args, env                                   []string
}

// launch starts the job in its own session so it survives the server, with
// stdout/stderr going to the job log. The job is still our child, so its exit
// is watched directly and an abnormal one is recorded.
func launch(s launchSpec) (int, error) {
	logFile, err := os.OpenFile(s.logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, err
	}
	defer logFile.Close()

	cmd := exec.Command(s.python, s.args...)
	cmd.Dir = s.dir
	cmd.Env = s.env
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	pid := cmd.Process.Pid
	setWatched(pid, true)
	go func() {
		defer setWatched(pid, false)
		err := cmd.Wait()
		if err == nil {
			return
		}
		var result string
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			if ws, ok := exitErr.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
				result = "signal " + signalName(ws.Signal())
			} else {
				result = fmt.Sprintf("exit code %d", exitErr.ExitCode())
			}
		} else {
			result = err.Error()
		}
		failRunning(s.jobID, s.logPath, "Job process terminated with "+result+".")
	}()
	return pid, nil
}

// signalName matches Node's signal names (SIGKILL, SIGTERM, ...).
func signalName(sig syscall.Signal) string {
	names := map[syscall.Signal]string{
		syscall.SIGHUP: "SIGHUP", syscall.SIGINT: "SIGINT", syscall.SIGQUIT: "SIGQUIT",
		syscall.SIGABRT: "SIGABRT", syscall.SIGKILL: "SIGKILL", syscall.SIGSEGV: "SIGSEGV",
		syscall.SIGPIPE: "SIGPIPE", syscall.SIGTERM: "SIGTERM", syscall.SIGBUS: "SIGBUS",
	}
	if n, ok := names[sig]; ok {
		return n
	}
	return sig.String()
}

// RunRelay is only used on Windows.
func RunRelay([]string) int { return 1 }

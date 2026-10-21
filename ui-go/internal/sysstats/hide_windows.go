//go:build windows

package sysstats

import (
	"os/exec"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

func dieWithParent(*exec.Cmd) {}

var (
	killJobOnce sync.Once
	killJob     windows.Handle
)

// adoptChild puts the started resident nvidia-smi into a kill-on-close job
// object: the job handle closes when this process exits (however it exits),
// which takes the child down with it, like Pdeathsig on Linux.
func adoptChild(cmd *exec.Cmd) {
	killJobOnce.Do(func() {
		h, err := windows.CreateJobObject(nil, nil)
		if err != nil {
			return
		}
		info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
		info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
		if _, err := windows.SetInformationJobObject(h, windows.JobObjectExtendedLimitInformation,
			uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info))); err != nil {
			windows.CloseHandle(h)
			return
		}
		killJob = h
	})
	if killJob == 0 || cmd.Process == nil {
		return
	}
	p, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(cmd.Process.Pid))
	if err != nil {
		return
	}
	defer windows.CloseHandle(p)
	_ = windows.AssignProcessToJobObject(killJob, p)
}

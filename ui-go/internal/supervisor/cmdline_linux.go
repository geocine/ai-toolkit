package supervisor

import (
	"os"
	"strconv"
	"strings"
)

func commandLine(pid int) (string, bool) {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cmdline")
	if err != nil {
		return "", false
	}
	return strings.ReplaceAll(string(b), "\x00", " "), true
}

//go:build !windows

package media

import (
	"os"
	"syscall"
)

// fileID is the inode, as in upstream's stat.ino.
func fileID(_ string, st os.FileInfo) uint64 {
	if s, ok := st.Sys().(*syscall.Stat_t); ok {
		return uint64(s.Ino)
	}
	return 0
}

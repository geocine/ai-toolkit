package photocheck

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/geocine/aitoolkit/internal/thumbs"
)

// cacheDirs are the trainer's per-image caches next to an image
// (toolkit/dataloader_mixins.py): <stem>_<urlsafe-b64(md5) without '='>.safetensors.
var cacheDirs = []string{"_latent_cache", "_t_e_cache", "_clip_vision_cache"}

// cacheName matches the cache files of one stem exactly: IMG_1 does not
// match IMG_1_2_<22> or IMG_1__face_<22>.
func cacheName(stem string) *regexp.Regexp {
	return regexp.MustCompile(`^` + regexp.QuoteMeta(stem) + `_[A-Za-z0-9_-]{22}\.safetensors$`)
}

// PurgeDerived removes what was made from one dataset photo: its .thumbs
// copies (.png and .jpg) and its files in the trainer's three caches. Every
// op that moves, rewrites, renames or deletes a photo calls it, and so do
// upload replace and file delete. rel is relative to dir, with "/".
func PurgeDerived(dir, rel string) {
	abs := filepath.Join(dir, filepath.FromSlash(rel))
	base := thumbs.Base(abs)
	os.Remove(base + ".png")
	os.Remove(base + ".jpg")
	purgeCaches(abs)
}

// purgeCaches removes the trainer's cache files for the image at abs.
func purgeCaches(abs string) {
	name := filepath.Base(abs)
	re := cacheName(strings.TrimSuffix(name, filepath.Ext(name)))
	for _, c := range cacheDirs {
		folder := filepath.Join(filepath.Dir(abs), c)
		entries, err := os.ReadDir(folder)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() && re.MatchString(e.Name()) {
				os.Remove(filepath.Join(folder, e.Name()))
			}
		}
	}
}

// moveThumbs moves a photo's .thumbs copies along with it (from and to are
// the photo's own paths). A thumb carries its source's mtime, which a
// rename keeps, so the moved thumb stays fresh.
func moveThumbs(from, to string) {
	src, dst := thumbs.Base(from), thumbs.Base(to)
	for _, ext := range []string{".png", ".jpg"} {
		if _, err := os.Stat(src + ext); err != nil {
			continue
		}
		if os.MkdirAll(filepath.Dir(dst), 0o755) != nil || os.Rename(src+ext, dst+ext) != nil {
			os.Remove(src + ext)
		}
	}
}

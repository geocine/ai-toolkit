package photocheck

import (
	"encoding/json"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/collation"
)

// SetAsideDir is the dataset folder photos are left out into. The leading
// dot makes the trainer, the captioner, walkDataset, bundles and portable
// sync skip it.
const SetAsideDir = ".set_aside"

// controlsDir is the trainer's control-image folder (toolkit/data_loader.py
// drops files whose parent folder has this name).
const controlsDir = "_controls"

// entry is one photo file in a dataset as a review sees it.
type entry struct {
	rel  string // dataset-relative with "/"; set-aside ones without the .set_aside/ prefix
	abs  string
	info fs.FileInfo
	sha  string   // "" until known
	m    *Metrics // nil while pending
}

// isPhoto reports a file the photo check looks at: what the trainer reads,
// plus the formats it skips that a photo check lists as unsupported.
func isPhoto(name string) bool {
	ext := strings.ToLower(path.Ext(name))
	return TrainerExts[ext] || ConvertibleExts[ext]
}

// walkPhotos lists a dataset's photos the way the trainer walks it: every
// folder below dir, skipping dot entries and _controls folders' files. The
// photos in .set_aside/ come back separately, with rel relative to that
// folder. Both lists are sorted by collation.Compare.
func walkPhotos(dir string) (items, aside []*entry, err error) {
	items, err = walkFolder(dir)
	if err != nil {
		return nil, nil, err
	}
	aside, err = walkFolder(filepath.Join(dir, SetAsideDir))
	if err != nil && !isNotExist(err) {
		return nil, nil, err
	}
	return items, aside, nil
}

func walkFolder(root string) ([]*entry, error) {
	var out []*entry
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == root {
				return err
			}
			return nil
		}
		if p == root {
			return nil
		}
		if strings.HasPrefix(d.Name(), ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() || !d.Type().IsRegular() || !isPhoto(d.Name()) {
			return nil
		}
		if filepath.Base(filepath.Dir(p)) == controlsDir {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(root, p)
		out = append(out, &entry{rel: filepath.ToSlash(rel), abs: p, info: info})
		return nil
	})
	slices.SortStableFunc(out, func(a, b *entry) int { return collation.Compare(a.rel, b.rel) })
	return out, err
}

// asideFile is a set-aside photo's dataset-relative path.
func asideFile(rel string) string { return SetAsideDir + "/" + rel }

// asideFromFile records where a set-aside photo came from when a name
// clash in .set_aside/ renamed it, so restore can put it back under its
// own name, and which set-aside photos are the originals a convert or a
// bake kept. Being a dot file, nothing that walks the folder lists it.
const asideFromFile = ".from.json"

// asideOrigin is one asideFromFile record: the dataset path, and the
// photo's sha256 when known, so a record never passes to another photo
// that later takes the name. Of is set on an original a convert or a bake
// kept: the dataset path of the photo made from it, with that photo's
// sha256 as it was made (OfSHA256), so restoring the original can tell the
// photo was not changed since.
type asideOrigin struct {
	From     string `json:"from,omitempty"`
	SHA256   string `json:"sha256,omitempty"`
	Of       string `json:"of,omitempty"`
	OfSHA256 string `json:"of_sha256,omitempty"`
}

// original is what a set-aside original was replaced by.
type original struct {
	of  string // the dataset path of the photo a convert or a bake made from it
	sha string // that photo's sha256 when it was made; "" in a preview
}

// readAside reads the records for the set-aside photos: from maps them to
// the dataset paths they came from, of maps the originals to the photos
// made from them. A record for a photo that is gone, or whose content
// changed, is dropped, and so is a path that is not a clean photo path
// inside the dataset (restore moves the photo there; the file may have
// been written by hand).
func readAside(dir string, aside []*entry) (from map[string]string, of map[string]original) {
	from, of = map[string]string{}, map[string]original{}
	data, err := os.ReadFile(filepath.Join(dir, SetAsideDir, asideFromFile))
	if err != nil {
		return from, of
	}
	var recs map[string]asideOrigin
	if json.Unmarshal(data, &recs) != nil {
		return from, of
	}
	// A drive letter is refused on every OS: "C:/x" is local on Linux but
	// leaves the dataset once the folder is opened on Windows.
	clean := func(p string) bool {
		_, rel, ok := photoPath(dir, p)
		return ok && rel == p && !strings.HasPrefix(rel, SetAsideDir+"/") && filepath.IsLocal(filepath.FromSlash(rel)) && !hasDrive(rel)
	}
	for _, e := range aside {
		r, ok := recs[e.rel]
		if !ok || (r.SHA256 != "" && e.sha != "" && r.SHA256 != e.sha) {
			continue
		}
		if r.From != "" && r.From != e.rel && clean(r.From) {
			from[e.rel] = r.From
		}
		if r.Of != "" && clean(r.Of) {
			of[e.rel] = original{of: r.Of, sha: r.OfSHA256}
		}
	}
	return from, of
}

// hasDrive reports whether p starts with a Windows drive letter ("C:").
func hasDrive(p string) bool {
	return len(p) >= 2 && p[1] == ':' && ('a' <= p[0]|0x20 && p[0]|0x20 <= 'z')
}

// writeAside stores the records for the set-aside photos, removing the
// file when there are none.
func writeAside(dir string, from map[string]string, of map[string]original, aside []*entry) error {
	p := filepath.Join(dir, SetAsideDir, asideFromFile)
	recs := map[string]asideOrigin{}
	for _, e := range aside {
		var r asideOrigin
		if f := from[e.rel]; f != "" && f != e.rel {
			r.From = f
		}
		if o, ok := of[e.rel]; ok {
			r.Of, r.OfSHA256 = o.of, o.sha
		}
		if r != (asideOrigin{}) {
			r.SHA256 = e.sha
			recs[e.rel] = r
		}
	}
	if len(recs) == 0 {
		if err := os.Remove(p); err != nil && !isNotExist(err) {
			return err
		}
		return nil
	}
	data, err := json.MarshalIndent(recs, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	return writeFileAtomic(p, data)
}

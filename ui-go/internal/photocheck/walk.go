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
// own name. Being a dot file, nothing that walks the folder lists it.
const asideFromFile = ".from.json"

// asideOrigin is one asideFromFile record: the dataset path, and the
// photo's sha256 when known, so a record never passes to another photo
// that later takes the name.
type asideOrigin struct {
	From   string `json:"from"`
	SHA256 string `json:"sha256,omitempty"`
}

// readAsideFrom maps the set-aside photos to the dataset paths recorded
// for them. A record for a photo that is gone, or whose content changed,
// is dropped, and so is one whose path is not a clean photo path inside
// the dataset (restore moves the photo there; the file may have been
// written by hand).
func readAsideFrom(dir string, aside []*entry) map[string]string {
	out := map[string]string{}
	data, err := os.ReadFile(filepath.Join(dir, SetAsideDir, asideFromFile))
	if err != nil {
		return out
	}
	var recs map[string]asideOrigin
	if json.Unmarshal(data, &recs) != nil {
		return out
	}
	for _, e := range aside {
		r, ok := recs[e.rel]
		if !ok || r.From == "" || r.From == e.rel || (r.SHA256 != "" && e.sha != "" && r.SHA256 != e.sha) {
			continue
		}
		if _, rel, ok := photoPath(dir, r.From); !ok || rel != r.From || strings.HasPrefix(rel, SetAsideDir+"/") ||
			!filepath.IsLocal(filepath.FromSlash(rel)) {
			continue
		}
		out[e.rel] = r.From
	}
	return out
}

// writeAsideFrom stores the records for the set-aside photos, removing
// the file when there are none.
func writeAsideFrom(dir string, from map[string]string, aside []*entry) error {
	p := filepath.Join(dir, SetAsideDir, asideFromFile)
	recs := map[string]asideOrigin{}
	for _, e := range aside {
		if f := from[e.rel]; f != "" && f != e.rel {
			recs[e.rel] = asideOrigin{From: f, SHA256: e.sha}
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

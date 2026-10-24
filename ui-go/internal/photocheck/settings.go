package photocheck

import (
	"context"
	"encoding/json"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/geocine/aitoolkit/internal/state"
)

// SaveSettings applies a photos/settings patch: the subject kind, the
// "this is them" reference and per-photo decisions. Files are named by
// their dataset path and stored by content, so a decision survives a
// rename or a trip through .set_aside/. It returns the new review.
func SaveSettings(ctx context.Context, dir string, p SettingsPatch) (*Review, error) {
	dir = filepath.Clean(dir)
	unlock := lockDir(dir)
	err := saveSettings(ctx, dir, p)
	unlock()
	if err != nil {
		return nil, err
	}
	return ReviewDataset(ctx, dir, Options{Wait: settleWait})
}

func saveSettings(ctx context.Context, dir string, p SettingsPatch) error {
	set, err := loadSet(ctx, dir)
	if err != nil {
		return err
	}
	if p.Kind != nil {
		if err := checkKind(*p.Kind); err != nil {
			return err
		}
		set.Kind = string(*p.Kind)
	}
	if ref := strings.TrimSpace(string(p.Reference)); ref != "" {
		var file *string
		if err := json.Unmarshal(p.Reference, &file); err != nil {
			return badRequest("reference must be a file name or null")
		}
		set.Reference = ""
		if file != nil && *file != "" {
			sha, err := photoSHA(ctx, dir, *file)
			if err != nil {
				return err
			}
			set.Reference = sha
		}
	}
	files := make([]string, 0, len(p.Decisions))
	for f := range p.Decisions {
		files = append(files, f)
	}
	slices.Sort(files)
	for _, f := range files {
		d := p.Decisions[f]
		if d != nil && *d != DecisionKeep && *d != DecisionIntended {
			return badRequest("decisions[%s] must be keep, intended or null", f)
		}
		sha, err := photoSHA(ctx, dir, f)
		if err != nil {
			return err
		}
		if d == nil {
			delete(set.Decisions, sha)
		} else {
			set.Decisions[sha] = string(*d)
		}
	}
	set.UpdatedAt = 0
	return state.PutPhotoSet(ctx, set)
}

// photoSHA is the content hash of a dataset photo named by its dataset
// path (a .set_aside/ path works too). A name that is not a photo of the
// dataset is a 400.
func photoSHA(ctx context.Context, dir, file string) (string, error) {
	abs, rel, ok := photoPath(dir, file)
	if !ok {
		return "", badRequest("%q is not a photo in this dataset", file)
	}
	st, err := os.Stat(abs)
	if err != nil || !st.Mode().IsRegular() {
		return "", badRequest("no photo named %s in this dataset", rel)
	}
	return state.FileSHA256(ctx, abs, st)
}

// photoPath resolves a dataset-relative photo name inside dir, refusing
// traversal and hidden folders other than .set_aside/.
func photoPath(dir, file string) (abs, rel string, ok bool) {
	rel = strings.TrimPrefix(path.Clean("/"+strings.ReplaceAll(file, `\`, "/")), "/")
	if rel == "" || rel == "." || !isPhoto(rel) {
		return "", rel, false
	}
	for i, part := range strings.Split(rel, "/") {
		if strings.HasPrefix(part, ".") && !(i == 0 && part == SetAsideDir) {
			return "", rel, false
		}
	}
	return filepath.Join(dir, filepath.FromSlash(rel)), rel, true
}

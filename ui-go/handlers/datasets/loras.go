package datasets

import (
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/geocine/aitoolkit/internal/httpx"
	"github.com/geocine/aitoolkit/internal/settings"
	"github.com/geocine/aitoolkit/internal/store"
)

type loraFile struct {
	Name  string  `json:"name"`
	Path  string  `json:"path"`
	Size  int64   `json:"size"`
	Mtime float64 `json:"mtime"`
}

type modelLora struct {
	Name    string  `json:"name"`
	Relpath string  `json:"relpath"`
	Path    string  `json:"path"`
	Size    int64   `json:"size"`
	Mtime   float64 `json:"mtime"`
}

// mtimeMs is Node's Stats.mtimeMs: sec * 1e3 + nsec / 1e6.
func mtimeMs(t time.Time) float64 {
	return float64(t.Unix())*1e3 + float64(t.Nanosecond())/1e6
}

// listSafetensors lists dir's .safetensors files newest first; any error
// yields an empty list.
func listSafetensors(dir string) []loraFile {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return []loraFile{}
	}
	out := []loraFile{}
	for _, e := range entries {
		if !e.Type().IsRegular() || !strings.HasSuffix(e.Name(), ".safetensors") {
			continue
		}
		p := filepath.Join(dir, e.Name())
		st, err := os.Stat(p)
		if err != nil {
			return []loraFile{}
		}
		out = append(out, loraFile{Name: e.Name(), Path: p, Size: st.Size(), Mtime: mtimeMs(st.ModTime())})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Mtime > out[j].Mtime })
	return out
}

// walkLoras collects .safetensors files up to 5 levels below root.
func walkLoras(root, rel string, depth int, out *[]modelLora) error {
	if depth > 5 {
		return nil
	}
	entries, err := os.ReadDir(filepath.Join(root, rel))
	if err != nil {
		return nil
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		r := e.Name()
		if rel != "" {
			r = filepath.Join(rel, e.Name())
		}
		if e.IsDir() {
			if err := walkLoras(root, r, depth+1, out); err != nil {
				return err
			}
		} else if e.Type().IsRegular() && strings.HasSuffix(e.Name(), ".safetensors") {
			p := filepath.Join(root, r)
			st, err := os.Stat(p)
			if err != nil {
				return err
			}
			*out = append(*out, modelLora{Name: e.Name(), Relpath: r, Path: p, Size: st.Size(), Mtime: mtimeMs(st.ModTime())})
		}
	}
	return nil
}

// listLoras returns every training job's saved checkpoints and everything
// under <MODELS_PATH>/loras (for the Generate page).
func listLoras(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	trainingFolder, err := settings.TrainingFolder(ctx)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	jobs, err := store.FindJobs(ctx, store.JobQuery{Where: `"job_type" = 'train'`, OrderBy: `"updated_at" DESC`})
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	type jobEntry struct {
		ID        string     `json:"id"`
		Name      string     `json:"name"`
		Status    string     `json:"status"`
		UpdatedAt store.Time `json:"updated_at"`
		Files     []loraFile `json:"files"`
	}
	jobEntries := []jobEntry{}
	for _, j := range jobs {
		files := listSafetensors(filepath.Join(trainingFolder, j.Name))
		if len(files) > 0 {
			jobEntries = append(jobEntries, jobEntry{j.ID, j.Name, j.Status, j.UpdatedAt, files})
		}
	}
	modelsPath, err := settings.ModelsPath(ctx)
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	lorasRoot := filepath.Join(modelsPath, "loras")
	models := []modelLora{}
	if err := walkLoras(lorasRoot, "", 0, &models); err != nil {
		httpx.Internal(w, r, err)
		return
	}
	sort.SliceStable(models, func(i, j int) bool { return localeCompare(models[i].Relpath, models[j].Relpath) < 0 })
	httpx.OK(w, struct {
		Jobs      []jobEntry  `json:"jobs"`
		Models    []modelLora `json:"models"`
		LorasRoot string      `json:"lorasRoot"`
	}{jobEntries, models, lorasRoot})
}

// Chunked upload of a .safetensors file into <MODELS_PATH>/loras.
//   POST ?action=start&fileName=x.safetensors&size=N          -> { uploadId }
//   POST ?action=chunk&uploadId=..&offset=N   (raw body bytes) -> { received }
//   POST ?action=finish&uploadId=..&fileName=..&size=N         -> { path, name }
//   POST ?action=cancel&uploadId=..                            -> { ok }
// Chunks are written at explicit offsets so a retried chunk is idempotent.

const (
	partDirName = ".uploads"
	staleAfter  = 24 * time.Hour
)

var uuidRe = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// cleanLoraName keeps a basename that must be a .safetensors file.
func cleanLoraName(raw string, present bool) string {
	if !present || raw == "" {
		return ""
	}
	name := jsTrim(raw)
	name = strings.NewReplacer(`\`, "_", "/", "_").Replace(name)
	name = replaceUTF16(name, func(r rune) bool {
		return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') ||
			strings.ContainsRune("_.- ()[]", r)
	}, "_")
	if !strings.HasSuffix(strings.ToLower(name), ".safetensors") || strings.HasPrefix(name, ".") || len(name) < 13 {
		return ""
	}
	return name
}

// parseSize is a positive safe integer, or 0 when invalid.
func parseSize(raw string, present bool) int64 {
	n, ok := jsNumber(raw, present)
	if !ok || !isSafeInteger(n) || n <= 0 {
		return 0
	}
	return int64(n)
}

func removeStaleParts(partDir string) {
	entries, err := os.ReadDir(partDir)
	if err != nil {
		return
	}
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".part") {
			continue
		}
		p := filepath.Join(partDir, e.Name())
		if st, err := os.Stat(p); err == nil && time.Since(st.ModTime()) > staleAfter {
			os.Remove(p)
		}
	}
}

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func uploadLora(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	param := func(k string) (string, bool) {
		v, ok := q[k]
		if !ok || len(v) == 0 {
			return "", false
		}
		return v[0], true
	}
	bad := func(msg string) { httpx.Error(w, http.StatusBadRequest, msg) }
	exists409 := func() {
		httpx.JSON(w, http.StatusConflict, struct {
			Error  string `json:"error"`
			Exists bool   `json:"exists"`
		}{"A file with that name already exists", true})
	}

	modelsPath, err := settings.ModelsPath(r.Context())
	if err != nil {
		httpx.Internal(w, r, err)
		return
	}
	lorasRoot := filepath.Join(modelsPath, "loras")
	if abs, err := filepath.Abs(lorasRoot); err == nil {
		lorasRoot = abs
	}
	partDir := filepath.Join(lorasRoot, partDirName)

	err = func() error {
		action, _ := param("action")
		overwrite, _ := param("overwrite")
		if action == "start" {
			fileName := cleanLoraName(param("fileName"))
			if fileName == "" {
				bad("File must be a .safetensors file")
				return nil
			}
			if parseSize(param("size")) == 0 {
				bad("Invalid file size")
				return nil
			}
			if overwrite != "1" && exists(filepath.Join(lorasRoot, fileName)) {
				exists409()
				return nil
			}
			if err := os.MkdirAll(partDir, 0o755); err != nil {
				return err
			}
			removeStaleParts(partDir)
			uploadID := uuid.NewString()
			// empty part file so chunk writes can open it r+
			if err := os.WriteFile(filepath.Join(partDir, uploadID+".part"), nil, 0o644); err != nil {
				return err
			}
			httpx.OK(w, struct {
				UploadID string `json:"uploadId"`
				FileName string `json:"fileName"`
			}{uploadID, fileName})
			return nil
		}

		uploadID, _ := param("uploadId")
		if !uuidRe.MatchString(uploadID) {
			bad("Invalid uploadId")
			return nil
		}
		part := filepath.Join(partDir, uploadID+".part")

		switch action {
		case "chunk":
			offset, ok := jsNumber(param("offset"))
			if !ok || !isSafeInteger(offset) || offset < 0 {
				bad("Invalid offset")
				return nil
			}
			if !exists(part) {
				httpx.Error(w, http.StatusNotFound, "Upload not found or expired")
				return nil
			}
			buf, err := io.ReadAll(r.Body)
			if err != nil {
				return err
			}
			if len(buf) == 0 {
				bad("Empty chunk")
				return nil
			}
			f, err := os.OpenFile(part, os.O_RDWR, 0)
			if err != nil {
				return err
			}
			_, werr := f.WriteAt(buf, int64(offset))
			if cerr := f.Close(); werr == nil {
				werr = cerr
			}
			if werr != nil {
				return werr
			}
			httpx.OK(w, map[string]int{"received": len(buf)})
		case "finish":
			fileName := cleanLoraName(param("fileName"))
			size := parseSize(param("size"))
			if fileName == "" || size == 0 {
				bad("Invalid fileName or size")
				return nil
			}
			if !exists(part) {
				httpx.Error(w, http.StatusNotFound, "Upload not found or expired")
				return nil
			}
			st, err := os.Stat(part)
			if err != nil {
				return err
			}
			if st.Size() != size {
				os.Remove(part)
				bad("Upload incomplete: got " + strconv.FormatInt(st.Size(), 10) + " of " + strconv.FormatInt(size, 10) + " bytes")
				return nil
			}
			dest := filepath.Join(lorasRoot, fileName)
			if overwrite != "1" && exists(dest) {
				os.Remove(part)
				exists409()
				return nil
			}
			if err := os.Rename(part, dest); err != nil {
				return err
			}
			httpx.OK(w, struct {
				Path string `json:"path"`
				Name string `json:"name"`
				Size int64  `json:"size"`
			}{dest, fileName, size})
		case "cancel":
			os.Remove(part)
			httpx.OK(w, map[string]bool{"ok": true})
		default:
			bad("Unknown action")
		}
		return nil
	}()
	if err != nil {
		log.Printf("LoRA upload error: %v", err)
		msg := err.Error()
		if msg == "" {
			msg = "Upload failed"
		}
		var pe *os.PathError
		if errors.As(err, &pe) {
			msg = pe.Op + " " + pe.Path + ": " + pe.Err.Error()
		}
		httpx.Error(w, http.StatusInternalServerError, msg)
	}
}

package bundles

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"path"
	"sort"
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/jsonx"
	"github.com/geocine/aitoolkit/internal/v1api"
)

// File is one zip entry to restore, Rel being its path inside the folder it
// restores into (a dataset or the job folder).
type File struct {
	Zip *zip.File
	Rel string
}

// PlanDataset is a dataset the bundle carries.
type PlanDataset struct {
	Ref DatasetRef
	// Source is what the bundle's config calls the dataset folder: the
	// aitk://datasets/<name> of a manifest, or a legacy config's folder_path.
	Source string
	Files  []File
	Bytes  int64
}

// Plan is a validated bundle: what goes where, before anything is written.
type Plan struct {
	Legacy   bool
	Manifest Manifest // read, or derived from a legacy config
	Datasets []PlanDataset
	JobFiles []File
	JobBytes int64
	Skipped  []string
}

// Bytes is everything the plan writes.
func (p *Plan) Bytes() int64 {
	n := p.JobBytes
	for _, d := range p.Datasets {
		n += d.Bytes
	}
	return n
}

func bad(msg string) error { return v1api.Errorf(v1api.CodeInvalidRequest, msg) }

// cleanName validates an entry name like planArchive in handlers/v1: a
// traversal, absolute path or symlink rejects the whole bundle.
func cleanName(f *zip.File) (string, error) {
	name := strings.ReplaceAll(f.Name, `\`, "/")
	if f.Mode()&fs.ModeSymlink != 0 {
		return "", bad("bundle contains a symlink: " + f.Name)
	}
	if strings.HasPrefix(name, "/") || (len(name) > 1 && name[1] == ':') {
		return "", bad("bundle contains an absolute path: " + f.Name)
	}
	for _, seg := range strings.Split(name, "/") {
		if seg == ".." {
			return "", bad("bundle contains a path leaving its folder: " + f.Name)
		}
	}
	return path.Clean(name), nil
}

var junkNames = map[string]bool{"Thumbs.db": true, "desktop.ini": true}

// junk is metadata and editor clutter: hidden files and folders
// (.ipynb_checkpoints, .DS_Store, .thumbs, .job_config.json which is
// regenerated), __MACOSX, Thumbs.db.
func junk(rel string) bool {
	if strings.HasPrefix(rel, "__MACOSX/") {
		return true
	}
	for _, seg := range strings.Split(rel, "/") {
		if strings.HasPrefix(seg, ".") || junkNames[seg] {
			return true
		}
	}
	return false
}

type entry struct {
	f   *zip.File
	rel string
}

// Read validates a bundle (or a legacy run zip) and plans its restore.
func Read(zr *zip.Reader) (*Plan, error) {
	var entries []entry
	var total uint64
	for _, f := range zr.File {
		name, err := cleanName(f)
		if err != nil {
			return nil, err
		}
		if f.FileInfo().IsDir() || strings.HasSuffix(f.Name, "/") {
			continue
		}
		total += f.UncompressedSize64
		entries = append(entries, entry{f, name})
	}
	if len(entries) > MaxEntries {
		return nil, v1api.Errorf(v1api.CodeTooLarge, fmt.Sprintf("bundle has %d files (max %d)", len(entries), MaxEntries))
	}
	if total > MaxBytes {
		return nil, v1api.Errorf(v1api.CodeTooLarge, "bundle unpacks to more than 1 TiB")
	}
	entries = stripTop(entries)

	p := &Plan{Skipped: []string{}}
	byName := map[string]*zip.File{}
	for _, e := range entries {
		byName[e.rel] = e.f
	}
	if mf, ok := byName[ManifestName]; ok {
		if err := p.readManifest(mf); err != nil {
			return nil, err
		}
	} else if err := p.readLegacy(entries, byName); err != nil {
		return nil, err
	}

	dirs := map[string]int{}
	for i, d := range p.Datasets {
		dirs[d.Ref.Dir] = i
	}
	jobPrefix := p.Manifest.JobDir + "/"
	for _, e := range entries {
		if e.rel == ManifestName {
			continue
		}
		top, rest, nested := strings.Cut(e.rel, "/")
		switch i, isData := dirs[top]; {
		case !nested || junk(e.rel):
			p.Skipped = append(p.Skipped, e.f.Name)
		case isData:
			p.Datasets[i].Files = append(p.Datasets[i].Files, File{e.f, rest})
			p.Datasets[i].Bytes += int64(e.f.UncompressedSize64)
		case strings.HasPrefix(e.rel, jobPrefix):
			p.JobFiles = append(p.JobFiles, File{e.f, strings.TrimPrefix(e.rel, jobPrefix)})
			p.JobBytes += int64(e.f.UncompressedSize64)
		default:
			p.Skipped = append(p.Skipped, e.f.Name) // output/generations, stray files
		}
	}
	if p.Legacy {
		// a legacy zip's data folders are whatever it holds
		kept := p.Datasets[:0]
		for _, d := range p.Datasets {
			if len(d.Files) > 0 {
				d.Ref.Images, d.Ref.Captions = count(d.Files)
				kept = append(kept, d)
			}
		}
		p.Datasets = kept
		p.Manifest.Datasets = []DatasetRef{}
		for _, d := range p.Datasets {
			p.Manifest.Datasets = append(p.Manifest.Datasets, d.Ref)
		}
		p.Manifest.Optimizer, p.Manifest.Samples = false, 0
		for _, f := range p.JobFiles {
			switch {
			case f.Rel == "optimizer.pt":
				p.Manifest.Optimizer = true
			case strings.HasPrefix(f.Rel, "samples/"):
				p.Manifest.Samples++
			}
			if inCache(f.Rel) {
				p.Manifest.Caches = true
			}
		}
		for _, d := range p.Datasets {
			for _, f := range d.Files {
				if inCache(f.Rel) {
					p.Manifest.Caches = true
				}
			}
		}
	}
	p.orderSaves()
	if len(p.JobFiles) == 0 && len(p.Datasets) == 0 {
		return nil, bad("bundle holds neither a job folder nor a dataset")
	}
	return p, nil
}

// orderSaves puts the job's saves last, oldest first: extraction then gives
// the newest save the newest ctime, which is how ai-toolkit picks the save
// to resume from. It also lists the saves and, for a legacy zip, reads the
// resume step from the newest one's metadata.
func (p *Plan) orderSaves() {
	type keyed struct {
		f    File
		step int
		save string
	}
	ks := make([]keyed, len(p.JobFiles))
	saves := map[string]int{}
	for i, f := range p.JobFiles {
		top, _, nested := strings.Cut(f.Rel, "/")
		k := keyed{f: f, step: -1}
		if step, ok := checkpointOf(top, p.Manifest.Name, nested); ok {
			k.step, k.save = step, top
			saves[top] = step
		}
		ks[i] = k
	}
	sort.SliceStable(ks, func(i, j int) bool {
		if ks[i].step != ks[j].step {
			return ks[i].step < ks[j].step
		}
		return ks[i].save < ks[j].save
	})
	for i, k := range ks {
		p.JobFiles[i] = k.f
	}
	if !p.Legacy {
		return
	}
	names := make([]string, 0, len(saves))
	for n := range saves {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool {
		if saves[names[i]] != saves[names[j]] {
			return saves[names[i]] < saves[names[j]]
		}
		return names[i] < names[j]
	})
	p.Manifest.Checkpoints = names
	p.Manifest.Step = 0
	for i := len(names) - 1; i >= 0; i-- {
		if step, ok := p.zipStep(names[i]); ok {
			p.Manifest.Step = step
			return
		}
		if s := saves[names[i]]; s != FinalStep {
			p.Manifest.Step = s
			return
		}
	}
}

// zipStep reads training_info.step from a save inside the zip.
func (p *Plan) zipStep(save string) (int, bool) {
	for _, f := range p.JobFiles {
		if f.Rel != save || !strings.EqualFold(path.Ext(f.Rel), ".safetensors") {
			continue
		}
		rc, err := f.Zip.Open()
		if err != nil {
			return 0, false
		}
		defer rc.Close()
		return metaStep(rc)
	}
	return 0, false
}

func count(files []File) (images, captions int) {
	for _, f := range files {
		if inCache(f.Rel) {
			continue
		}
		if IsMedia(f.Rel) {
			images++
		} else if strings.EqualFold(path.Ext(f.Rel), ".txt") {
			captions++
		}
	}
	return
}

func isModel(name string) bool {
	switch strings.ToLower(path.Ext(name)) {
	case ".safetensors", ".ckpt":
		return true
	}
	return false
}

// stepOf parses <name>_<step>.<ext>; -1 when there is no step.
func stepOf(file string) int {
	base := strings.TrimSuffix(file, path.Ext(file))
	i := strings.LastIndex(base, "_")
	if i < 0 {
		return -1
	}
	n, err := strconv.Atoi(base[i+1:])
	if err != nil {
		return -1
	}
	return n
}

// stripTop drops one folder every entry shares when the bundle's own
// layout is inside it (run.zip -> run/data/..., run/output/...).
func stripTop(entries []entry) []entry {
	if len(entries) == 0 {
		return entries
	}
	top, _, nested := strings.Cut(entries[0].rel, "/")
	if !nested || top == "data" || top == "output" || strings.HasPrefix(top, "data_") {
		return entries
	}
	for _, e := range entries {
		if t, _, ok := strings.Cut(e.rel, "/"); !ok || t != top {
			return entries
		}
	}
	out := make([]entry, len(entries))
	for i, e := range entries {
		out[i] = entry{e.f, strings.TrimPrefix(e.rel, top+"/")}
	}
	return out
}

func readAll(f *zip.File, limit int64) ([]byte, error) {
	if f.UncompressedSize64 > uint64(limit) {
		return nil, bad(fmt.Sprintf("%s is larger than %d bytes", f.Name, limit))
	}
	rc, err := f.Open()
	if err != nil {
		return nil, bad("corrupt bundle entry " + f.Name + ": " + err.Error())
	}
	defer rc.Close()
	return io.ReadAll(io.LimitReader(rc, limit))
}

func validJobDir(dir string) bool {
	rest, ok := strings.CutPrefix(dir, "output/")
	return ok && rest != "" && !strings.ContainsAny(rest, `/\`) && rest != "." && rest != ".." && !strings.HasPrefix(rest, ".")
}

func (p *Plan) readManifest(f *zip.File) error {
	raw, err := readAll(f, 16<<20)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &p.Manifest); err != nil {
		return bad(ManifestName + ": " + err.Error())
	}
	m := &p.Manifest
	if m.Format < 1 || m.Format > Format {
		return bad(fmt.Sprintf("%s is format %d; this server reads up to %d", ManifestName, m.Format, Format))
	}
	if m.JobDir == "" {
		m.JobDir = "output/" + m.Name
	}
	if !validJobDir(m.JobDir) {
		return bad(fmt.Sprintf("%s names an unusable job folder %q", ManifestName, m.JobDir))
	}
	if _, err := jsonx.ParseObject(m.Config); err != nil {
		return bad(ManifestName + ": config is not a JSON object")
	}
	seen := map[string]bool{}
	for _, d := range m.Datasets {
		if d.Dir == "" || strings.ContainsAny(d.Dir, `/\`) || d.Dir == "output" || hidden(d.Dir) || seen[d.Dir] {
			return bad(fmt.Sprintf("%s names an unusable dataset folder %q", ManifestName, d.Dir))
		}
		seen[d.Dir] = true
		p.Datasets = append(p.Datasets, PlanDataset{Ref: d, Source: "aitk://datasets/" + d.Name})
	}
	if m.Warnings == nil {
		m.Warnings = []string{}
	}
	return nil
}

var configNames = []string{"config.yaml", "config.yml", "config.json"}

// readLegacy reads a zip without a manifest: output/<x>/config.yaml is the
// job (name from its config.name), data/ its first dataset, data_<n>/ the
// n-th.
func (p *Plan) readLegacy(entries []entry, byName map[string]*zip.File) error {
	p.Legacy = true
	jobDirs := map[string]bool{}
	for _, e := range entries {
		rest, ok := strings.CutPrefix(e.rel, "output/")
		if !ok {
			continue
		}
		dir, file, ok := strings.Cut(rest, "/")
		if ok && dir != "generations" && !hidden(dir) && !strings.Contains(file, "/") {
			for _, c := range configNames {
				if file == c {
					jobDirs[dir] = true
				}
			}
		}
	}
	if len(jobDirs) == 0 {
		return bad("not a run bundle: no " + ManifestName + " and no output/<name>/config.yaml")
	}
	if len(jobDirs) > 1 {
		names := make([]string, 0, len(jobDirs))
		for d := range jobDirs {
			names = append(names, d)
		}
		sort.Strings(names)
		return bad("bundle holds several job folders (" + strings.Join(names, ", ") + "); expected one")
	}
	var dir string
	for d := range jobDirs {
		dir = d
	}
	var cfg *jsonx.Object
	for _, c := range configNames {
		f, ok := byName["output/"+dir+"/"+c]
		if !ok {
			continue
		}
		raw, err := readAll(f, 4<<20)
		if err != nil {
			return err
		}
		if strings.HasSuffix(c, ".json") {
			cfg, err = jsonx.ParseObject(raw)
		} else {
			cfg, err = YAMLObject(raw)
		}
		if err != nil {
			return bad(fmt.Sprintf("output/%s/%s: %v", dir, c, err))
		}
		break
	}
	inner := cfg.Object("config")
	if inner == nil || len(inner.Array("process")) == 0 {
		return bad(fmt.Sprintf("output/%s/config: config.process is missing", dir))
	}
	name := dir
	if n, ok := inner.Get("name"); ok {
		if s, ok := n.(string); ok && s != "" {
			name = s
		}
	}
	raw, err := json.Marshal(cfg)
	if err != nil {
		return err
	}
	p.Manifest = Manifest{Format: Format, Name: name, Type: "train", Config: raw, JobDir: "output/" + dir,
		Datasets: []DatasetRef{}, Warnings: []string{}}
	p.Manifest.BaseModel, p.Manifest.Arch, p.Manifest.TotalSteps = Describe(raw)

	// data/ is datasets[0], data_<n>/ datasets[n]; Source is the config's path
	var sources []string
	if p0, ok := inner.Array("process")[0].(*jsonx.Object); ok {
		for _, d := range p0.Array("datasets") {
			src := ""
			if o, ok := d.(*jsonx.Object); ok {
				if v, ok := o.Get("folder_path"); ok {
					src, _ = v.(string)
				}
			}
			sources = append(sources, src)
		}
	}
	var idx []int
	seen := map[int]bool{}
	for _, e := range entries {
		t, _, ok := strings.Cut(e.rel, "/")
		if !ok {
			continue
		}
		i := -1
		if t == "data" {
			i = 0
		} else if n, ok := strings.CutPrefix(t, "data_"); ok {
			if v, err := strconv.Atoi(n); err == nil && v > 0 && DatasetDir(v) == t {
				i = v
			}
		}
		if i >= 0 && !seen[i] {
			seen[i] = true
			idx = append(idx, i)
		}
	}
	sort.Ints(idx)
	for _, i := range idx {
		src := ""
		if i < len(sources) {
			src = sources[i]
		}
		p.Datasets = append(p.Datasets, PlanDataset{Ref: DatasetRef{Dir: DatasetDir(i)}, Source: src})
	}
	return nil
}

// Describe reads a job config's base model, arch and step count.
func Describe(config json.RawMessage) (base, arch string, steps *int) {
	cfg, err := jsonx.ParseObject(config)
	if err != nil {
		return
	}
	inner := cfg.Object("config")
	if inner == nil {
		return
	}
	procs := inner.Array("process")
	if len(procs) == 0 {
		return
	}
	p0, _ := procs[0].(*jsonx.Object)
	if p0 == nil {
		return
	}
	if model := p0.Object("model"); model != nil {
		if v, ok := model.Get("name_or_path"); ok {
			base, _ = v.(string)
		}
		if v, ok := model.Get("arch"); ok {
			arch, _ = v.(string)
		}
	}
	if train := p0.Object("train"); train != nil {
		if v, ok := train.Get("steps"); ok {
			if n, ok := v.(json.Number); ok {
				if i, err := n.Int64(); err == nil {
					s := int(i)
					steps = &s
				}
			}
		}
	}
	return
}

// PeekManifest reads aitk_bundle.json from the first bytes of a bundle this
// server wrote (it is the first entry, stored): enough to know the job, its
// step and base model while the rest is still downloading. false for a
// legacy zip or one written otherwise.
func PeekManifest(head []byte) (*Manifest, bool) {
	const local = 30
	if len(head) < local || binary.LittleEndian.Uint32(head) != 0x04034b50 {
		return nil, false
	}
	method := binary.LittleEndian.Uint16(head[8:])
	nameLen := int(binary.LittleEndian.Uint16(head[26:]))
	extraLen := int(binary.LittleEndian.Uint16(head[28:]))
	if method != zip.Store || len(head) < local+nameLen+extraLen || string(head[local:local+nameLen]) != ManifestName {
		return nil, false
	}
	var m Manifest
	if err := json.NewDecoder(bytes.NewReader(head[local+nameLen+extraLen:])).Decode(&m); err != nil || m.Format < 1 || m.Name == "" {
		return nil, false
	}
	return &m, true
}

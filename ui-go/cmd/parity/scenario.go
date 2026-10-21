package main

import (
	"fmt"
	"path/filepath"
	"strings"
)

// scenario is the request sequence both servers receive, in order. Later
// steps depend on state earlier ones create, so append rather than reorder.
func scenario() []Step {
	var s []Step
	s = append(s, coreSteps()...)
	s = append(s, jobFileSteps()...)
	s = append(s, datasetSteps()...)
	s = append(s, mediaSteps()...)
	s = append(s, systemSteps()...)
	s = append(s, finalSteps()...)
	return s
}

func jobConfig(name string, lr float64) map[string]any {
	return map[string]any{
		"job": "extension",
		"config": map[string]any{
			"name": name,
			"process": []any{map[string]any{
				"type":            "ui_trainer",
				"training_folder": "{{ROOT}}/output",
				"lr":              lr,
				"datasets":        []any{map[string]any{"folder_path": "{{ROOT}}/datasets/set1"}},
			}},
		},
		"meta": map[string]any{"name": "[name]", "version": "1.0"},
	}
}

func coreSteps() []Step {
	return []Step{
		{Name: "auth", Method: "GET", Path: "/api/auth"},
		{Name: "settings fresh", Method: "GET", Path: "/api/settings"},
		{Name: "settings save missing key", Method: "POST", Path: "/api/settings", Body: map[string]any{"HF_TOKEN": "x"}},
		{Name: "settings save", Method: "POST", Path: "/api/settings", Body: map[string]any{
			"HF_TOKEN": "hf_test", "TRAINING_FOLDER": "{{ROOT}}/output",
			"DATASETS_FOLDER": "{{ROOT}}/datasets", "MODELS_PATH": "",
		}},
		{Name: "settings after save", Method: "GET", Path: "/api/settings"},

		{Name: "jobs empty", Method: "GET", Path: "/api/jobs"},
		{Name: "job create", Method: "POST", Path: "/api/jobs", Capture: map[string]string{"job": "id"},
			Body: map[string]any{"name": "fixture_job", "gpu_ids": "0", "job_config": jobConfig("fixture_job", 0.0001)}},
		{Name: "job create duplicate", Method: "POST", Path: "/api/jobs",
			Body: map[string]any{"name": "fixture_job", "gpu_ids": "0", "job_config": jobConfig("fixture_job", 0.0001)}},
		{Name: "job create second", Method: "POST", Path: "/api/jobs", Capture: map[string]string{"job2": "id"},
			Body: map[string]any{"name": "second_job", "gpu_ids": "1", "job_config": jobConfig("second_job", 0.0002),
				"job_type": "caption", "job_ref": "{{ROOT}}/datasets/set1"}},
		{Name: "job update", Method: "POST", Path: "/api/jobs",
			Body: map[string]any{"id": "{{job}}", "name": "fixture_job", "gpu_ids": "0",
				"job_config": jobConfig("fixture_job", 0.0003), "job_ref": "ref-1"}},
		{Name: "job update missing", Method: "POST", Path: "/api/jobs",
			Body: map[string]any{"id": "00000000-0000-4000-8000-000000000000", "name": "nope", "gpu_ids": "0", "job_config": map[string]any{}}},
		{Name: "job by id", Method: "GET", Path: "/api/jobs?id={{job}}"},
		{Name: "job by missing id", Method: "GET", Path: "/api/jobs?id=missing"},
		{Name: "job by ref", Method: "GET", Path: "/api/jobs?job_ref=ref-1"},
		{Name: "jobs by type", Method: "GET", Path: "/api/jobs?job_type=caption"},
		{Name: "jobs all", Method: "GET", Path: "/api/jobs"},

		{Name: "job start", Method: "GET", Path: "/api/jobs/{{job}}/start"},
		{Name: "job after start", Method: "GET", Path: "/api/jobs?id={{job}}"},
		{Name: "jobs active", Method: "GET", Path: "/api/jobs?only_active=true"},
		{Name: "queue list", Method: "GET", Path: "/api/queue"},
		{Name: "job sample_now", Method: "GET", Path: "/api/jobs/{{job}}/sample_now"},
		{Name: "job save_now", Method: "GET", Path: "/api/jobs/{{job}}/save_now"},
		{Name: "job stop (no pid)", Method: "GET", Path: "/api/jobs/{{job}}/stop"},
		{Name: "job mark_stopped", Method: "GET", Path: "/api/jobs/{{job}}/mark_stopped"},
		{Name: "job kill (no pid)", Method: "GET", Path: "/api/jobs/{{job}}/kill"},
		{Name: "job after kill", Method: "GET", Path: "/api/jobs?id={{job}}"},

		{Name: "queue start existing", Method: "GET", Path: "/api/queue/0/start"},
		{Name: "queue start new", Method: "GET", Path: "/api/queue/2,3/start"},
		{Name: "queue stop existing", Method: "GET", Path: "/api/queue/0/stop"},
		{Name: "queue stop missing", Method: "GET", Path: "/api/queue/9/stop"},
		{Name: "queue list after", Method: "GET", Path: "/api/queue"},

		{Name: "missing job start", Method: "GET", Path: "/api/jobs/missing/start"},
		{Name: "missing job stop", Method: "GET", Path: "/api/jobs/missing/stop"},
		{Name: "missing job delete", Method: "GET", Path: "/api/jobs/missing/delete"},
		{Name: "missing job kill", Method: "GET", Path: "/api/jobs/missing/kill"},
		{Name: "missing job save_now", Method: "GET", Path: "/api/jobs/missing/save_now", Mode: Status},
		{Name: "missing job mark_stopped", Method: "GET", Path: "/api/jobs/missing/mark_stopped", Mode: Status},
		{Name: "unknown route", Method: "GET", Path: "/api/does-not-exist", Mode: Status},
		{Name: "wrong method", Method: "DELETE", Path: "/api/jobs", Mode: Status},
	}
}

// jobFileSteps read the fixture_job folder written by writeFixtures.
func jobFileSteps() []Step {
	j := "/api/jobs/{{job}}"
	return []Step{
		{Name: "jobfiles files", Method: "GET", Path: j + "/files"},
		{Name: "jobfiles files missing job", Method: "GET", Path: "/api/jobs/missing/files"},
		{Name: "jobfiles log", Method: "GET", Path: j + "/log"},
		{Name: "jobfiles log offset", Method: "GET", Path: j + "/log?offset=7"},
		{Name: "jobfiles log offset past end", Method: "GET", Path: j + "/log?offset=99999"},
		{Name: "jobfiles log offset -1", Method: "GET", Path: j + "/log?offset=-1"},
		{Name: "jobfiles log offset -5", Method: "GET", Path: j + "/log?offset=-5"},
		{Name: "jobfiles log offset junk", Method: "GET", Path: j + "/log?offset=abc"},
		{Name: "jobfiles log second job (no folder)", Method: "GET", Path: "/api/jobs/{{job2}}/log"},
		{Name: "jobfiles loss", Method: "GET", Path: j + "/loss"},
		{Name: "jobfiles loss lr stride", Method: "GET", Path: j + "/loss?key=lr&stride=7&limit=3"},
		{Name: "jobfiles loss since", Method: "GET", Path: j + "/loss?since_step=25"},
		{Name: "jobfiles loss text key", Method: "GET", Path: j + "/loss?key=note"},
		{Name: "jobfiles loss bad stride", Method: "GET", Path: j + "/loss?stride=x"},
		{Name: "jobfiles loss bad limit", Method: "GET", Path: j + "/loss?limit=x", Mode: Status},
		{Name: "jobfiles loss no db", Method: "GET", Path: "/api/jobs/{{job2}}/loss"},
		{Name: "jobfiles loss delete range", Method: "DELETE", Path: j + "/loss", Body: map[string]any{"min_step": 10, "max_step": "12"}},
		{Name: "jobfiles loss after delete", Method: "GET", Path: j + "/loss?since_step=8&limit=6"},
		{Name: "jobfiles loss delete bad", Method: "DELETE", Path: j + "/loss", Body: map[string]any{"min_step": "x"}},
		{Name: "jobfiles notes empty", Method: "GET", Path: j + "/notes"},
		{Name: "jobfiles notes save", Method: "POST", Path: j + "/notes", Body: map[string]any{"content": "# notes\nlr looks high"}},
		{Name: "jobfiles notes read", Method: "GET", Path: j + "/notes"},
		{Name: "jobfiles notes non-string", Method: "POST", Path: j + "/notes", Body: map[string]any{"content": 42}},
		{Name: "jobfiles notes after non-string", Method: "GET", Path: j + "/notes"},
		{Name: "jobfiles plugin none", Method: "GET", Path: j + "/plugin"},
		{Name: "jobfiles plugin check", Method: "GET", Path: j + "/plugin?check=1"},
		{Name: "jobfiles samples", Method: "GET", Path: j + "/samples"},
		{Name: "jobfiles samples missing job", Method: "GET", Path: "/api/jobs/missing/samples"},
	}
}

func datasetSteps() []Step {
	lora := []byte("0123456789abcdef") // 16 bytes, uploaded as two chunks
	up := "/api/loras/upload?action="
	return []Step{
		{Name: "datasets list", Method: "GET", Path: "/api/datasets/list"},
		{Name: "datasets listImages", Method: "POST", Path: "/api/datasets/listImages", Body: map[string]any{"datasetName": "set1"}},
		{Name: "datasets listImages spaced", Method: "POST", Path: "/api/datasets/listImages", Body: map[string]any{"datasetName": "set 2"}},
		{Name: "datasets listImages missing", Method: "POST", Path: "/api/datasets/listImages", Body: map[string]any{"datasetName": "nope"}},
		{Name: "datasets listImages traversal", Method: "POST", Path: "/api/datasets/listImages", Body: map[string]any{"datasetName": "../output"}},
		{Name: "datasets listImages non-string", Method: "POST", Path: "/api/datasets/listImages", Body: map[string]any{"datasetName": 5}},
		{Name: "datasets create", Method: "POST", Path: "/api/datasets/create", Body: map[string]any{"name": "My New Set!"}},
		{Name: "datasets create again", Method: "POST", Path: "/api/datasets/create", Body: map[string]any{"name": "my new set"}},
		{Name: "datasets create bad", Method: "POST", Path: "/api/datasets/create", Body: map[string]any{"name": 7}},
		{Name: "datasets upload", Method: "POST", Path: "/api/datasets/upload", Body: Multipart{
			Fields: map[string]string{"datasetName": "my_new_set_"},
			Files: []MultipartFile{
				{Field: "files", Name: "up one.png", Data: pngImage(6, 6, true)},
				{Field: "files", Name: "dir/two.txt", Data: []byte("caption two")},
			},
		}},
		{Name: "datasets list after upload", Method: "GET", Path: "/api/datasets/list"},
		{Name: "datasets listImages uploaded", Method: "POST", Path: "/api/datasets/listImages", Body: map[string]any{"datasetName": "my_new_set_"}},
		{Name: "datasets delete", Method: "POST", Path: "/api/datasets/delete", Body: map[string]any{"name": "my_new_set_"}},
		{Name: "datasets delete traversal", Method: "POST", Path: "/api/datasets/delete", Body: map[string]any{"name": "../output"}},
		{Name: "datasets list after delete", Method: "GET", Path: "/api/datasets/list"},

		{Name: "loras empty", Method: "GET", Path: "/api/loras", Ignore: []string{"mtime"}},
		{Name: "loras upload start", Method: "POST", Path: up + "start&fileName=my%20lora.safetensors&size=16",
			Capture: map[string]string{"upload": "uploadId"}},
		{Name: "loras upload bad name", Method: "POST", Path: up + "start&fileName=x.bin&size=16"},
		{Name: "loras upload bad size", Method: "POST", Path: up + "start&fileName=x.safetensors&size=0"},
		{Name: "loras upload chunk 1", Method: "POST", Path: up + "chunk&uploadId={{upload}}&offset=0", Body: Raw(lora[:10])},
		{Name: "loras upload chunk 2", Method: "POST", Path: up + "chunk&uploadId={{upload}}&offset=10", Body: Raw(lora[10:])},
		{Name: "loras upload bad offset", Method: "POST", Path: up + "chunk&uploadId={{upload}}&offset=-1", Body: Raw(lora[:1])},
		{Name: "loras upload finish", Method: "POST", Path: up + "finish&uploadId={{upload}}&fileName=my%20lora.safetensors&size=16"},
		{Name: "loras upload exists", Method: "POST", Path: up + "start&fileName=my%20lora.safetensors&size=16"},
		{Name: "loras upload bad id", Method: "POST", Path: up + "chunk&uploadId=../../x&offset=0", Body: Raw(lora)},
		{Name: "loras upload unknown action", Method: "POST", Path: up + "nope&uploadId={{upload}}"},
		{Name: "loras list", Method: "GET", Path: "/api/loras", Ignore: []string{"mtime"}},
	}
}

// encodeURIComponent matches JavaScript's: everything but A-Z a-z 0-9 - _ . ! ~ * ' ( ) is escaped.
func encodeURIComponent(s string) string {
	var b strings.Builder
	for _, c := range []byte(s) {
		if c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || strings.IndexByte("-_.!~*'()", c) >= 0 {
			b.WriteByte(c)
		} else {
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}

// native joins root and slash-separated rel with the OS separator.
func native(root, rel string) string { return filepath.Join(root, filepath.FromSlash(rel)) }

// uiURL encodes a file path the way the UI does (encodeFilePathForUrl in
// ui/src/utils/basic.ts): `<encoded folder>/<encoded filename>`.
func uiURL(prefix, p string) string {
	idx := max(strings.LastIndex(p, "/"), strings.LastIndex(p, `\`))
	folder := p[:idx]
	if idx == 0 {
		folder = p[:1]
	}
	return prefix + encodeURIComponent(folder) + "/" + encodeURIComponent(p[idx+1:])
}

// legacyURL is the older single-segment form: encodeURIComponent(fullPath).
func legacyURL(prefix, p string) string { return prefix + encodeURIComponent(p) }

// proxyURL is what a proxy that decodes %2F and %5C forwards.
func proxyURL(prefix, p string, dropRoot bool) string {
	segs := strings.Split(strings.ReplaceAll(p, `\`, "/"), "/")
	for i, s := range segs {
		segs[i] = encodeURIComponent(s)
	}
	out := strings.Join(segs, "/")
	if dropRoot {
		out = strings.TrimPrefix(out, "/")
		if i := strings.Index(out, "%3A/"); i == 1 { // Windows drive letter
			out = out[i+4:]
		}
	}
	return prefix + out
}

func mediaSteps() []Step {
	a := func(root string) string { return native(root, "datasets/set1/a.png") }
	return []Step{
		{Name: "media img UI form", Method: "GET", Mode: Bytes, Headers: []string{"Cache-Control"},
			PathFn: func(r string) string { return uiURL("/api/img/", a(r)) }},
		{Name: "media img HEAD", Method: "HEAD", Mode: Status, Headers: []string{"Content-Type", "Content-Length"},
			PathFn: func(r string) string { return uiURL("/api/img/", a(r)) }},
		{Name: "media img legacy", Method: "GET", Mode: Bytes,
			PathFn: func(r string) string { return legacyURL("/api/img/", native(r, "datasets/set1/b.jpg")) }},
		{Name: "media img % & name", Method: "GET", Mode: Bytes,
			PathFn: func(r string) string { return uiURL("/api/img/", native(r, "datasets/set 2/100% & more.png")) }},
		{Name: "media img .. in name", Method: "GET", Mode: Bytes,
			PathFn: func(r string) string { return uiURL("/api/img/", native(r, "datasets/set 2/a..b.png")) }},
		{Name: "media img sample", Method: "GET", Mode: Bytes,
			PathFn: func(r string) string {
				return uiURL("/api/img/", native(r, "output/fixture_job/samples/1700000000000__000000100_0.jpg"))
			}},
		{Name: "media img range", Method: "GET", Mode: Bytes, ReqHeaders: map[string]string{"Range": "bytes=0-7"},
			Headers: []string{"Content-Range", "Content-Length"},
			PathFn:  func(r string) string { return uiURL("/api/img/", a(r)) }},
		{Name: "media img range unsatisfiable", Method: "GET", Mode: Status, ReqHeaders: map[string]string{"Range": "bytes=99999-"},
			PathFn: func(r string) string { return uiURL("/api/img/", a(r)) }},
		{Name: "media img thumb png (alpha)", Method: "GET", Mode: Media,
			PathFn: func(r string) string { return uiURL("/api/img/", a(r)) + "?thumb=1" }},
		{Name: "media img thumb jpg", Method: "GET", Mode: Media,
			PathFn: func(r string) string { return uiURL("/api/img/", native(r, "datasets/set1/b.jpg")) + "?thumb=1" }},
		{Name: "media img proxy-decoded", Method: "GET", Mode: Bytes,
			PathFn: func(r string) string { return proxyURL("/api/img/", a(r), false) }},
		{Name: "media img traversal", Method: "GET",
			PathFn: func(r string) string { return uiURL("/api/img/", native(r, "datasets/set1/../../outside-secret.png")) }},
		{Name: "media img outside roots", Method: "GET",
			PathFn: func(r string) string { return uiURL("/api/img/", native(r, "outside-secret.png")) }},
		{Name: "media img missing", Method: "GET",
			PathFn: func(r string) string { return uiURL("/api/img/", native(r, "datasets/set1/nope.png")) }},
		{Name: "media files download", Method: "GET", Mode: Bytes, Headers: []string{"Content-Disposition", "Content-Type"},
			PathFn: func(r string) string {
				return uiURL("/api/files/", native(r, "output/fixture_job/fixture_job.safetensors"))
			}},
		{Name: "media files outside", Method: "GET",
			PathFn: func(r string) string { return uiURL("/api/files/", native(r, "outside-secret.png")) }},
		{Name: "media audio art no tag", Method: "GET", Mode: Status,
			PathFn: func(r string) string { return uiURL("/api/audio/art/", native(r, "datasets/set1/song.mp3")) }},

		{Name: "media caption get", Method: "POST", Path: "/api/caption/get", BodyFn: func(r string) any { return map[string]any{"imgPath": a(r)} }},
		{Name: "media caption get missing", Method: "POST", Path: "/api/caption/get",
			BodyFn: func(r string) any { return map[string]any{"imgPath": native(r, "datasets/set1/c.png")} }},
		{Name: "media caption getBatch", Method: "POST", Path: "/api/caption/getBatch", BodyFn: func(r string) any {
			return map[string]any{"imgPaths": []any{a(r), native(r, "datasets/set1/b.jpg"), native(r, "datasets/set1/c.png")}}
		}},
		{Name: "media caption save", Method: "POST", Path: "/api/img/caption", BodyFn: func(r string) any {
			return map[string]any{"imgPath": native(r, "datasets/set1/c.png"), "caption": "a small square"}
		}},
		{Name: "media caption after save", Method: "POST", Path: "/api/caption/get",
			BodyFn: func(r string) any { return map[string]any{"imgPath": native(r, "datasets/set1/c.png")} }},
		{Name: "media img delete", Method: "POST", Path: "/api/img/delete",
			BodyFn: func(r string) any { return map[string]any{"imgPath": native(r, "datasets/set1/c.png")} }},
		{Name: "media listImages after delete", Method: "POST", Path: "/api/datasets/listImages", Body: map[string]any{"datasetName": "set1"}},
		{Name: "media files delete", Method: "POST", Path: "/api/files/delete", BodyFn: func(r string) any {
			return map[string]any{"filePath": native(r, "output/fixture_job/fixture_job_000000100.safetensors")}
		}},
		{Name: "media files delete outside", Method: "POST", Path: "/api/files/delete",
			BodyFn: func(r string) any { return map[string]any{"filePath": native(r, "outside-secret.png")} }},
		{Name: "media job files after delete", Method: "GET", Path: "/api/jobs/{{job}}/files"},
		{Name: "media img upload", Method: "POST", Path: "/api/img/upload", Body: Multipart{
			Files: []MultipartFile{{Field: "files", Name: "control.png", Data: pngImage(5, 5, false)}},
		}},
		{Name: "media zip dataset", Method: "POST", Path: "/api/zip",
			Body: map[string]any{"zipTarget": "dataset", "datasetName": "set1"}},
		{Name: "media zip dataset captions", Method: "POST", Path: "/api/zip",
			Body: map[string]any{"zipTarget": "dataset_captions", "datasetName": "set1"}},
		{Name: "media zip dataset contents", Method: "GET", Mode: Zip,
			PathFn: func(r string) string { return uiURL("/api/files/", native(r, "datasets/.zips/set1_dataset.zip")) }},
		{Name: "media zip captions contents", Method: "GET", Mode: Zip,
			PathFn: func(r string) string { return uiURL("/api/files/", native(r, "datasets/.zips/set1_captions.zip")) }},
		{Name: "media zip dataset missing name", Method: "POST", Path: "/api/zip", Body: map[string]any{"zipTarget": "dataset"}},
		{Name: "media zip samples", Method: "POST", Path: "/api/zip", Body: map[string]any{"jobName": "fixture_job"}},
		{Name: "media zip samples contents", Method: "GET", Mode: Zip,
			PathFn: func(r string) string { return uiURL("/api/files/", native(r, "output/fixture_job/samples.zip")) }},
		{Name: "media zip no job", Method: "POST", Path: "/api/zip", Body: map[string]any{}},
	}
}

// systemSteps cover machine stats (shape only: values differ moment to
// moment), extension UIs, scripts and the inference proxy with no engine.
// /api/monitor is an endless SSE stream and is covered by unit tests instead.
func systemSteps() []Step {
	return []Step{
		{Name: "system cpu", Method: "GET", Path: "/api/cpu", Mode: Shape},
		{Name: "system gpu", Method: "GET", Path: "/api/gpu", Mode: Shape},
		{Name: "system model_archs", Method: "GET", Path: "/api/model_archs", Mode: Shape},
		{Name: "system ostris_cloud", Method: "GET", Path: "/api/ostris_cloud"},
		{Name: "system scripts run", Method: "POST", Path: "/api/scripts",
			Body: map[string]any{"script": "parity_echo.py", "args": map[string]any{"name": "x y", "flag": true, "off": false}}},
		{Name: "system scripts list args", Method: "POST", Path: "/api/scripts",
			Body: map[string]any{"script": "parity_echo.py", "args": []any{"a", 2, true, nil}}},
		{Name: "system scripts stream", Method: "POST", Path: "/api/scripts", Mode: Status, Headers: []string{"Content-Type"},
			Body: map[string]any{"script": "parity_echo.py", "stream": true}},
		{Name: "system scripts failing", Method: "POST", Path: "/api/scripts", Body: map[string]any{"script": "parity_fail.py"}},
		{Name: "system scripts traversal", Method: "POST", Path: "/api/scripts", Body: map[string]any{"script": "../run.py"}},
		{Name: "system scripts missing", Method: "POST", Path: "/api/scripts", Body: map[string]any{"script": "nope.py"}},
		{Name: "system scripts bad args", Method: "POST", Path: "/api/scripts",
			Body: map[string]any{"script": "parity_echo.py", "args": []any{map[string]any{}}}},
		{Name: "inference status", Method: "GET", Path: "/api/inference/status"},
		{Name: "inference no engine", Method: "GET", Path: "/api/inference/models"},
		{Name: "inference post no engine", Method: "POST", Path: "/api/inference/generate", Body: map[string]any{"prompt": "x"}},
		{Name: "inference outputs delete", Method: "POST", Path: "/api/inference/outputs/delete",
			BodyFn: func(r string) any { return map[string]any{"paths": []any{native(r, "outside-secret.png")}} }},
		{Name: "inference delete no engine", Method: "DELETE", Path: "/api/inference/jobs/1"},
	}
}

// finalSteps run last: they destroy state other steps read.
func finalSteps() []Step {
	return []Step{
		{Name: "job delete", Method: "GET", Path: "/api/jobs/{{job2}}/delete"},
		{Name: "jobs after delete", Method: "GET", Path: "/api/jobs"},
	}
}

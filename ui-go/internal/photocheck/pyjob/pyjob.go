// Package pyjob holds the face check's Python script (photo_faces.py,
// section 12.3) and the opencv_zoo models it fetches. The worker writes the
// script into the job folder and runs it with the toolkit's Python; the Go
// judge reads the faces.json it leaves behind.
package pyjob

import (
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
)

// ScriptName is the script's file name in the job folder.
const ScriptName = "photo_faces.py"

// ProcessType is config.process[0].type of a face check job.
const ProcessType = "aitk_photo_faces"

// ModelsSubdir is the folder under the models path that holds the models.
const ModelsSubdir = "opencv_zoo"

//go:embed photo_faces.py
var script []byte

// Script returns the embedded script.
func Script() []byte { return append([]byte(nil), script...) }

// Model is one pinned opencv_zoo file. The pins come from the files' git LFS
// objects (sha256 and size), checked against a download of each.
type Model struct {
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// File is the model's file name in the models folder.
func (m Model) File() string { return filepath.Base(m.URL) }

const zoo = "https://github.com/opencv/opencv_zoo/raw/main/models/"

// Models are the face check's models by config key (section 12.2): yunet4
// is YuNet 2023mar (fixed input shape; OpenCV 4.x), yunet5 the same weights
// re-exported with dynamic dims for OpenCV 5.x, sface the recognizer and
// ppocr the text detector (only when text is on). photo_faces.py carries
// the same table for --selftest; pyjob_test keeps the two in step.
var Models = map[string]Model{
	"yunet4": {
		URL:    zoo + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
		SHA256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
		Bytes:  232589,
	},
	"yunet5": {
		URL:    zoo + "face_detection_yunet/face_detection_yunet_2026may.onnx",
		SHA256: "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
		Bytes:  229738,
	},
	"sface": {
		URL:    zoo + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
		SHA256: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
		Bytes:  38696353,
	},
	"ppocr": {
		URL:    zoo + "text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
		SHA256: "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587",
		Bytes:  2423490,
	},
}

// Write puts the embedded script in dir (the job folder, created when
// missing) as photo_faces.py and returns its path. It replaces an older
// copy atomically, so a re-queued job always runs this server's script.
func Write(dir string) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("face check script: %w", err)
	}
	path := filepath.Join(dir, ScriptName)
	tmp, err := os.CreateTemp(dir, "."+ScriptName+".*")
	if err != nil {
		return "", fmt.Errorf("face check script: %w", err)
	}
	_, werr := tmp.Write(script)
	cerr := tmp.Close()
	if werr == nil {
		werr = cerr
	}
	if werr == nil {
		werr = os.Chmod(tmp.Name(), 0o644)
	}
	if werr == nil {
		werr = os.Rename(tmp.Name(), path)
	}
	if werr != nil {
		os.Remove(tmp.Name())
		return "", fmt.Errorf("face check script: %w", werr)
	}
	return path, nil
}

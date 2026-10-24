package paths

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestSetRoot(t *testing.T) {
	saved := []string{ToolkitRoot, DefaultTrainFolder, DefaultDatasetsFolder, DefaultDataRoot, DefaultModelsFolder, DBPath}
	t.Cleanup(func() {
		ToolkitRoot, DefaultTrainFolder, DefaultDatasetsFolder, DefaultDataRoot, DefaultModelsFolder, DBPath =
			saved[0], saved[1], saved[2], saved[3], saved[4], saved[5]
	})
	tmp := t.TempDir()
	SetRoot(tmp)
	if ToolkitRoot != tmp || DBPath != filepath.Join(tmp, "aitk_db.db") {
		t.Fatalf("root %q, db %q", ToolkitRoot, DBPath)
	}
	for name, p := range map[string]string{"train": DefaultTrainFolder, "datasets": DefaultDatasetsFolder,
		"data": DefaultDataRoot, "models": DefaultModelsFolder} {
		if filepath.Dir(p) != tmp || !strings.HasPrefix(p, tmp) {
			t.Errorf("%s folder %q is not under %q", name, p, tmp)
		}
	}
}

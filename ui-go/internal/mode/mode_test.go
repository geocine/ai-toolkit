package mode

import "testing"

func TestFromEnv(t *testing.T) {
	for _, tc := range []struct {
		env       string
		studioAPI bool
		want      Mode
		err       bool
	}{
		{"", false, Standalone, false},
		{"", true, StudioAPI, false},
		{"worker", false, Worker, false},
		{" Studio ", false, Studio, false},
		{"studio-api", true, StudioAPI, false},
		{"worker", true, "", true},
		{"gpu", false, "", true},
	} {
		t.Setenv("AITK_MODE", tc.env)
		got, err := FromEnv(tc.studioAPI)
		if (err != nil) != tc.err || got != tc.want {
			t.Errorf("AITK_MODE=%q studioAPI=%v: %q, %v", tc.env, tc.studioAPI, got, err)
		}
	}
}

// The GPU image's binary refuses studio-api, however it is asked for.
func TestGPUImageRefusesStudioAPI(t *testing.T) {
	image = "gpu"
	t.Cleanup(func() { image = "" })
	for _, tc := range []struct {
		env       string
		studioAPI bool
	}{{"studio-api", false}, {"", true}, {"studio-api", true}} {
		t.Setenv("AITK_MODE", tc.env)
		if _, err := FromEnv(tc.studioAPI); err == nil {
			t.Errorf("AITK_MODE=%q studioAPI=%v accepted on the GPU image", tc.env, tc.studioAPI)
		}
	}
	for _, m := range []string{"worker", "standalone", "studio"} {
		t.Setenv("AITK_MODE", m)
		if got, err := FromEnv(false); err != nil || string(got) != m {
			t.Errorf("%s on the GPU image: %q, %v", m, got, err)
		}
	}
}

func TestSurfaces(t *testing.T) {
	type s struct{ console, studio, mcp, trainer bool }
	for m, want := range map[Mode]s{
		Worker:     {false, false, false, true},
		Standalone: {true, false, true, true},
		Studio:     {false, true, true, true},
		StudioAPI:  {false, true, true, false},
	} {
		if got := (s{m.Console(), m.StudioUI(), m.MCP(), m.Trainer()}); got != want {
			t.Errorf("%s: %+v, want %+v", m, got, want)
		}
	}
}

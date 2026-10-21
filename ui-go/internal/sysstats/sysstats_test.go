package sysstats

import (
	"encoding/json"
	"testing"
)

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestJSParse(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"42", "42"}, {"  42 MiB", "42"}, {"-7", "-7"}, {"0x1A", "26"},
		{"[N/A]", "null"}, {"", "null"}, {"3.9", "3"},
	}
	for _, c := range cases {
		if got := mustJSON(t, ParseInt(c.in)); got != c.want {
			t.Errorf("ParseInt(%q) = %s, want %s", c.in, got, c.want)
		}
	}
	fcases := []struct{ in, want string }{
		{"123.45", "123.45"}, {"[N/A]", "null"}, {"1e3", "1000"}, {".5W", "0.5"}, {"Infinity", "null"},
	}
	for _, c := range fcases {
		if got := mustJSON(t, ParseFloat(c.in)); got != c.want {
			t.Errorf("ParseFloat(%q) = %s, want %s", c.in, got, c.want)
		}
	}
}

const smiOutput = "1, NVIDIA GeForce RTX 3090, 591.44, 41, 3, 1, 24576, 23110, 1104, 31.52, 350.00, 210, 405, 0\r\n" +
	"0, NVIDIA GB10, 580.95.05, 45, 7, [N/A], [N/A], [N/A], [N/A], 11.20, [N/A], 2405, [N/A], [N/A]\r\n"

func TestParseRouteGPUs(t *testing.T) {
	gpus := ParseRouteGPUs(smiOutput)
	if len(gpus) != 2 {
		t.Fatalf("got %d gpus", len(gpus))
	}
	want0 := `{"index":1,"name":"NVIDIA GeForce RTX 3090","driverVersion":"591.44","temperature":41,` +
		`"utilization":{"gpu":3,"memory":1},"memory":{"total":24576,"free":23110,"used":1104},` +
		`"power":{"draw":31.52,"limit":350},"clocks":{"graphics":210,"memory":405},"fan":{"speed":0}}`
	if got := mustJSON(t, gpus[0]); got != want0 {
		t.Errorf("gpu0\n got %s\nwant %s", got, want0)
	}
	// route variant: no sorting, no known power limit, [N/A] -> null, fan -> 0
	want1 := `{"index":0,"name":"NVIDIA GB10","driverVersion":"580.95.05","temperature":45,` +
		`"utilization":{"gpu":7,"memory":null},"memory":{"total":null,"free":null,"used":null},` +
		`"power":{"draw":11.2,"limit":null},"clocks":{"graphics":2405,"memory":null},"fan":{"speed":0}}`
	if got := mustJSON(t, gpus[1]); got != want1 {
		t.Errorf("gpu1\n got %s\nwant %s", got, want1)
	}
}

func TestParseRouteGPUsShortLine(t *testing.T) {
	// an empty nvidia-smi output still yields one entry upstream ("".split
	// gives [""]), with every field past the first undefined: string fields
	// drop out of the JSON, numbers are NaN -> null
	gpus := ParseRouteGPUs("")
	if len(gpus) != 1 {
		t.Fatalf("got %d gpus", len(gpus))
	}
	want := `{"index":null,"temperature":null,"utilization":{"gpu":null,"memory":null},` +
		`"memory":{"total":null,"free":null,"used":null},"power":{"draw":null,"limit":null},` +
		`"clocks":{"graphics":null,"memory":null},"fan":{"speed":0}}`
	if got := mustJSON(t, gpus[0]); got != want {
		t.Errorf("\n got %s\nwant %s", got, want)
	}
}

func TestParseMonitorGPULine(t *testing.T) {
	if _, ok := ParseMonitorGPULine("garbage line"); ok {
		t.Error("non-numeric index should be dropped")
	}
	g, ok := ParseMonitorGPULine("0, NVIDIA GB10, 580.95.05, 45, 7, [N/A], [N/A], [N/A], [N/A], 11.20, [N/A], 2405, [N/A], [N/A]")
	if !ok {
		t.Fatal("line dropped")
	}
	if g.Power.Limit != 140 {
		t.Errorf("GB10 power limit = %v, want 140", g.Power.Limit)
	}
	cpu := &CPUInfo{TotalMemory: 122880.4, AvailableMemory: 100000.6}
	shared := WithSharedMemory(GPUResponse{HasNvidiaSmi: true, IsMac: boolPtr(false), GPUs: []GPUInfo{g}}, cpu)
	want := `{"total":122880,"free":100001,"used":22879,"shared":true}`
	if got := mustJSON(t, shared.GPUs[0].Memory); got != want {
		t.Errorf("shared memory = %s, want %s", got, want)
	}
	// the original must be left untouched
	if !g.Memory.Total.IsNaN() {
		t.Error("WithSharedMemory mutated its input")
	}
}

func TestGPUResponseShape(t *testing.T) {
	got := mustJSON(t, GPUResponse{HasNvidiaSmi: true, GPUs: []GPUInfo{}})
	if got != `{"hasNvidiaSmi":true,"gpus":[]}` {
		t.Errorf("nvidia success shape has isMac: %s", got)
	}
	got = mustJSON(t, GPUResponse{HasNvidiaSmi: false, IsMac: boolPtr(false), GPUs: []GPUInfo{}, Error: "nvidia-smi not found or not accessible"})
	if got != `{"hasNvidiaSmi":false,"isMac":false,"gpus":[],"error":"nvidia-smi not found or not accessible"}` {
		t.Errorf("not-found shape: %s", got)
	}
}

func TestCPUDisplayName(t *testing.T) {
	cases := map[string]string{
		"Intel(R) Core(TM) i9-12900K CPU @ 3.20GHz": "Intel® Core™ i9-12900K",
		"AMD Ryzen 9 5950X 16-Core Processor      ": "AMD Ryzen 9 5950X 16-Core Processor",
		"Apple M3 Max": "Apple M3 Max",
	}
	for in, want := range cases {
		if got := cpuDisplayName(in); got != want {
			t.Errorf("cpuDisplayName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestLoadAndCPUInfo(t *testing.T) {
	var s LoadSampler
	first := s.Sample()
	if first < 0 || first > 100 {
		t.Errorf("load %v out of range", first)
	}
	info, err := BuildCPUInfo(first, 0)
	if err != nil {
		t.Fatal(err)
	}
	if info.Cores <= 0 || float64(info.TotalMemory) <= 0 || info.AvailableMemory > info.TotalMemory {
		t.Errorf("implausible cpu info: %+v", info)
	}
	t.Logf("cpu: %s", mustJSON(t, info))
}

func TestHistoryPoint(t *testing.T) {
	cpu := &CPUInfo{CurrentLoad: 12.5, TotalMemory: 1000, AvailableMemory: 400}
	g, _ := ParseMonitorGPULine("0, X, 1, 40, 55, 3, 100, 60, 40, 10, 20, 1, 2, 30")
	p := historyPointFromSample(Sample{T: 5, CPU: cpu, GPU: GPUResponse{GPUs: []GPUInfo{g}}})
	want := `{"t":5,"cpu":{"load":12.5,"memUsedMb":600},"gpus":[{"load":55,"memUsedMb":40}]}`
	if got := mustJSON(t, p); got != want {
		t.Errorf("\n got %s\nwant %s", got, want)
	}
	empty := historyPointFromSample(Sample{T: 1, GPU: GPUResponse{GPUs: []GPUInfo{}}})
	if got := mustJSON(t, empty); got != `{"t":1,"cpu":{"load":0,"memUsedMb":0},"gpus":[]}` {
		t.Errorf("no-cpu point: %s", got)
	}
}

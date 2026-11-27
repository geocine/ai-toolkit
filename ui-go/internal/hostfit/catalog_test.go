package hostfit

import "testing"

func TestCatalogAliases(t *testing.T) {
	for alias, row := range archAliases {
		_, vram := vramTable[row]
		_, files := modelFiles[row]
		if !vram && !files {
			t.Errorf("alias %s -> %s: no table row", alias, row)
		}
		if _, ok := catalogEntry(alias); !ok {
			t.Errorf("alias %s is not an upstream arch", alias)
		}
	}
	// an alias shares its row's verdicts; a tag that is not an alias does not
	for _, gb := range TierSizes {
		a, b := VRAMFor(ModelRef{Arch: "wan22_14b:t2v"}, gb), VRAMFor(ModelRef{Arch: "wan22_14b"}, gb)
		if a.Verdict != b.Verdict || a.Source != b.Source || a.Recipe != b.Recipe {
			t.Errorf("%v GB: alias %+v, row %+v", gb, a, b)
		}
	}
	if KnownArch("wan21:1b") {
		t.Error("wan21:1b must not inherit the 14B row")
	}
	if got := ArchName("fibo2"); got != "Fibo 2.0 Dev" {
		t.Errorf("ArchName(fibo2) = %q", got)
	}
	if got := ArchName("wan21"); got != "Wan 2.1 14B" {
		t.Errorf("ArchName(wan21) = %q", got)
	}
}

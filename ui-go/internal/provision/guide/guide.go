// Package guide is the single source of the manual template instructions
// (GET /v1/providers/{kind}/setup, studio/src/lib/guides.json, docs/providers.md).
package guide

import "github.com/geocine/aitoolkit/internal/provision"

type Options struct {
	Image      string // default provision.DefaultImage
	DiskGB     int    // default provision.DefaultDiskGB
	CORSOrigin string // "" -> placeholder "<your studio origin>"
	Token      string // "" -> placeholder "<a long random token>"
}

func For(k provision.Kind, o Options) (provision.SetupGuide, error) {
	return provision.SetupGuide{Kind: k, Port: provision.Port}, nil
}

package guide

import (
	"strconv"
	"strings"

	"github.com/geocine/aitoolkit/internal/provision"
)

// Markdown renders g as the block docs/providers.md carries between
// <!-- guide:<kind> --> and <!-- /guide:<kind> --> (guide_test -update
// rewrites it).
func Markdown(g provision.SetupGuide) string {
	var b strings.Builder
	b.WriteString("| Field | Value | Note |\n|---|---|---|\n")
	for _, f := range g.Fields {
		b.WriteString("| " + cell(f.Label) + " | `" + cell(f.Value) + "` | " + cell(text(f.Note)) + " |\n")
	}
	if g.EnvBlock != "" {
		b.WriteString("\nEnvironment variables:\n\n```\n" + g.EnvBlock + "\n```\n")
	}
	if g.DockerOptions != "" {
		b.WriteString("\nDocker options:\n\n```\n" + g.DockerOptions + "\n```\n")
	}
	b.WriteString("\n")
	for i, s := range g.Steps {
		b.WriteString(strconv.Itoa(i+1) + ". " + text(s) + "\n")
	}
	b.WriteString("\nURL: `" + g.URLPattern + "`\n")
	return b.String()
}

// text escapes the characters Markdown would read as HTML in a sentence
// ("<pod id>", "<a long random token>").
func text(s string) string { return strings.ReplaceAll(s, "<", `\<`) }

func cell(s string) string { return strings.ReplaceAll(s, "|", `\|`) }

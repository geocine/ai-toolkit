package imagecheck

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
)

// HostChecks, when set (by hostfit), judges an image against host facts for
// the CLI's --host-json. imagecheck cannot import hostfit, which imports it.
var HostChecks func(img *Image, hostJSON []byte) ([]Check, error)

// CLI exit codes.
const (
	exitOK     = 0
	exitBlock  = 1 // a block, or the check itself failed
	exitDenied = 2 // the registry refused an anonymous pull
)

// RunCLI is `aitk-server image-check [flags] <image>`: it reads an image's
// registry facts and prints them with the checks. Exit 0 when nothing
// blocks, 1 on a block or an error, 2 when anonymous pulls are denied.
// build_and_push.sh runs it with --require-go after a push.
func RunCLI(args []string) int {
	if os.Getenv("AITK_PROVISION_SIMULATE") == "1" && os.Getenv("AITK_PROVISION_SIMULATE_REGISTRY") != "real" {
		Init(true)
	} else if Default == nil {
		Init(false)
	}
	return runCLI(context.Background(), Default, args, os.Stdout, os.Stderr)
}

func runCLI(ctx context.Context, c *Client, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("image-check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	requireGo := fs.Bool("require-go", false, "fail unless it is the ai-toolkit Go server image with an sm label")
	platform := fs.String("platform", defaultPlatform, "platform to inspect")
	hostJSON := fs.String("host-json", "", "file with host facts (HostFacts JSON) to check the image against")
	asJSON := fs.Bool("json", false, "print the result as JSON")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: aitk-server image-check [flags] <image>\n\n"+
			"Reads an image's facts from its registry with an anonymous pull (no download)\n"+
			"and checks it can run as a GPU server. Exit 0 ok, 1 block or error, 2 denied.\n\nflags:\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return exitOK
		}
		return exitBlock
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return exitBlock
	}
	ref := fs.Arg(0)

	img, err := c.Inspect(ctx, ref, *platform, true)
	if err != nil {
		check := ErrorCheck(err)
		if *asJSON {
			writeJSON(stdout, map[string]any{"ok": false, "level": check.Level, "image": nil, "checks": []Check{check}})
		} else {
			fmt.Fprintf(stderr, "%s: %s\n", ref, check.Message)
			if check.Fix != "" {
				fmt.Fprintf(stderr, "  %s\n", check.Fix)
			}
			if r, perr := ParseRef(ref); perr == nil && r.Name() == DefaultRepo && check.Code == "image_denied" {
				fmt.Fprintf(stderr, "  Make the package public at %s\n", SettingsURL)
			}
			fmt.Fprintf(stderr, "  (%v)\n", err)
		}
		if KindOf(err) == KindDenied {
			return exitDenied
		}
		return exitBlock
	}

	checks := ImageChecks(img)
	if *requireGo && img.GoServer && img.Requires.Sources["sm"] != "label" {
		checks = append(checks, Check{Level: LevelBlock, Code: "image_labels",
			Message: "This image has no sm label, so hosts can't be matched to it.",
			Fix:     "Build it with ui-go/scripts/build_and_push.sh, which writes the labels."})
	}
	if *hostJSON != "" {
		hc, err := hostCLIChecks(img, *hostJSON)
		if err != nil {
			fmt.Fprintf(stderr, "image-check: %v\n", err)
			return exitBlock
		}
		checks = append(checks, hc...)
	}
	level := Worst(checks)
	blocked := level == LevelBlock

	if *asJSON {
		writeJSON(stdout, map[string]any{"ok": !blocked, "level": level, "image": img, "checks": checks})
	} else {
		printFacts(stdout, img, checks)
	}
	if blocked {
		return exitBlock
	}
	return exitOK
}

func hostCLIChecks(img *Image, path string) ([]Check, error) {
	if HostChecks == nil {
		return nil, errors.New("--host-json needs the host checks, which this build does not include")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return HostChecks(img, b)
}

func writeJSON(w io.Writer, v any) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

// printFacts is the short human table: what the image is, what it needs,
// then each check.
func printFacts(w io.Writer, img *Image, checks []Check) {
	req := img.Requires
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	row := func(k, v string) {
		if v != "" {
			fmt.Fprintf(tw, "%s\t%s\n", k, v)
		}
	}
	row("Image", img.Ref)
	row("Digest", img.Digest)
	row("Platform", img.Platform+" (built for "+strings.Join(img.Platforms, ", ")+")")
	row("Download", fmt.Sprintf("%s in %d layers (largest %s)", gb(img.DownloadBytes), img.Layers, gb(img.LargestLayer)))
	row("Unpacked", fmt.Sprintf("%s (%s)", gb(img.UnpackedBytes), img.UnpackedSource))
	if img.GoServer {
		row("Server", fmt.Sprintf("ai-toolkit Go server, API %d, port %d", req.API, req.Port))
	} else {
		row("Server", "not the ai-toolkit Go server")
	}
	if img.GoServer {
		row("Build", strings.TrimSpace(img.Revision+" "+img.Labels[LabelVersion]))
	}
	if req.CUDAMin != "" {
		row("Needs", fmt.Sprintf("CUDA %s (driver %d or newer)", req.CUDAMin, req.DriverMin))
	}
	row("GPUs", smList(req.SM)+sourceNote(req.Sources["sm"]))
	row("Flash attn", smList(req.FlashAttnSM))
	row("Torch", strings.TrimSpace(req.Torch+" "+req.Flavor))
	_ = tw.Flush()
	if len(checks) == 0 {
		fmt.Fprintln(w, "\nNo problems found.")
		return
	}
	fmt.Fprintln(w)
	for _, c := range checks {
		fmt.Fprintf(w, "%-5s  %s: %s\n", c.Level, c.Code, c.Message)
		if c.Fix != "" {
			fmt.Fprintf(w, "       %s\n", c.Fix)
		}
	}
}

func smList(sm []int) string {
	parts := make([]string, len(sm))
	for i, s := range sm {
		parts[i] = "sm_" + strconv.Itoa(s)
	}
	return strings.Join(parts, " ")
}

func sourceNote(src string) string {
	if src == "" || src == "label" {
		return ""
	}
	return " (from the " + src + ")"
}

func gb(b int64) string { return fmt.Sprintf("%.1f GB", float64(b)/1e9) }

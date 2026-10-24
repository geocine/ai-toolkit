package imagecheck

import (
	"fmt"
	"strings"
)

// Machine problems a classified failure leads to.
const (
	ProblemImagePull = "image_pull_failed"
	ProblemCUDA      = "cuda_failed"
)

// Failure is a named cause found in provider or trainer text after launch,
// where the pre-launch checks cannot reach (IMAGE-PLAN §4.6).
type Failure struct {
	Code    string // image_not_found | image_denied | image_platform | disk_full | driver_too_old | no_kernel_image
	Problem string // ProblemImagePull or ProblemCUDA: the Machine.problem it sets
	Message string // the copy for Machine.detail
}

// failurePatterns are checked in order; each matches when all of its
// lowercase parts appear in the text. Platform comes first because its
// text also says "manifest". Docker's "no matching manifest for <p>" names
// the host's platform: linux/amd64 there means the image lacks amd64, any
// other platform means the host isn't amd64.
var failurePatterns = []struct {
	any     [][]string
	failure Failure
}{
	{[][]string{{"no matching manifest for linux/amd64"}},
		Failure{"image_platform", ProblemImagePull, "The image isn't built for amd64."}},
	{[][]string{{"no matching manifest for"}, {"exec format error"}},
		Failure{"image_platform", ProblemImagePull, "The host isn't amd64."}},
	{[][]string{{"manifest unknown"}, {"manifest", "not found"}},
		Failure{"image_not_found", ProblemImagePull, "The provider couldn't find the image tag."}},
	{[][]string{{"unauthorized"}, {"denied"}, {"authentication required"}, {"may require 'docker login'"}},
		Failure{"image_denied", ProblemImagePull, "The provider couldn't pull the image: it's private or missing."}},
	{[][]string{{"no space left on device"}},
		Failure{"disk_full", ProblemImagePull, "The disk filled up while pulling. Use a larger disk."}},
	{[][]string{{"unsatisfied condition: cuda>="}, {"driver version is insufficient"}, {"nvidia driver on your system is too old"}},
		Failure{"driver_too_old", ProblemCUDA, "The host's driver is too old for this image."}},
	{[][]string{{"no kernel image is available"}, {"not compatible with the current pytorch installation"}},
		Failure{"no_kernel_image", ProblemCUDA, "This GPU isn't supported by the image's PyTorch build."}},
}

// ClassifyFailure maps provider status text, logs or trainer stderr to a
// named failure. ok is false when nothing known matches.
func ClassifyFailure(text string) (Failure, bool) {
	t := strings.ToLower(text)
	for _, p := range failurePatterns {
		for _, parts := range p.any {
			all := true
			for _, s := range parts {
				if !strings.Contains(t, s) {
					all = false
					break
				}
			}
			if all {
				return p.failure, true
			}
		}
	}
	return Failure{}, false
}

// WithDisk names the disk size to ask for in a disk_full message.
func (f Failure) WithDisk(gb int) Failure {
	if f.Code == "disk_full" && gb > 0 {
		f.Message = fmt.Sprintf("The disk filled up while pulling. Use at least %d GB.", gb)
	}
	return f
}

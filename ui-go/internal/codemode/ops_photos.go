package codemode

// Photo checks (handlers/v1/photos*.go, docs/headless.md "Photo checks"). The
// types are PhotoTypesTS (photo_types.go); UploadResult there is also the
// result of datasets.uploadFiles.

func init() {
	RegisterTypes(PhotoTypesTS)
	Ops = append(Ops, photoOps...)
}

const photoResolutionTS = "number | number[]"

var photoOps = []Op{
	{Group: "photos", Name: "assess", Method: "POST", Path: "/v1/photos/assess", Result: "PhotoReview",
		Doc: "Judge photo metrics measured elsewhere (the studio measures in the browser before upload) with the same rules as checkPhotos. With dataset, stored photos join the duplicate check. items: [] returns the formats the server can convert.",
		Params: []Param{p("items", "PhotoMetrics[]", "body", "at most 2000"), opt("kind", "PhotoKind", "body", "default person"),
			opt("resolution", photoResolutionTS, "body", "the run's resolution(s), default 1024"), opt("dataset", "string", "body", "a dataset whose stored photos join the duplicate check"),
			opt("pick", "number", "body", "also pick the best n"), opt("decisions", "Record<string, PhotoDecision>", "body", "by file"),
			opt("captions", "Record<string, string>", "body", "caption text by image file")}},
	{Group: "datasets", Name: "checkPhotos", Method: "GET", Path: "/v1/datasets/{name}/photos/check", Result: "PhotoReview",
		Doc: "Every photo with what is wrong with it for training: duplicate, similar, burst, tiny, small (trains below the run's resolution; the trainer never upscales), blurry, dark, bright, screenshot, format traps (unsupported, gray16, transparent, same_stem), caption mentions; after checkFaces also no_face, others, small_face, not_them, identity_check, watermark. Plus duplicate groups, dataset findings (each may carry a ready-to-send fixPhotos op), an A-F grade, coverage and what to add. pending > 0 means analysis is still running: call again. Use before training; fixPhotos acts on it; aitk_view lets you look at a photo yourself.",
		Params: []Param{p("name", "string", "path", ""), opt("kind", "PhotoKind", "query", "default: the stored kind, then person"),
			opt("resolution", photoResolutionTS, "query", "the run's resolution(s), e.g. [512, 768, 1024]; default 1024"),
			opt("pick", "number", "query", "also pick the best n (pick in the review)"), opt("wait_ms", "number", "query", "wait for analysis up to this long (default 3000, max 20000)")}},
	{Group: "datasets", Name: "fixPhotos", Method: "POST", Path: "/v1/datasets/{name}/photos/fix", Result: "PhotoFix",
		Doc: "Ops run in order: set_aside {files} / restore {files | all} (moved to .set_aside/, never deleted), set_aside_flagged {flags?} (default duplicate, tiny, unreadable), keep_best {n?} (the best n for quality and variety; the rest set aside), bake_orientation, convert (bmp/gif/tiff/heic when supported/gray16/transparent/cmyk/huge to JPEG or PNG), rename_same_stem, face_crops {max?}, remove_face_crops. Preview unless apply: true; 409 while a training or caption job uses the dataset.",
		Params: []Param{p("name", "string", "path", ""), p("ops", "PhotoFixOp[]", "body", ""), opt("files", "string[]", "body", "only these files"),
			opt("apply", "boolean", "body", "write the changes (default false: preview)"), opt("kind", "PhotoKind", "body", "for the review returned"),
			opt("resolution", photoResolutionTS, "body", "the run's resolution(s), for keep_best and the review")}},
	{Group: "datasets", Name: "photoSettings", Method: "PUT", Path: "/v1/datasets/{name}/photos/settings", Result: "PhotoReview",
		Doc: "Subject kind (person, style, object), the reference photo (\"this is them\", by file; null clears) and per-photo decisions by file: keep (keep anyway) or intended (no_face, sunglasses, black and white are on purpose); null clears. Returns the new review.",
		Params: []Param{p("name", "string", "path", ""), opt("kind", "PhotoKind", "body", ""), opt("reference", "string | null", "body", "file of a photo of the subject"),
			opt("decisions", "Record<string, PhotoDecision | null>", "body", "by file")}},
	{Group: "datasets", Name: "forgetPhotos", Method: "DELETE", Path: "/v1/datasets/{name}/photos/analysis", Result: "null",
		Doc:    "Forget this dataset's photo analysis: scores, face results, face-to-face similarities, reference and decisions. Photos are untouched.",
		Params: []Param{p("name", "string", "path", "")}},
}

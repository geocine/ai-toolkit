package v1

// The photo checks (photos_routes.go) in the OpenAPI document. Their shapes
// (PhotoReview, PhotoFix, the request bodies) are in docs/headless.md,
// Datasets, "Photo checks".
func init() {
	ops = append(ops,
		op{"POST", "/v1/photos/assess", "datasets", "Judge photo metrics measured elsewhere with the photos/check rules (items: [] returns the formats the server converts)", nil, "object", 200, "object"},
		op{"GET", "/v1/datasets/{name}/photos/check", "datasets", "Every photo with its flags, duplicate groups, findings, grade, coverage and hints (pending > 0: poll again)", []string{"kind:string", "resolution:string", "pick:integer", "wait_ms:integer"}, "", 200, "object"},
		op{"POST", "/v1/datasets/{name}/photos/fix", "datasets", "Photo fixes (set aside, restore, keep the best N, convert, ...); preview unless apply, 409 while a training or caption job uses the dataset", nil, "object", 200, "object"},
		op{"PUT", "/v1/datasets/{name}/photos/settings", "datasets", "Subject kind, the reference photo and per-photo decisions; returns the new review", nil, "object", 200, "object"},
		op{"DELETE", "/v1/datasets/{name}/photos/analysis", "datasets", "Forget the dataset's photo analysis (the photos stay)", nil, "", 204, ""},
		op{"POST", "/v1/datasets/{name}/photos/faces", "datasets", "Check faces: queues a photo_faces job on the cpu queue (body {text?}); 409 while one runs for the dataset", nil, "object", 201, "Job"},
	)
}

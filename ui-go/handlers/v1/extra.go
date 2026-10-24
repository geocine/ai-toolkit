package v1

import "net/http"

// extraRoutes lets feature files in this package (photos_routes.go,
// system_cuda.go) register their routes from init() without editing register.go.
var extraRoutes []func(*http.ServeMux)

func addRoutes(fn func(*http.ServeMux)) { extraRoutes = append(extraRoutes, fn) }

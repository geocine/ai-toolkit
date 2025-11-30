package auth

import (
	"encoding/json"
	"net/http"
)

func RouteHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	json.NewEncoder(w).Encode(map[string]bool{"isAuthenticated": true})
}

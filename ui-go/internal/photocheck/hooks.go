package photocheck

// FacesAvailable reports whether this server can run the face check (M2).
// faces.go sets it from its init; until then reviews say faces.available
// false and the studio hides the face UI.
var FacesAvailable = false

package photocheck

import (
	"regexp"
	"slices"
	"strings"
	"time"
)

// Caption words (section 4.4). Captions are lower-cased with the trigger
// removed; every match is a whole word.
var (
	reCovered = regexp.MustCompile(`\b(sunglasses|shades|face mask|surgical mask|balaclava|hooded|hood|hat|cap|beanie|helmet|bandana|veil)\b`)
	rePeople  = regexp.MustCompile(`\b((?:two|three|several) (?:people|women|men|girls|boys|friends)|group of|crowd|couple|` +
		`with (?:a|her|his|their) (?:friend|boyfriend|girlfriend|husband|wife|man|woman|child|baby))\b`)
	reText   = regexp.MustCompile(`\b(watermark|logo|text reading|caption reading|lettering|subtitles|subtitle|written)\b`)
	reTone   = regexp.MustCompile(`\b(black and white|black-and-white|monochrome|sepia|grayscale|greyscale)\b`)
	reScreen = regexp.MustCompile(`\b(screenshot|user interface|app screen|phone screen|status bar)\b`)

	reOutfit = regexp.MustCompile(`\b(black|white|grey|gray|red|blue|navy|green|yellow|orange|pink|purple|brown|beige|cream|tan|gold|silver|maroon|teal|olive|khaki)` +
		` (?:(denim|leather|wool|knit|silk|satin|linen|cotton|lace|velvet|floral|striped|plaid|checked) )?` +
		`(tank top|t-shirt|tee|shirt|blouse|top|sweater|jumper|hoodie|cardigan|jacket|coat|blazer|suit|dress|gown|skirt|jeans|trousers|pants|shorts|vest|uniform|kimono|saree|scarf)\b`)
	// The adjective is lazy, so "in a coffee shop" keeps "coffee shop"
	// rather than reading "coffee" as an adjective of "shop".
	rePlace = regexp.MustCompile(`\b(?:in|at|on|inside) (?:(?:a|an|the|her|his|their) )?(?:[a-z'-]+ )??` +
		`(living room|coffee shop|kitchen|bedroom|bathroom|office|classroom|studio|gym|cafe|restaurant|bar|car|street|park|beach|garden|forest|mountain|lake|field|city|hallway|stairs|balcony|rooftop|library|shop|store|mall|airport|train|bus|pool)\b`)
	reBackdrop = regexp.MustCompile(`\b(plain backdrop|white background|grey background|gray background|studio backdrop)\b`)
	reLight    = map[string]*regexp.Regexp{
		"day":     regexp.MustCompile(`\b(daylight|harsh sunlight|sunlight|overcast|window light|soft light)\b`),
		"evening": regexp.MustCompile(`\b(golden hour|sunset|sunrise|candlelight|lamp light|dim light)\b`),
		"night":   regexp.MustCompile(`\b(nighttime|night|neon)\b`),
		"studio":  regexp.MustCompile(`\b(studio lighting|flash)\b`),
	}
	reSideBack = regexp.MustCompile(`\b(side|back|rear|behind|profile)\b`)
	reSpace    = regexp.MustCompile(`\s+`)
)

// captionInfo is what one caption says.
type captionInfo struct {
	covered, people, text, tone, screen string // the first matching word or phrase
	outfits, places, light              []string
	sideBack                            bool
}

// keys are the outfit and place keys, for the pick's tag similarity.
func (c *captionInfo) keys() []string {
	if c == nil {
		return nil
	}
	out := make([]string, 0, len(c.outfits)+len(c.places))
	for _, o := range c.outfits {
		out = append(out, "outfit:"+o)
	}
	for _, p := range c.places {
		out = append(out, "place:"+p)
	}
	return out
}

// cleanCaption lower-cases a caption, removes the trigger (and the
// "[trigger]" placeholder) as whole words and collapses white space.
func cleanCaption(text, trigger string) string {
	s := strings.ToLower(text)
	s = strings.ReplaceAll(s, "[trigger]", " ")
	if t := strings.ToLower(strings.TrimSpace(trigger)); t != "" {
		re, err := regexp.Compile(`(^|[^a-z0-9_])` + regexp.QuoteMeta(t) + `($|[^a-z0-9_])`)
		if err == nil {
			// twice, so adjacent repeats ("sks sks") both go
			s = re.ReplaceAllString(re.ReplaceAllString(s, "$1 $2"), "$1 $2")
		}
	}
	return strings.TrimSpace(reSpace.ReplaceAllString(s, " "))
}

// parseCaption reads the caption words and the variety keys.
func parseCaption(text, trigger string) *captionInfo {
	s := cleanCaption(text, trigger)
	c := &captionInfo{
		covered: reCovered.FindString(s),
		people:  rePeople.FindString(s),
		text:    reText.FindString(s),
		tone:    reTone.FindString(s),
		screen:  reScreen.FindString(s),
	}
	for _, m := range reOutfit.FindAllStringSubmatch(s, -1) {
		colour, material, garment := m[1], m[2], m[3]
		if colour == "gray" {
			colour = "grey"
		}
		if garment == "tee" {
			garment = "t-shirt"
		}
		key := colour + " " + garment
		if material != "" {
			key = colour + " " + material + " " + garment
		}
		c.outfits = appendNew(c.outfits, key)
	}
	for _, m := range rePlace.FindAllStringSubmatch(s, -1) {
		c.places = appendNew(c.places, m[1])
	}
	if reBackdrop.MatchString(s) {
		c.places = appendNew(c.places, "backdrop")
	}
	for _, b := range []string{"day", "evening", "night", "studio"} {
		if reLight[b].MatchString(s) {
			c.light = append(c.light, b)
		}
	}
	c.sideBack = reSideBack.MatchString(s)
	return c
}

func appendNew(s []string, v string) []string {
	if slices.Contains(s, v) {
		return s
	}
	return append(s, v)
}

// captions sets the caption flags (all info).
func (j *judge) captions() {
	for _, p := range j.own {
		if p.Pending || p.Caption == nil {
			continue
		}
		c := parseCaption(*p.Caption, j.opts.Trigger)
		p.cap = c
		if c.covered != "" {
			p.set(FlagMentionsCovered, LevelInfo, "The caption mentions "+c.covered+".")
		}
		if c.people != "" {
			p.set(FlagMentionsPeople, LevelInfo, "The caption mentions other people (\""+c.people+"\").")
		}
		if c.text != "" {
			p.set(FlagMentionsText, LevelInfo, "The caption mentions "+c.text+".")
		}
		if c.tone != "" {
			p.set(FlagMentionsTone, LevelInfo, "The caption says "+c.tone+".")
		}
		if c.screen != "" {
			p.set(FlagMentionsScreen, LevelInfo, "The caption mentions "+c.screen+".")
		}
	}
}

// parseExifTime reads ExifInfo.Time ("2006-01-02T15:04:05" plus optional
// fractional seconds), camera-local, as UTC.
func parseExifTime(s string) (time.Time, bool) {
	if s == "" {
		return time.Time{}, false
	}
	t, err := time.Parse("2006-01-02T15:04:05", s)
	if err != nil || t.Year() < 1900 {
		return time.Time{}, false
	}
	return t, true
}

// parseTimes reads every photo's capture time.
func (j *judge) parseTimes() {
	for _, p := range j.all {
		if p.Pending || p.Metrics.Exif == nil {
			continue
		}
		p.when, p.dated = parseExifTime(p.Metrics.Exif.Time)
	}
}

// counted are the photos coverage counts: new, measured, and not a copy
// of another photo.
func (j *judge) counted() []*photo {
	var out []*photo
	for _, p := range j.own {
		if !p.Pending && p.dupOf == "" {
			out = append(out, p)
		}
	}
	return out
}

// sessions numbers the dated photos' sessions (1-based, in time order): a
// session ends at a gap over SessionGapMin.
func (j *judge) sessions() {
	var dated []*photo
	for _, p := range j.counted() {
		if p.dated {
			dated = append(dated, p)
		}
	}
	slices.SortStableFunc(dated, func(a, b *photo) int { return a.when.Compare(b.when) })
	gap := time.Duration(j.th.SessionGapMin * float64(time.Minute))
	n := 0
	for i, p := range dated {
		if i == 0 || p.when.Sub(dated[i-1].when) > gap {
			n++
		}
		p.session = n
	}
}

// coverage is what the set covers, on the axes with data.
func (j *judge) coverage() Coverage {
	c := Coverage{Basis: []string{}}
	var pixels, exif, captions bool
	sizes := map[int]int{}
	var first, last time.Time
	for _, p := range j.counted() {
		pixels = pixels || p.Metrics.Decoded
		if p.dated {
			exif = true
			c.Dated++
			sizes[p.session]++
			if first.IsZero() || p.when.Before(first) {
				first = p.when
			}
			if p.when.After(last) {
				last = p.when
			}
		}
		if p.cap != nil {
			captions = true
			for _, o := range p.cap.outfits {
				c.Outfits = inc(c.Outfits, o)
			}
			for _, pl := range p.cap.places {
				c.Places = inc(c.Places, pl)
			}
			for _, l := range p.cap.light {
				c.Light = inc(c.Light, l)
			}
		}
		if f := p.fv; f != nil && f.checked {
			if f.shot != "" {
				c.Shot = inc(c.Shot, f.shot)
			}
			if f.yaw != "" {
				c.Yaw = inc(c.Yaw, f.yaw)
			}
		}
	}
	c.Sessions = len(sizes)
	for _, n := range sizes {
		c.LargestSession = max(c.LargestSession, n)
	}
	if c.Dated > 0 {
		c.SpanDays = int(last.Sub(first).Hours() / 24)
	}
	if pixels {
		c.Basis = append(c.Basis, "pixels")
	}
	if exif {
		c.Basis = append(c.Basis, "exif")
	}
	if captions {
		c.Basis = append(c.Basis, "captions")
	}
	if j.checked > 0 {
		c.Basis = append(c.Basis, "faces")
	}
	return c
}

func inc(m map[string]int, k string) map[string]int {
	if m == nil {
		m = map[string]int{}
	}
	m[k]++
	return m
}

// topKey is the most frequent key (ties: alphabetical) and its count.
func topKey(m map[string]int) (string, int) {
	best, n := "", 0
	for k, v := range m {
		if v > n || (v == n && k < best) {
			best, n = k, v
		}
	}
	return best, n
}

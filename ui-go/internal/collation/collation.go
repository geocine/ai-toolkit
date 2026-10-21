// Package collation orders strings like JavaScript's String.prototype.localeCompare
// under Node's ICU (CLDR root collation), which upstream uses to sort file lists.
package collation

import (
	"sync"

	"golang.org/x/text/collate"
	"golang.org/x/text/language"
)

// a Collator is not safe for concurrent use
var pool = sync.Pool{New: func() any { return collate.New(language.Und) }}

// Compare is a.localeCompare(b): negative, zero or positive.
func Compare(a, b string) int {
	c := pool.Get().(*collate.Collator)
	defer pool.Put(c)
	return c.CompareString(a, b)
}

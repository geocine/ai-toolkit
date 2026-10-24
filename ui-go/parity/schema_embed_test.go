package parity

import (
	"database/sql"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/geocine/aitoolkit/internal/store"
	"github.com/geocine/aitoolkit/internal/testutil"
	_ "modernc.org/sqlite"
)

// TestEmbeddedPrismaSchema fails when upstream changed its schema and the
// copy the studio API creates databases from (internal/store/schema/) did not.
func TestEmbeddedPrismaSchema(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(upstreamUI, "prisma", "schema.prisma"))
	if err != nil {
		t.Skipf("upstream schema not found: %v", err)
	}
	lf := func(s string) string { return strings.ReplaceAll(s, "\r\n", "\n") }
	if lf(store.PrismaSchema) != lf(string(b)) {
		t.Fatal("upstream changed ui/prisma/schema.prisma: copy it to internal/store/schema/schema.prisma and update schema.sql to match prisma db push")
	}
}

// prismaField is one field line of a model block. A relation field (a list,
// another model's type, or @relation) has no column.
type prismaField struct {
	name, typ                     string
	optional, unique, id          bool
	dbDefault, relation, sequence bool
}

type prismaModel struct {
	fields           []prismaField
	uniques, indexes [][]string
}

// clientDefaults are @default values Prisma fills in itself (or SQLite's
// rowid does), so the column carries no DEFAULT.
var clientDefaults = []string{"autoincrement()", "uuid()", "cuid()", "nanoid()", "ulid()"}

// prismaSchema parses the model blocks of a Prisma schema, attributes included.
func prismaSchema(src string) map[string]prismaModel {
	type raw struct {
		model string
		line  string
	}
	var lines []raw
	names := map[string]bool{}
	current := ""
	for _, l := range strings.Split(src, "\n") {
		l = strings.TrimSpace(stripComment(l))
		switch {
		case l == "":
		case strings.HasPrefix(l, "model ") && strings.HasSuffix(l, "{"):
			current = strings.Fields(l)[1]
			names[current] = true
		case l == "}":
			current = ""
		case current != "":
			lines = append(lines, raw{current, l})
		}
	}
	models := map[string]prismaModel{}
	for name := range names {
		models[name] = prismaModel{}
	}
	for _, r := range lines {
		m := models[r.model]
		switch {
		case strings.HasPrefix(r.line, "@@unique"):
			m.uniques = append(m.uniques, attrColumns(r.line))
		case strings.HasPrefix(r.line, "@@index"):
			m.indexes = append(m.indexes, attrColumns(r.line))
		case strings.HasPrefix(r.line, "@@"):
		default:
			parts := strings.Fields(r.line)
			if len(parts) < 2 {
				continue
			}
			f := prismaField{name: parts[0], typ: parts[1]}
			attrs := strings.Join(parts[2:], " ")
			if strings.HasSuffix(f.typ, "[]") {
				f.relation = true
			}
			if strings.HasSuffix(f.typ, "?") {
				f.optional, f.typ = true, strings.TrimSuffix(f.typ, "?")
			}
			if names[strings.TrimSuffix(f.typ, "[]")] || strings.Contains(attrs, "@relation") {
				f.relation = true
			}
			f.unique = hasAttr(attrs, "@unique")
			f.id = hasAttr(attrs, "@id")
			if i := strings.Index(attrs, "@default("); i >= 0 {
				v := balanced(attrs[i+len("@default"):])
				if slices.Contains(clientDefaults, v) {
					f.sequence = true
				} else {
					f.dbDefault = true
				}
			}
			m.fields = append(m.fields, f)
		}
		models[r.model] = m
	}
	return models
}

// stripComment drops a // comment that is not inside a string.
func stripComment(l string) string {
	in := false
	for i := 0; i < len(l); i++ {
		switch {
		case l[i] == '"' && (i == 0 || l[i-1] != '\\'):
			in = !in
		case !in && strings.HasPrefix(l[i:], "//"):
			return l[:i]
		}
	}
	return l
}

// hasAttr: attrs holds the attribute name itself, not a longer one.
func hasAttr(attrs, name string) bool {
	for rest := attrs; ; {
		i := strings.Index(rest, name)
		if i < 0 {
			return false
		}
		rest = rest[i+len(name):]
		if rest == "" || !(rest[0] == '_' || rest[0] >= 'a' && rest[0] <= 'z' || rest[0] >= 'A' && rest[0] <= 'Z') {
			return true
		}
	}
}

// balanced returns what is inside the parentheses s starts with.
func balanced(s string) string {
	depth := 0
	for i, c := range s {
		switch c {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return strings.TrimSpace(s[1:i])
			}
		}
	}
	return ""
}

// attrColumns reads the field list of @@unique([a, b]) or @@index([a], map: "x").
func attrColumns(l string) []string {
	open, end := strings.Index(l, "["), strings.Index(l, "]")
	if open < 0 || end < open {
		return nil
	}
	var cols []string
	for _, c := range strings.Split(l[open+1:end], ",") {
		c = strings.TrimSpace(c)
		if i := strings.Index(c, "("); i >= 0 { // name(sort: Desc)
			c = c[:i]
		}
		if c != "" {
			cols = append(cols, c)
		}
	}
	return cols
}

// sqliteType is the column type prisma db push gives each Prisma scalar.
var sqliteType = map[string]string{"Int": "INTEGER", "String": "TEXT", "Boolean": "BOOLEAN", "DateTime": "DATETIME",
	"Float": "REAL", "BigInt": "BIGINT", "Decimal": "DECIMAL", "Bytes": "BLOB", "Json": "JSONB"}

// TestEmbeddedDDLMatchesPrisma never skips: schema.sql must create, column for
// column and index for index, what the embedded schema.prisma describes (the
// copy TestEmbeddedPrismaSchema ties to upstream).
func TestEmbeddedDDLMatchesPrisma(t *testing.T) {
	models := prismaSchema(store.PrismaSchema)
	if len(models) == 0 {
		t.Fatal("no models parsed from the embedded schema.prisma")
	}
	path := filepath.Join(testutil.TempDir(t), "schema.db")
	db, err := sql.Open("sqlite", store.FileURI(path))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(store.SchemaSQL); err != nil {
		t.Fatalf("schema.sql: %v", err)
	}

	var tables []string
	rows, err := db.Query(`SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'`)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var n string
		rows.Scan(&n)
		tables = append(tables, n)
	}
	rows.Close()
	var want []string
	for name := range models {
		want = append(want, name)
	}
	sort.Strings(tables)
	sort.Strings(want)
	if !slices.Equal(tables, want) {
		t.Fatalf("schema.sql tables %v, schema.prisma models %v", tables, want)
	}

	for name, m := range models {
		type col struct {
			typ     string
			notnull bool
			dflt    sql.NullString
			pk      bool
		}
		cols := map[string]col{}
		rows, err := db.Query(`SELECT name, type, "notnull", dflt_value, pk FROM pragma_table_info(?)`, name)
		if err != nil {
			t.Fatal(err)
		}
		for rows.Next() {
			var n string
			var c col
			var pk int
			if err := rows.Scan(&n, &c.typ, &c.notnull, &c.dflt, &pk); err != nil {
				t.Fatal(err)
			}
			c.pk = pk > 0
			cols[n] = c
		}
		rows.Close()

		scalars := map[string]bool{}
		for _, f := range m.fields {
			if f.relation {
				continue
			}
			scalars[f.name] = true
			c, ok := cols[f.name]
			if !ok {
				t.Errorf("%s.%s: no column in schema.sql", name, f.name)
				continue
			}
			if wt, known := sqliteType[f.typ]; known && !strings.EqualFold(c.typ, wt) {
				t.Errorf("%s.%s: type %s, want %s", name, f.name, c.typ, wt)
			}
			if f.optional == c.notnull {
				t.Errorf("%s.%s: optional %v but NOT NULL %v", name, f.name, f.optional, c.notnull)
			}
			if f.dbDefault && !c.dflt.Valid {
				t.Errorf("%s.%s: @default but no DEFAULT", name, f.name)
			}
			if !f.dbDefault && c.dflt.Valid {
				t.Errorf("%s.%s: DEFAULT %s but no @default", name, f.name, c.dflt.String)
			}
			if f.id != c.pk {
				t.Errorf("%s.%s: @id %v but PRIMARY KEY %v", name, f.name, f.id, c.pk)
			}
			if f.unique {
				m.uniques = append(m.uniques, []string{f.name})
			}
		}
		for n := range cols {
			if !scalars[n] {
				t.Errorf("%s.%s: column in schema.sql, not in schema.prisma", name, n)
			}
		}

		type index struct {
			unique bool
			cols   []string
		}
		var indexes []index
		rows, err = db.Query(`SELECT name, "unique" FROM pragma_index_list(?)`, name)
		if err != nil {
			t.Fatal(err)
		}
		var listed []struct {
			name   string
			unique bool
		}
		for rows.Next() {
			var e struct {
				name   string
				unique bool
			}
			rows.Scan(&e.name, &e.unique)
			listed = append(listed, e)
		}
		rows.Close()
		for _, e := range listed {
			ix := index{unique: e.unique}
			rows, err := db.Query(`SELECT name FROM pragma_index_info(?) ORDER BY seqno`, e.name)
			if err != nil {
				t.Fatal(err)
			}
			for rows.Next() {
				var c string
				rows.Scan(&c)
				ix.cols = append(ix.cols, c)
			}
			rows.Close()
			indexes = append(indexes, ix)
		}
		has := func(cols []string, unique bool) bool {
			for _, ix := range indexes {
				if slices.Equal(ix.cols, cols) && (!unique || ix.unique) {
					return true
				}
			}
			return false
		}
		for _, u := range m.uniques {
			if !has(u, true) {
				t.Errorf("%s: no unique index on %v", name, u)
			}
		}
		for _, ix := range m.indexes {
			if !has(ix, false) {
				t.Errorf("%s: no index on %v", name, ix)
			}
		}
	}
}

// The parser itself, on the shapes upstream's schema does not use yet.
func TestEmbeddedSchemaParser(t *testing.T) {
	m := prismaSchema(`
model User {
  id    String  @id @default(cuid())
  email String  @unique // "quoted // not a comment"
  note  String? @default("a // b")
  posts Post[]
  @@unique([email, note])
}
model Post {
  id     Int  @id @default(autoincrement())
  author User @relation(fields: [userId], references: [id])
  userId String
  at     DateTime @default(now())
  @@index([userId(sort: Desc)], map: "by_user")
}`)
	u, p := m["User"], m["Post"]
	if len(u.fields) != 4 || !u.fields[0].id || !u.fields[0].sequence || !u.fields[1].unique ||
		!u.fields[2].optional || !u.fields[2].dbDefault || !u.fields[3].relation {
		t.Fatalf("User = %+v", u)
	}
	if len(u.uniques) != 1 || !slices.Equal(u.uniques[0], []string{"email", "note"}) {
		t.Fatalf("User uniques = %v", u.uniques)
	}
	if !p.fields[1].relation || p.fields[2].relation || !p.fields[3].dbDefault ||
		len(p.indexes) != 1 || !slices.Equal(p.indexes[0], []string{"userId"}) {
		t.Fatalf("Post = %+v", p)
	}
}

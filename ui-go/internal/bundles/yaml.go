package bundles

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"go.yaml.in/yaml/v3"

	"github.com/geocine/aitoolkit/internal/jsonx"
)

// YAMLObject parses an ai-toolkit config.yaml into the ordered JSON tree the
// rest of the server edits configs as (key order kept, numbers json.Number).
func YAMLObject(data []byte) (*jsonx.Object, error) {
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	if doc.Kind != yaml.DocumentNode || len(doc.Content) == 0 {
		return nil, fmt.Errorf("empty document")
	}
	v, err := fromYAML(doc.Content[0], 0)
	if err != nil {
		return nil, err
	}
	o, ok := v.(*jsonx.Object)
	if !ok {
		return nil, fmt.Errorf("top level is not a mapping")
	}
	return o, nil
}

func fromYAML(n *yaml.Node, depth int) (any, error) {
	if depth > 64 {
		return nil, fmt.Errorf("nested too deep")
	}
	switch n.Kind {
	case yaml.AliasNode:
		return fromYAML(n.Alias, depth+1)
	case yaml.MappingNode:
		o := jsonx.NewObject()
		for i := 0; i+1 < len(n.Content); i += 2 {
			k, v := n.Content[i], n.Content[i+1]
			if k.Tag == "!!merge" {
				merged, err := fromYAML(v, depth+1)
				if err != nil {
					return nil, err
				}
				if mo, ok := merged.(*jsonx.Object); ok {
					for _, mk := range mo.Keys() {
						if _, has := o.Get(mk); !has {
							mv, _ := mo.Get(mk)
							o.Set(mk, mv)
						}
					}
				}
				continue
			}
			val, err := fromYAML(v, depth+1)
			if err != nil {
				return nil, err
			}
			o.Set(k.Value, val)
		}
		return o, nil
	case yaml.SequenceNode:
		out := make([]any, 0, len(n.Content))
		for _, c := range n.Content {
			v, err := fromYAML(c, depth+1)
			if err != nil {
				return nil, err
			}
			out = append(out, v)
		}
		return out, nil
	case yaml.ScalarNode:
		switch n.ShortTag() {
		case "!!null":
			return nil, nil
		case "!!bool":
			var b bool
			if err := n.Decode(&b); err != nil {
				return nil, err
			}
			return b, nil
		case "!!int":
			var i int64
			if err := n.Decode(&i); err == nil {
				return json.Number(strconv.FormatInt(i, 10)), nil
			}
			return n.Value, nil
		case "!!float":
			var f float64
			if err := n.Decode(&f); err == nil && !math.IsInf(f, 0) && !math.IsNaN(f) {
				return json.Number(strconv.FormatFloat(f, 'g', -1, 64)), nil
			}
			return n.Value, nil
		}
		return n.Value, nil
	}
	return nil, fmt.Errorf("unsupported YAML node at line %d", n.Line)
}

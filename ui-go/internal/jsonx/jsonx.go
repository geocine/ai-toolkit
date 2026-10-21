// Package jsonx decodes JSON objects with their key order intact, so configs
// the server patches and writes back read like the JSON.stringify output of
// the Next UI (and numbers keep their original text).
package jsonx

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// Object is a JSON object that remembers key order.
type Object struct {
	keys   []string
	values map[string]any
}

func NewObject() *Object { return &Object{values: map[string]any{}} }

func (o *Object) Get(key string) (any, bool) {
	v, ok := o.values[key]
	return v, ok
}

// Set adds or replaces key, keeping its position when it already exists.
func (o *Object) Set(key string, v any) {
	if _, ok := o.values[key]; !ok {
		o.keys = append(o.keys, key)
	}
	o.values[key] = v
}

func (o *Object) Keys() []string { return append([]string(nil), o.keys...) }

// Object returns the nested object at key, or nil.
func (o *Object) Object(key string) *Object {
	v, _ := o.values[key].(*Object)
	return v
}

// Array returns the nested array at key, or nil.
func (o *Object) Array(key string) []any {
	v, _ := o.values[key].([]any)
	return v
}

func (o *Object) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, k := range o.keys {
		if i > 0 {
			buf.WriteByte(',')
		}
		kb, _ := marshal(k)
		buf.Write(kb)
		buf.WriteByte(':')
		vb, err := marshal(o.values[k])
		if err != nil {
			return nil, err
		}
		buf.Write(vb)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

func marshal(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// Indent renders v like JSON.stringify(v, null, 2).
func Indent(v any) ([]byte, error) {
	raw, err := marshal(v)
	if err != nil {
		return nil, err
	}
	var out bytes.Buffer
	if err := json.Indent(&out, raw, "", "  "); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// Parse decodes data; objects become *Object, numbers json.Number.
func Parse(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("jsonx: trailing data")
	}
	return v, nil
}

// ParseObject decodes data that must be a JSON object.
func ParseObject(data []byte) (*Object, error) {
	v, err := Parse(data)
	if err != nil {
		return nil, err
	}
	o, ok := v.(*Object)
	if !ok {
		return nil, fmt.Errorf("jsonx: not an object")
	}
	return o, nil
}

func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			o := NewObject()
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, _ := kt.(string)
				v, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				o.Set(key, v)
			}
			_, err := dec.Token() // '}'
			return o, err
		case '[':
			arr := []any{}
			for dec.More() {
				v, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				arr = append(arr, v)
			}
			_, err := dec.Token() // ']'
			return arr, err
		}
		return nil, fmt.Errorf("jsonx: unexpected %v", t)
	default:
		return t, nil
	}
}

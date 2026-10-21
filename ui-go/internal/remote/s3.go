package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// s3 is any S3-compatible bucket: Cloudflare R2 (endpoint
// https://<account>.r2.cloudflarestorage.com, region "auto"), AWS S3, MinIO.
// Config {endpoint, bucket, region, prefix, path_style}; secret
// {access_key_id, secret_access_key}.
func init() { RegisterKind("s3", newS3) }

type s3Config struct {
	Endpoint  string `json:"endpoint"`
	Bucket    string `json:"bucket"`
	Region    string `json:"region"`
	Prefix    string `json:"prefix"`
	PathStyle bool   `json:"path_style"`
}

type s3Secret struct {
	AccessKeyID     string `json:"access_key_id"`
	SecretAccessKey string `json:"secret_access_key"`
}

type s3Client struct {
	mc     *minio.Client
	bucket string
	prefix string
	base   string // URL of the bucket, for Put's returned URL
}

// s3PartSize keeps parts well above S3's 5 MiB minimum while bounding memory.
const s3PartSize = 16 << 20

func newS3(config, secret json.RawMessage) (Client, error) {
	var c s3Config
	if err := json.Unmarshal(config, &c); err != nil {
		return nil, fmt.Errorf("s3 remote config: %w", err)
	}
	var s s3Secret
	if len(secret) > 0 {
		if err := json.Unmarshal(secret, &s); err != nil {
			return nil, fmt.Errorf("s3 remote credentials: %w", err)
		}
	}
	if c.Endpoint == "" || c.Bucket == "" {
		return nil, errors.New("s3 remote needs config.endpoint and config.bucket")
	}
	if s.AccessKeyID == "" || s.SecretAccessKey == "" {
		return nil, errors.New("s3 remote needs credentials.access_key_id and credentials.secret_access_key")
	}
	host, secure, err := parseEndpoint(c.Endpoint)
	if err != nil {
		return nil, err
	}
	region := c.Region
	if region == "" && strings.HasSuffix(host, ".r2.cloudflarestorage.com") {
		region = "auto"
	}
	lookup := minio.BucketLookupAuto
	if c.PathStyle {
		lookup = minio.BucketLookupPath
	}
	mc, err := minio.New(host, &minio.Options{
		Creds:        credentials.NewStaticV4(s.AccessKeyID, s.SecretAccessKey, ""),
		Secure:       secure,
		Region:       region,
		BucketLookup: lookup,
	})
	if err != nil {
		return nil, fmt.Errorf("s3 remote: %w", err)
	}
	scheme := "https"
	if !secure {
		scheme = "http"
	}
	return &s3Client{
		mc: mc, bucket: c.Bucket, prefix: strings.Trim(c.Prefix, "/"),
		base: fmt.Sprintf("%s://%s/%s", scheme, host, c.Bucket),
	}, nil
}

// parseEndpoint accepts "host[:port]" or a URL; no scheme means https.
func parseEndpoint(ep string) (host string, secure bool, err error) {
	if !strings.Contains(ep, "://") {
		return strings.TrimRight(ep, "/"), true, nil
	}
	u, err := url.Parse(ep)
	if err != nil {
		return "", false, fmt.Errorf("s3 remote endpoint: %w", err)
	}
	if u.Host == "" {
		return "", false, fmt.Errorf("s3 remote endpoint %q has no host", ep)
	}
	if u.Path != "" && u.Path != "/" {
		return "", false, fmt.Errorf("s3 remote endpoint %q must not contain a path; use config.bucket and config.prefix", ep)
	}
	return u.Host, u.Scheme != "http", nil
}

func (c *s3Client) key(k string) (string, error) {
	clean := path.Clean("/" + strings.ReplaceAll(k, `\`, "/"))
	if clean == "/" {
		return "", fmt.Errorf("invalid key %q", k)
	}
	if c.prefix == "" {
		return clean[1:], nil
	}
	return c.prefix + clean, nil
}

// rel turns a full object name back into a key relative to the prefix.
func (c *s3Client) rel(name string) string {
	if c.prefix == "" {
		return name
	}
	return strings.TrimPrefix(name, c.prefix+"/")
}

func isNotFound(err error) bool {
	code := minio.ToErrorResponse(err).Code
	return code == "NoSuchKey" || code == "NotFound" || code == "NoSuchObject"
}

// progressReader is how minio-go reports upload progress: it reads from it
// as bytes go out.
type progressReader struct {
	n  int64
	fn Progress
}

func (p *progressReader) Read(b []byte) (int, error) {
	p.n += int64(len(b))
	p.fn(p.n)
	return len(b), nil
}

func (c *s3Client) Put(ctx context.Context, key, localPath string, progress Progress) (string, error) {
	k, err := c.key(key)
	if err != nil {
		return "", err
	}
	opts := minio.PutObjectOptions{PartSize: s3PartSize, ContentType: "application/octet-stream"}
	if progress != nil {
		opts.Progress = &progressReader{fn: progress}
	}
	if _, err := c.mc.FPutObject(ctx, c.bucket, k, localPath, opts); err != nil {
		return "", fmt.Errorf("s3 put %s: %w", k, err)
	}
	return c.base + "/" + k, nil
}

func (c *s3Client) Get(ctx context.Context, key, localPath string, progress Progress) error {
	k, err := c.key(key)
	if err != nil {
		return err
	}
	obj, err := c.mc.GetObject(ctx, c.bucket, k, minio.GetObjectOptions{})
	if err != nil {
		return fmt.Errorf("s3 get %s: %w", k, err)
	}
	defer obj.Close()
	// GetObject is lazy; stat surfaces a missing key before any file is made
	if _, err := obj.Stat(); err != nil {
		if isNotFound(err) {
			return fmt.Errorf("%s: %w", key, ErrNotFound)
		}
		return fmt.Errorf("s3 get %s: %w", k, err)
	}
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(localPath), ".part-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	var r io.Reader = obj
	if progress != nil {
		r = &countingReader{r: obj, fn: progress}
	}
	if _, err := io.Copy(tmp, r); err != nil {
		tmp.Close()
		return fmt.Errorf("s3 get %s: %w", k, err)
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), localPath)
}

func (c *s3Client) PutBytes(ctx context.Context, key string, data []byte) error {
	k, err := c.key(key)
	if err != nil {
		return err
	}
	_, err = c.mc.PutObject(ctx, c.bucket, k, bytes.NewReader(data), int64(len(data)),
		minio.PutObjectOptions{ContentType: contentTypeFor(k)})
	if err != nil {
		return fmt.Errorf("s3 put %s: %w", k, err)
	}
	return nil
}

func contentTypeFor(k string) string {
	if strings.HasSuffix(k, ".json") {
		return "application/json"
	}
	return "application/octet-stream"
}

func (c *s3Client) GetBytes(ctx context.Context, key string) ([]byte, error) {
	k, err := c.key(key)
	if err != nil {
		return nil, err
	}
	obj, err := c.mc.GetObject(ctx, c.bucket, k, minio.GetObjectOptions{})
	if err != nil {
		return nil, fmt.Errorf("s3 get %s: %w", k, err)
	}
	defer obj.Close()
	b, err := io.ReadAll(obj)
	if err != nil {
		if isNotFound(err) {
			return nil, fmt.Errorf("%s: %w", key, ErrNotFound)
		}
		return nil, fmt.Errorf("s3 get %s: %w", k, err)
	}
	return b, nil
}

func (c *s3Client) Stat(ctx context.Context, key string) (*Object, error) {
	k, err := c.key(key)
	if err != nil {
		return nil, err
	}
	info, err := c.mc.StatObject(ctx, c.bucket, k, minio.StatObjectOptions{})
	if err != nil {
		if isNotFound(err) {
			return nil, fmt.Errorf("%s: %w", key, ErrNotFound)
		}
		return nil, fmt.Errorf("s3 stat %s: %w", k, err)
	}
	return &Object{Key: key, Size: info.Size, Modified: info.LastModified.UTC()}, nil
}

func (c *s3Client) List(ctx context.Context, prefix string) ([]Object, error) {
	full := strings.TrimLeft(prefix, "/")
	if c.prefix != "" {
		full = c.prefix + "/" + full
	}
	out := []Object{}
	for info := range c.mc.ListObjects(ctx, c.bucket, minio.ListObjectsOptions{Prefix: full, Recursive: true}) {
		if info.Err != nil {
			return nil, fmt.Errorf("s3 list %s: %w", full, info.Err)
		}
		if strings.HasSuffix(info.Key, "/") {
			continue
		}
		out = append(out, Object{Key: c.rel(info.Key), Size: info.Size, Modified: info.LastModified.UTC()})
	}
	// S3 lists in UTF-8 binary order already; keys relative to the prefix keep it
	return out, nil
}

func (c *s3Client) Delete(ctx context.Context, key string) error {
	k, err := c.key(key)
	if err != nil {
		return err
	}
	if err := c.mc.RemoveObject(ctx, c.bucket, k, minio.RemoveObjectOptions{}); err != nil && !isNotFound(err) {
		return fmt.Errorf("s3 delete %s: %w", k, err)
	}
	return nil
}

func (c *s3Client) Check(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	ok, err := c.mc.BucketExists(ctx, c.bucket)
	if err != nil {
		return fmt.Errorf("s3 bucket %s: %w", c.bucket, err)
	}
	if !ok {
		return fmt.Errorf("s3 bucket %s does not exist", c.bucket)
	}
	probe := fmt.Sprintf(".aitk-check-%d", time.Now().UnixNano())
	if err := c.PutBytes(ctx, probe, []byte("ok")); err != nil {
		return err
	}
	return c.Delete(ctx, probe)
}

package provision

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/geocine/aitoolkit/internal/state"
)

// machineSpec is the machines.spec JSON: what a launch asked for, never an
// env value. Its shape is fixed here; later units only set its fields.
type machineSpec struct {
	Offer          Offer           `json:"offer"`
	GPUCount       int             `json:"gpu_count"`
	Cloud          string          `json:"cloud"`
	DiskGB         int             `json:"disk_gb"`
	Image          string          `json:"image"`
	TemplateID     string          `json:"template_id"`
	WhenDone       WhenDone        `json:"when_done"`
	SelfDestroyKey bool            `json:"self_destroy_key"`
	CopiedRemote   string          `json:"copied_remote"`
	EnvKeys        []string        `json:"env_keys"`
	Warnings       []string        `json:"warnings"`
	ImageCheck     json.RawMessage `json:"image_check,omitempty"` // IMG §4.5 step 6 snapshot (I-I5)
	ImageWarnings  []string        `json:"image_warnings"`
	PullS          float64         `json:"pull_s"`
}

// Pending is the action this server asked the provider for and has not seen
// finish yet (the status overlay reads it).
const (
	pendingStop      = "stop"
	pendingStart     = "start"
	pendingTerminate = "terminate"
)

// row is one machines row. Times are unix milliseconds.
type row struct {
	ID             string
	Provider       Kind
	ProviderID     string
	Label          string
	Name           string
	Token          string
	Status         Status
	Problem        Problem
	Detail         string
	ProviderStatus string
	Pending        string
	Spec           machineSpec
	BaseURL        string
	Endpoints      []Endpoint
	ImageLaunchRef string
	ImageDigest    string
	Price          *float64
	StoppedPrice   *float64
	RequestID      *string
	Simulated      bool
	CreatedAt      int64
	UpdatedAt      int64
	RunningAt      *int64
	ReadyAt        *int64
	LastOKAt       *int64
	EndedAt        *int64
}

const rowColumns = `id, provider, provider_id, label, name, token, status, problem, detail, provider_status, pending,
	spec, base_url, endpoints, image_launch_ref, image_digest, price_per_hour, stopped_price_per_hour, request_id,
	simulated, created_at, updated_at, running_at, ready_at, last_ok_at, ended_at`

var errNoRow = errors.New("no such machine")

func scanRow(sc interface{ Scan(...any) error }) (*row, error) {
	var r row
	var provider, status, problem, spec, endpoints string
	var price, stopped sql.NullFloat64
	var reqID sql.NullString
	var simulated int
	var running, ready, lastOK, ended sql.NullInt64
	err := sc.Scan(&r.ID, &provider, &r.ProviderID, &r.Label, &r.Name, &r.Token, &status, &problem, &r.Detail,
		&r.ProviderStatus, &r.Pending, &spec, &r.BaseURL, &endpoints, &r.ImageLaunchRef, &r.ImageDigest,
		&price, &stopped, &reqID, &simulated, &r.CreatedAt, &r.UpdatedAt, &running, &ready, &lastOK, &ended)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errNoRow
	}
	if err != nil {
		return nil, err
	}
	r.Provider, r.Status, r.Problem, r.Simulated = Kind(provider), Status(status), Problem(problem), simulated != 0
	if err := json.Unmarshal([]byte(spec), &r.Spec); err != nil {
		return nil, fmt.Errorf("machine %s spec: %w", r.ID, err)
	}
	if err := json.Unmarshal([]byte(endpoints), &r.Endpoints); err != nil {
		return nil, fmt.Errorf("machine %s endpoints: %w", r.ID, err)
	}
	if price.Valid {
		r.Price = &price.Float64
	}
	if stopped.Valid {
		r.StoppedPrice = &stopped.Float64
	}
	if reqID.Valid {
		r.RequestID = &reqID.String
	}
	r.RunningAt, r.ReadyAt, r.LastOKAt, r.EndedAt = nullMs(running), nullMs(ready), nullMs(lastOK), nullMs(ended)
	return &r, nil
}

func nullMs(v sql.NullInt64) *int64 {
	if !v.Valid {
		return nil
	}
	return &v.Int64
}

// values is the row in rowColumns order.
func (r *row) values() ([]any, error) {
	spec, err := json.Marshal(r.Spec)
	if err != nil {
		return nil, err
	}
	eps := r.Endpoints
	if eps == nil {
		eps = []Endpoint{}
	}
	endpoints, err := json.Marshal(eps)
	if err != nil {
		return nil, err
	}
	sim := 0
	if r.Simulated {
		sim = 1
	}
	return []any{r.ID, string(r.Provider), r.ProviderID, r.Label, r.Name, r.Token, string(r.Status), string(r.Problem),
		r.Detail, r.ProviderStatus, r.Pending, string(spec), r.BaseURL, string(endpoints), r.ImageLaunchRef, r.ImageDigest,
		r.Price, r.StoppedPrice, r.RequestID, sim, r.CreatedAt, r.UpdatedAt, r.RunningAt, r.ReadyAt, r.LastOKAt, r.EndedAt}, nil
}

func insertRow(ctx context.Context, r *row) error {
	vals, err := r.values()
	if err != nil {
		return err
	}
	_, err = state.DB.ExecContext(ctx, `INSERT INTO machines (`+rowColumns+`)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, vals...)
	return err
}

// updateRow writes every column of r except id and created_at.
func updateRow(ctx context.Context, r *row) error {
	vals, err := r.values()
	if err != nil {
		return err
	}
	_, err = state.DB.ExecContext(ctx, `UPDATE machines SET provider = ?, provider_id = ?, label = ?, name = ?, token = ?,
		status = ?, problem = ?, detail = ?, provider_status = ?, pending = ?, spec = ?, base_url = ?, endpoints = ?,
		image_launch_ref = ?, image_digest = ?, price_per_hour = ?, stopped_price_per_hour = ?, request_id = ?,
		simulated = ?, updated_at = ?, running_at = ?, ready_at = ?, last_ok_at = ?, ended_at = ? WHERE id = ?`,
		append(append(vals[1:20:20], vals[21:]...), r.ID)...)
	return err
}

// getRow finds a machine by id, or by the name of a live machine.
func getRow(ctx context.Context, idOrName string) (*row, error) {
	r, err := scanRow(state.DB.QueryRowContext(ctx, `SELECT `+rowColumns+` FROM machines WHERE id = ?`, idOrName))
	if !errors.Is(err, errNoRow) {
		return r, err
	}
	return scanRow(state.DB.QueryRowContext(ctx, `SELECT `+rowColumns+` FROM machines
		WHERE name = ? AND ended_at IS NULL`, idOrName))
}

func rowByRequestID(ctx context.Context, requestID string) (*row, error) {
	return scanRow(state.DB.QueryRowContext(ctx, `SELECT `+rowColumns+` FROM machines WHERE request_id = ?`, requestID))
}

// listRows returns machines newest first; live only with activeOnly.
func listRows(ctx context.Context, activeOnly bool) ([]*row, error) {
	q := `SELECT ` + rowColumns + ` FROM machines`
	if activeOnly {
		q += ` WHERE ended_at IS NULL`
	}
	rows, err := state.DB.QueryContext(ctx, q+` ORDER BY created_at DESC, id DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []*row
	for rows.Next() {
		r, err := scanRow(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func deleteRow(ctx context.Context, id string) error {
	_, err := state.DB.ExecContext(ctx, `DELETE FROM machines WHERE id = ?`, id)
	return err
}

// liveCount counts the machines of k that may be billing: listed at the
// provider and not ended, still being created, or a create that never
// listed (it may have been made). Only rows of the current mode (simulated
// or real) count, so simulate never blocks real launches.
func liveCount(ctx context.Context, k Kind, simulated bool) (int, error) {
	sim := 0
	if simulated {
		sim = 1
	}
	var n int
	err := state.DB.QueryRowContext(ctx, `SELECT COUNT(*) FROM machines WHERE provider = ? AND simulated = ?
		AND ended_at IS NULL AND (provider_id != '' OR status = ? OR problem = ?)`,
		string(k), sim, string(StatusCreating), string(ProblemNeverListed)).Scan(&n)
	return n, err
}

// nameTaken reports whether a live machine other than exceptID has name.
func nameTaken(ctx context.Context, name, exceptID string) (bool, error) {
	var n int
	err := state.DB.QueryRowContext(ctx, `SELECT COUNT(*) FROM machines WHERE name = ? AND ended_at IS NULL AND id != ?`,
		name, exceptID).Scan(&n)
	return n > 0, err
}

// deleteSetting removes a fork setting (state has no helper for it).
func deleteSetting(ctx context.Context, key string) error {
	_, err := state.DB.ExecContext(ctx, `DELETE FROM settings WHERE key = ?`, key)
	return err
}

func msTime(ms *int64) *time.Time {
	if ms == nil {
		return nil
	}
	t := state.Time(*ms)
	return &t
}

func ms(t time.Time) *int64 {
	v := t.UnixMilli()
	return &v
}

// selfIDs are this process's own instance ids per provider (Machine.self).
func selfID(k Kind) string {
	switch k {
	case RunPod:
		return strings.TrimSpace(getenv("RUNPOD_POD_ID"))
	case Vast:
		return strings.TrimSpace(getenv("CONTAINER_ID"))
	case QuickPod:
		return strings.TrimSpace(getenv("CONTAINER_LABEL"))
	}
	return ""
}

// machine is the API view of r (never the token).
func (s *Service) machine(r *row) Machine {
	m := Machine{
		ID: r.ID, Name: r.Name, Label: r.Label, Provider: r.Provider, ProviderID: r.ProviderID, Status: r.Status,
		Detail: r.Detail, ProviderStatus: r.ProviderStatus, GPU: r.Spec.Offer.GPU, GPUCount: r.Spec.GPUCount,
		VRAMGB: r.Spec.Offer.VRAMGB, Location: r.Spec.Offer.Location, Image: r.Spec.Image,
		ImageWarnings: nonNil(r.Spec.ImageWarnings), DiskGB: r.Spec.DiskGB, Secure: strings.HasPrefix(r.BaseURL, "https://"),
		RelayPath: "/v1/machines/" + r.ID + "/relay", PricePerHour: r.Price, StoppedPricePerHour: r.StoppedPrice,
		WhenDone: r.Spec.WhenDone, SelfDestroyKey: r.Spec.SelfDestroyKey, CopiedRemote: r.Spec.CopiedRemote,
		Actions: s.actions(r), Self: r.isSelf(), Simulated: r.Simulated, Warnings: nonNil(r.Spec.Warnings),
		CreatedAt: state.Time(r.CreatedAt), UpdatedAt: state.Time(r.UpdatedAt), ReadyAt: msTime(r.ReadyAt),
		EndedAt: msTime(r.EndedAt),
	}
	if m.GPUCount == 0 {
		m.GPUCount = r.Spec.Offer.GPUCount
	}
	if r.Problem != "" {
		p := r.Problem
		m.Problem = &p
	}
	if r.BaseURL != "" {
		b := r.BaseURL
		m.BaseURL = &b
	}
	if r.ImageDigest != "" {
		d := r.ImageDigest
		m.ImageDigest = &d
	}
	if r.ImageLaunchRef != "" {
		l := r.ImageLaunchRef
		m.ImageLaunchRef = &l
	}
	return m
}

func (r *row) isSelf() bool {
	return r.ProviderID != "" && r.ProviderID == selfID(r.Provider)
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

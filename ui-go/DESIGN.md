# UI design: "quiet tool"

Shared by the console (`web/`) and the studio (`studio/`) so they read as one
product. The console is denser, the studio roomier; the tokens, type and
components are the same. The UI stays neutral so samples and the loss chart
carry the colour.

## Rules

- Hairline borders all the way round. Never a coloured border on one side of a
  rounded box, and no inset-shadow stripes. Callouts use a tinted background, a
  full border and an icon.
- Sentence case everywhere. No all-caps labels, no letter-spaced eyebrows, no
  serif display type.
- Mono only for IDs, paths, code, logs and config. Numbers use the sans with
  `font-variant-numeric: tabular-nums`.
- One accent colour, used for primary actions, focus, progress and the active
  run. Status colours appear as small dots or tinted badges, not big fills.
- No shadows on cards (border only). Shadows only on menus, popovers, dialogs.
- Transitions 120–150 ms on colour and background only.

## Tokens

Both apps define these names on `:root`, switch with `html[data-theme]`
(`light`, `dark`, or absent = follow the system), and persist the choice in
localStorage inside try/catch.

| token | light | dark |
|---|---|---|
| `--bg` | `#fafafa` | `#0a0a0b` |
| `--surface` | `#ffffff` | `#111113` |
| `--surface-2` (hover, fills, code) | `#f4f4f5` | `#18181b` |
| `--border` | `#e4e4e7` | `#232327` |
| `--border-strong` | `#d4d4d8` | `#2e2e33` |
| `--text` | `#09090b` | `#ededef` |
| `--text-2` (secondary) | `#52525b` | `#a1a1aa` |
| `--text-3` (muted) | `#a1a1aa` | `#63636b` |
| `--accent` | `#5b5bd6` | `#7c7cf0` |
| `--accent-fg` (text on accent) | `#ffffff` | `#ffffff` |
| `--accent-soft` | `rgb(91 91 214 / 0.10)` | `rgb(124 124 240 / 0.16)` |
| `--ok` | `#15803d` | `#3dd68c` |
| `--warn` | `#b45309` | `#ffb224` |
| `--danger` | `#dc2626` | `#ff6369` |

Soft status fills are the status colour at 10% (light) / 16% (dark) via
`color-mix`.

Radius: `--r-sm: 6px` (controls, badges, thumbnails), `--r-md: 8px` (cards,
panels, logs), `--r-lg: 12px` (dialogs). `999px` only for status dots and the
progress track.

## Type

Geist (400, 500, 600) and Geist Mono (400, 500) from Google Fonts, falling back
to `system-ui` and `ui-monospace`.

Scale: 12 / 13 / 14 / 16 / 20 / 24 / 32 px. Body 14 px (studio prose may use
15). Headings weight 600, letter-spacing −0.01em at 20 px, −0.02em at 24 px and
up. Labels 13 px weight 500 in `--text-2`.

## Layout

- Left sidebar (console 220 px, studio 232 px): small mark and app name, nav
  items with 16 px lucide icons; the active item gets `--surface-2` and
  `--text`, nothing else. Footer holds connection or server status, the theme
  toggle and sign-out where relevant.
- Under 768 px the sidebar becomes a top bar with a menu button; no horizontal
  page scroll at 390 px, 16 px side gutters.
- Page header: title (20–24 px), one line of `--text-2` description, actions
  aligned right.
- Content width: console fluid up to 1280 px; studio up to 1080 px.

## Components

- Buttons: height 32 (console) / 36 (studio), radius 6. Primary = accent fill;
  secondary = surface + border; ghost = no border, `--surface-2` on hover;
  destructive = danger text on secondary, filled danger only inside a confirm.
- Inputs and selects: same heights, border, radius 6; focus = accent border
  plus a 3 px `--accent-soft` ring.
- Status: 8 px dot + label. running = accent (gentle pulse), queued = text-3,
  completed = ok, failed = danger, stopped = text-2, interrupted = warn. Badge
  variant = soft fill + coloured text, no border.
- Tabs: text-2 labels; the active one gets `--text` and a 2 px bottom rule.
- Tables: 12–13 px header in text-2 weight 500, 40 px rows, hairline row
  borders, `--surface-2` on hover.
- Cards: surface, border, radius 8, padding 16 (console) / 20 (studio).
- Progress: 6 px track in `--surface-2`, accent fill, radius 999, with
  `step / total`, it/s and ETA beside it in tabular numbers.
- Logs: Geist Mono 12 px on `--surface-2`, border, radius 8.
- Charts: smoothed loss as a 1.5 px accent line, raw points in text-3 at low
  opacity, gridlines in `--border`, 11 px tabular axis labels in text-3.
- Sample grids: radius 6 thumbnails on the page background, 8 px gaps, a 13 px
  "Step 250" label above each row. No black bands or film strips.
- Empty states: centred icon, one sentence, one action.

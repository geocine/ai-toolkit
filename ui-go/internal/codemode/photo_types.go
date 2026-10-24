package codemode

// PhotoTypesTS declares the photo-check shapes for the aitk client. It is
// added to the declaration by ops_photos.go through RegisterTypes; keep it
// equal to studio/src/lib/photoTypes.ts and web/src/photoTypes.ts (minus
// `export`, and with PhotoFaces.job typed as the declared Job).
const PhotoTypesTS = `type PhotoKind = 'person' | 'style' | 'object';
type PhotoLevel = 'block' | 'warn' | 'info';
type PhotoFlag =
  | 'unreadable' | 'unsupported' | 'huge' | 'gray16' | 'transparent' | 'animated' | 'cmyk' | 'same_stem' | 'rotated' | 'generated' | 'derived'
  | 'duplicate' | 'similar' | 'burst'
  | 'tiny' | 'small' | 'upscaled' | 'blurry' | 'dark' | 'bright' | 'flat' | 'grayscale' | 'strong_color' | 'noisy' | 'compressed'
  | 'screenshot' | 'from_screen' | 'bars' | 'aspect'
  | 'mentions_covered' | 'mentions_people' | 'mentions_text' | 'mentions_tone' | 'mentions_screen'
  | 'no_face' | 'others' | 'small_face' | 'face_cut' | 'face_blurry' | 'backlit' | 'not_them' | 'identity_check' | 'smoothed' | 'text' | 'watermark'
  | 'eyes_hidden' | 'covered' | 'filtered' | 'collage' | 'back_view' | 'expression_extreme' | 'lookalike';
type PhotoDecision = 'keep' | 'intended';
type PhotoExif = { time?: string; make?: string; model?: string; software?: string };
type PhotoMetrics = {
  version: 1; file: string; sha256?: string; bytes: number;
  format: 'jpeg' | 'png' | 'webp' | 'gif' | 'bmp' | 'tiff' | 'heic' | 'heif' | 'avif' | 'jxl' | 'other';
  width: number; height: number; orientation: number; decoded: boolean; error?: string; source: 'server' | 'browser';
  mode?: 'rgb' | 'gray' | 'gray16' | 'cmyk' | 'paletted'; alpha_transparent?: number; animated?: boolean;
  dhash?: string; dhash_flip?: string; thumb32?: string;
  sharp?: number; luma_mean?: number; luma_std?: number; clip_lo?: number; clip_hi?: number; clip_hi_center?: number;
  colorful?: number; chroma_spread?: number; layout?: string; flat?: number; bars?: number; ui_band?: boolean;
  noise?: number; detail?: number; jpeg_q?: number; png_color_type?: number; png_bit_depth?: number;
  exif?: PhotoExif; generated?: boolean;
};
type PhotoItemFace = { count: number; px: number; shot?: 'close' | 'bust' | 'half' | 'full' | 'back'; yaw?: 'front' | 'three_quarter' | 'profile';
  cohesion?: number; ref_sim?: number; box?: [number, number, number, number] };
type PhotoItem = {
  file: string; sha256?: string; width?: number; height?: number; bytes: number; px?: number;
  flags: PhotoFlag[]; waived?: PhotoFlag[]; level?: 'warn' | 'info'; notes: Partial<Record<PhotoFlag, string>>;
  group?: string; keeper?: boolean; same_as?: string; sharp_rel?: number; capture_time?: string; session?: number;
  face?: PhotoItemFace; decision?: PhotoDecision; reference?: boolean; derived_from?: string; pending?: boolean; usable: boolean;
};
type PhotoFixOp =
  | { op: 'set_aside'; files: string[] }
  | { op: 'restore'; files?: string[]; all?: boolean }
  | { op: 'set_aside_flagged'; flags?: PhotoFlag[] }
  | { op: 'keep_best'; n?: number }
  | { op: 'bake_orientation'; files?: string[] }
  | { op: 'convert'; files?: string[] }
  | { op: 'rename_same_stem' }
  | { op: 'face_crops'; max?: number }
  | { op: 'remove_face_crops' };
type PhotoFinding = { id: string; level: PhotoLevel; message: string; fix?: string; files?: string[]; op?: PhotoFixOp };
type PhotoGroup = { id: string; kind: 'duplicate' | 'similar' | 'burst'; files: string[]; keep: string; distance: number; confirmed?: boolean };
type PhotoCoverage = { basis: ('pixels' | 'exif' | 'captions' | 'faces')[]; shot?: Record<string, number>; yaw?: Record<string, number>;
  dated: number; sessions: number; largest_session: number; span_days: number;
  outfits?: Record<string, number>; places?: Record<string, number>; light?: Record<string, number> };
type PhotoFaces = { available: boolean; state: 'none' | 'queued' | 'running' | 'done' | 'failed' | 'stopped'; checked: number;
  stale: boolean; added: number; job_id?: string; job: Job | null; skipped?: string[] };
type PhotoPick = { n: number; kept: string[]; left_out: { file: string; reason: string }[]; note?: string };
type PhotoReview = {
  kind: PhotoKind; resolution: number; metrics_version: number;
  items: PhotoItem[]; total: number; flagged: number; usable: number; pending: number;
  counts: Partial<Record<PhotoFlag, number>>; groups: PhotoGroup[];
  grade: 'A' | 'B' | 'C' | 'D' | 'F'; grade_line: string; provisional: boolean; basis: string[];
  coverage: PhotoCoverage; findings: PhotoFinding[]; hints: string[];
  set_aside: { file: string; from: string; flags: PhotoFlag[] }[];
  faces: PhotoFaces; reference?: string; pick?: PhotoPick; thresholds: Record<string, number>; convert: string[];
};
type PhotoFix = { changes: { file: string; op: string; to?: string; reason?: string }[]; applied: boolean; review: PhotoReview };
type PhotoSettingsPatch = { kind?: PhotoKind; reference?: string | null; decisions?: Record<string, PhotoDecision | null> };
type PhotoAssess = { items: PhotoMetrics[]; kind?: PhotoKind; resolution?: number | number[]; dataset?: string; pick?: number;
  decisions?: Record<string, PhotoDecision>; captions?: Record<string, string> };
type UploadResult = { items: string[]; renamed: { from: string; to: string }[]; skipped: string[] };
`

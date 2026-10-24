import type { ReactNode } from 'react';
import { actionLabel, gb, minutes, shortDigest, uncheckedOnly } from '../lib/serverImage';
import type { CheckAction, ImageCheck, ImageCheckItem, ImageFacts } from '../lib/serverImageTypes';
import { Callout } from './ui';

/** "AI Toolkit 0.13.34 · build f53a407" for a Go-server image, else its reference. */
export function imageName(img: ImageFacts | null | undefined, ref: string): string {
  if (!img?.go_server) return ref;
  const version = img.labels['io.github.geocine.aitk.server'] === 'go' ? img.labels['org.opencontainers.image.version'] : '';
  let s = 'AI Toolkit' + (version ? ` ${version}` : '');
  if (img.revision) s += ` · build ${img.revision}`;
  else if (img.tag) s += ` · ${img.tag}`;
  return s;
}

/** The parts of the value line after the name: CUDA floor and download size. */
function imageFacts(img: ImageFacts): string[] {
  const out: string[] = [];
  if (img.requires?.cuda_min) out.push(`CUDA ${img.requires.cuda_min}`);
  if (img.requires?.flavor && img.requires.flavor !== 'cu130') out.push(img.requires.flavor);
  if (img.download_bytes > 0) out.push(gb(img.download_bytes));
  return out;
}

/**
 * The checks of an image check, rendered like the job preflight
 * (NewRun.tsx): one danger callout for every block, one warn callout, and
 * the infos as a quiet list. Each line is the message, the fix in muted
 * text and an optional action button.
 */
export function CheckList({ checks, onAction }: { checks: ImageCheckItem[]; onAction?: (a: CheckAction) => void }) {
  const line = (c: ImageCheckItem) => (
    <li key={`${c.code}:${c.message}`}>
      {c.message}
      {c.fix && <span className="muted"> · {c.fix}</span>}
      {c.action && onAction && (
        <>
          {' '}
          <button type="button" className="btn btn-ghost btn-sm check-action" onClick={() => onAction(c.action!)}>
            {actionLabel(c.action)}
          </button>
        </>
      )}
    </li>
  );
  const infos = checks.filter(c => c.level === 'info');
  return (
    <>
      {(['block', 'warn'] as const).map(level => {
        const list = checks.filter(c => c.level === level);
        return (
          list.length > 0 && (
            <Callout key={level} kind={level === 'block' ? 'danger' : 'warn'}>
              <ul className="problems">{list.map(line)}</ul>
            </Callout>
          )
        );
      })}
      {infos.length > 0 && <ul className="problems preflight-info">{infos.map(line)}</ul>}
    </>
  );
}

type RowState = { tone: string; label: string };

function rowState(check: ImageCheck | null, checking: boolean, error: unknown, hasHost: boolean): RowState {
  if (checking) return { tone: 'tone-quiet', label: 'Checking…' };
  if (error || !check) return { tone: 'tone-quiet', label: "Couldn't check" };
  if (uncheckedOnly(check.checks)) return { tone: 'tone-quiet', label: "Couldn't check" };
  switch (check.level) {
    case 'block':
      return { tone: 'tone-danger', label: hasHost ? "Won't run on this host" : "This image can't be used" };
    case 'warn':
      return { tone: 'tone-warn', label: hasHost ? 'Fits, with warnings' : 'Usable, with warnings' };
    default:
      return { tone: 'tone-ok', label: hasHost ? 'Fits this host' : 'Ready to run' };
  }
}

/**
 * The Image row of the create dialog (IMAGE-PLAN §6.3): what the GPU server
 * will run, whether the chosen host runs it, and why not.
 */
export function ImageRow({
  label,
  image,
  check,
  checking,
  error,
  hasHost,
  extra = [],
  onChange,
  onAction,
}: {
  label: ReactNode;
  image: string;
  check: ImageCheck | null;
  checking: boolean;
  error: unknown;
  hasHost: boolean;
  /** Notes from images.recommend, shown with the checks (a newer build, the template's tag). */
  extra?: ImageCheckItem[];
  onChange?: () => void;
  onAction: (a: CheckAction) => void;
}) {
  const img = check?.image ?? null;
  const state = rowState(check, checking, error, hasHost);
  const pull = !checking && hasHost ? check?.pull : null;
  const seen = new Set((check?.checks ?? []).map(c => `${c.code}:${c.message}`));
  const checks = checking ? [] : [...(check?.checks ?? []), ...extra.filter(c => !seen.has(`${c.code}:${c.message}`))];
  const need = check?.need;
  return (
    <div className="field field-wide image-row">
      <span>{label}</span>
      <div className="image-row-box">
        <div className="image-row-main">
          <p className="image-row-value">
            {img ? (
              <>
                {imageName(img, image)}
                {imageFacts(img).map(f => (
                  <span key={f} className="image-row-fact">
                    {' '}
                    · {f}
                  </span>
                ))}
              </>
            ) : (
              <span className="mono">{image || 'No image'}</span>
            )}
          </p>
          <p className={`image-row-state ${state.tone}`}>
            <span className="dot" aria-hidden />
            {state.label}
            {pull && <span className="muted"> · about {minutes(pull.image_min)} to download</span>}
          </p>
          {error != null && !checking && <p className="image-row-error">{error instanceof Error ? error.message : String(error)}</p>}
        </div>
        {onChange && (
          <button type="button" className="btn btn-sm" onClick={onChange}>
            Change
          </button>
        )}
      </div>
      {img && (
        <details className="image-row-details">
          <summary>Details</summary>
          <dl>
            <dt>Image</dt>
            <dd className="mono">{image}</dd>
            {check?.launch_ref && check.launch_ref !== image && (
              <>
                <dt>Launches as</dt>
                <dd className="mono">{check.launch_ref}</dd>
              </>
            )}
            {img.digest && (
              <>
                <dt>Digest</dt>
                <dd className="mono" title={img.digest}>
                  {shortDigest(img.digest)}
                </dd>
              </>
            )}
            {img.requires?.torch && (
              <>
                <dt>PyTorch</dt>
                <dd>{img.requires.torch}</dd>
              </>
            )}
            {img.unpacked_bytes > 0 && (
              <>
                <dt>Unpacked</dt>
                <dd>
                  {gb(img.unpacked_bytes)}
                  {img.unpacked_source === 'estimate' ? ' (estimate)' : ''}
                </dd>
              </>
            )}
            {need && (
              <>
                <dt>Disk it needs</dt>
                <dd>{diskBreakdown(need)}</dd>
              </>
            )}
          </dl>
        </details>
      )}
      <CheckList checks={checks} onAction={onAction} />
    </div>
  );
}

/** "image 20 GB + model 34 GB + room for checkpoints". */
export function diskParts(n: NonNullable<ImageCheck['need']>, modelName?: string): string {
  const parts = [`image ${Math.round(n.image_peak_gb)} GB`];
  if (n.model_gb > 0) parts.push(`${modelName ?? 'model'} ${Math.round(n.model_gb)} GB`);
  parts.push('room for checkpoints');
  return parts.join(' + ');
}

/** "80 GB (image 20 GB + model 34 GB + room for checkpoints)". */
export const diskBreakdown = (n: NonNullable<ImageCheck['need']>, modelName?: string) => `${n.disk_gb} GB (${diskParts(n, modelName)})`;

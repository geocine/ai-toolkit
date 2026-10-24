import { Dialog } from './Dialog';
import { FlagBadge, PhotoThumb, type GridPhoto } from './PhotoGrid';

// Two photos side by side (PHOTO-PLAN section 5): a duplicate, similar or
// burst pair is a claim the user checks with their own eyes, so each side
// can be left out or kept from here. A locked side (a photo already in the
// dataset, next to a dropped copy) is shown only: it is changed in the
// dataset, not here.

function Side({ photo, onLeaveOut, onKeep, locked }: { photo: GridPhoto; onLeaveOut(file: string): void; onKeep(file: string): void; locked?: boolean }) {
  const it = photo.item;
  return (
    <figure className="compare-side">
      <PhotoThumb photo={photo} full />
      <figcaption>
        <span className="mono compare-file" title={photo.file}>
          {photo.file}
        </span>
        {it && (it.width ?? 0) > 0 && (
          <span className="muted num">
            {it.width} × {it.height}
            {it.px ? ` · trains at about ${it.px} px` : ''}
          </span>
        )}
        <span className="compare-tags">
          {photo.leftOut ? <span className="badge tone-quiet">left out</span> : it && <FlagBadge item={it} />}
        </span>
        {locked ? (
          <span className="muted">Already in the dataset</span>
        ) : photo.leftOut ? (
          <button type="button" className="btn btn-sm" onClick={() => onKeep(photo.file)}>
            Keep this one
          </button>
        ) : (
          <button type="button" className="btn btn-sm" onClick={() => onLeaveOut(photo.file)}>
            Leave this one out
          </button>
        )}
      </figcaption>
    </figure>
  );
}

export function CompareDialog(p: {
  a: GridPhoto;
  b: GridPhoto;
  onClose(): void;
  onLeaveOut(file: string): void;
  onKeep(file: string): void;
  /** Stored photos are shown without keep or leave out (the local view, adding to a dataset). */
  lockStored?: boolean;
}) {
  const locked = (x: GridPhoto) => !!p.lockStored && x.stored;
  return (
    <Dialog
      title="Compare"
      description="Keep the sharper one, or both if they really differ."
      onClose={p.onClose}
      wide
      footer={
        <button type="button" className="btn" onClick={p.onClose}>
          Done
        </button>
      }
    >
      <div className="compare-grid">
        <Side photo={p.a} onLeaveOut={p.onLeaveOut} onKeep={p.onKeep} locked={locked(p.a)} />
        <Side photo={p.b} onLeaveOut={p.onLeaveOut} onKeep={p.onKeep} locked={locked(p.b)} />
      </div>
    </Dialog>
  );
}

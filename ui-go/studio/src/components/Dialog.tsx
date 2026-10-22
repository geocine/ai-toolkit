import { X } from 'lucide-react';
import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

/** A modal dialog, portalled above the shell. Escape and the backdrop close it. */
export function Dialog({
  title,
  description,
  onClose,
  children,
  footer,
  wide,
}: {
  title: ReactNode;
  description?: ReactNode;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  const panel = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close.current();
    window.addEventListener('keydown', onKey);
    const first = panel.current?.querySelector<HTMLElement>('input:not([type=checkbox]), select, textarea');
    (first ?? panel.current)?.focus();
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, []);
  return createPortal(
    <div className="dialog-backdrop" onMouseDown={e => e.target === e.currentTarget && onClose()}>
      <div ref={panel} className={wide ? 'dialog dialog-wide' : 'dialog'} role="dialog" aria-modal="true" aria-label={typeof title === 'string' ? title : undefined} tabIndex={-1}>
        <header className="dialog-head">
          <div className="dialog-titles">
            <h2>{title}</h2>
            {description && <p className="dialog-desc">{description}</p>}
          </div>
          <button className="btn btn-ghost btn-icon" onClick={onClose} aria-label="Close">
            <X />
          </button>
        </header>
        <div className="dialog-body">{children}</div>
        {footer && <footer className="dialog-foot">{footer}</footer>}
      </div>
    </div>,
    document.body,
  );
}

/** A confirm dialog for destructive actions (filled danger lives only here). */
export function Confirm({
  title,
  children,
  confirm,
  onConfirm,
  onClose,
  busy,
}: {
  title: string;
  children: ReactNode;
  confirm: string;
  onConfirm: () => void;
  onClose: () => void;
  busy?: boolean;
}) {
  return (
    <Dialog
      title={title}
      onClose={onClose}
      footer={
        <>
          <button className="btn btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-danger-fill" onClick={onConfirm} disabled={busy}>
            {confirm}
          </button>
        </>
      }
    >
      <div className="dialog-text">{children}</div>
    </Dialog>
  );
}

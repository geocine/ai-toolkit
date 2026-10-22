import { Upload } from 'lucide-react';
import { useRef, useState } from 'react';

const accepted = /\.(jpe?g|png|webp|bmp|gif|avif|heic|txt)$/i;

/** Reads files and dropped folders (one level of entries per folder). */
async function fromDrop(dt: DataTransfer): Promise<File[]> {
  const out: File[] = [];
  const entries = Array.from(dt.items)
    .map(i => i.webkitGetAsEntry?.())
    .filter((e): e is FileSystemEntry => Boolean(e));
  if (!entries.length) return Array.from(dt.files);
  const walk = async (e: FileSystemEntry): Promise<void> => {
    if (e.isFile) {
      out.push(await new Promise<File>((res, rej) => (e as FileSystemFileEntry).file(res, rej)));
    } else if (e.isDirectory) {
      const reader = (e as FileSystemDirectoryEntry).createReader();
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const c of batch) await walk(c);
      }
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

export function Dropzone({ onFiles }: { onFiles: (f: File[]) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const take = (fs: File[]) => onFiles(fs.filter(f => accepted.test(f.name)));
  return (
    <div
      className={over ? 'dropzone is-over' : 'dropzone'}
      onDragOver={e => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={async e => {
        e.preventDefault();
        setOver(false);
        take(await fromDrop(e.dataTransfer));
      }}
      onClick={() => input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
    >
      <input
        ref={input}
        type="file"
        multiple
        className="visually-hidden"
        aria-label="Choose photos and captions"
        accept="image/*,.txt"
        onClick={e => e.stopPropagation()}
        onChange={e => take(Array.from(e.target.files ?? []))}
      />
      <span className="dropzone-icon">
        <Upload aria-hidden />
      </span>
      <p className="dropzone-title">Drop photos or a folder here</p>
      <p className="dropzone-hint">or click to choose. Matching .txt files become captions.</p>
    </div>
  );
}

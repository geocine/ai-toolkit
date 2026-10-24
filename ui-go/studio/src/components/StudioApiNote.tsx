import { Cpu } from 'lucide-react';
import { go } from '../lib/route';
import { Empty, PageHeader } from './ui';

/** Stands in for a training page when the active server is the studio API (Health.mode). */
export function StudioApiNote({ title }: { title?: string }) {
  return (
    <>
      {title && <PageHeader title={title} />}
      <Empty
        icon={<Cpu />}
        action={
          <button className="btn btn-primary" onClick={() => go({ page: 'servers', create: true })}>
            Create a GPU server
          </button>
        }
      >
        This is the studio API. It creates GPU servers but doesn't train.
      </Empty>
    </>
  );
}

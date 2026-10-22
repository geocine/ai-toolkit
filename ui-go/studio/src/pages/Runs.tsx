import { Activity, Plus } from 'lucide-react';
import { Empty, ErrorNote, NeedServer, PageHeader, Progress, StatePill, useLoad } from '../components/ui';
import { get, type Items, type Job } from '../lib/api';
import { useServer } from '../lib/context';
import { ago } from '../lib/format';
import { href } from '../lib/route';

export const totalSteps = (j: Job): number | null => j.total_steps ?? j.config?.config?.process?.[0]?.train?.steps ?? null;

export function Runs() {
  const server = useServer();
  const jobs = useLoad(server, s => get<Items<Job>>(s, '/v1/jobs'), [], 5000);
  if (!server) return <NeedServer title="Runs" />;
  const items = (jobs.data?.items ?? []).filter(j => j.type === 'train' || !j.type);
  return (
    <>
      <PageHeader
        title="Runs"
        description={<>Training runs on {server.name}.</>}
        actions={
          <a className="btn btn-primary" href={href({ page: 'new' })}>
            <Plus />
            New run
          </a>
        }
      />
      <ErrorNote error={jobs.error} />
      {jobs.data && !items.length && (
        <Empty
          icon={<Activity />}
          action={
            <a className="btn btn-primary" href={href({ page: 'new' })}>
              Start the first run
            </a>
          }
        >
          Nothing has been trained on this server yet.
        </Empty>
      )}
      {items.length > 0 && (
        <div className="runs" role="table" aria-label="Runs">
          <div className="runs-row runs-head" role="row">
            <span role="columnheader">Name</span>
            <span role="columnheader">Trigger</span>
            <span role="columnheader">Model</span>
            <span role="columnheader">Status</span>
            <span role="columnheader">Progress</span>
            <span role="columnheader" className="num">
              Updated
            </span>
          </div>
          {items.map(j => {
            const total = totalSteps(j);
            const p = j.config?.config?.process?.[0] ?? {};
            return (
              <a key={j.id} className="runs-row" role="row" href={href({ page: 'run', id: j.id })}>
                <span role="cell" className="runs-name">
                  {j.name}
                </span>
                <span role="cell" className="runs-trigger">
                  {p.trigger_word ?? '–'}
                </span>
                <span role="cell" className="runs-model">
                  {p.model?.arch ?? '–'} · {p.network?.type ?? 'lora'}
                </span>
                <span role="cell" className="runs-state">
                  <StatePill state={j.state} />
                </span>
                <span role="cell" className="runs-progress">
                  <Progress value={j.step} max={total} />
                  <span className="num">
                    {j.step.toLocaleString()}
                    <span className="of"> / {total?.toLocaleString() ?? '?'}</span>
                  </span>
                </span>
                <span role="cell" className="runs-updated num">
                  {ago(j.updated_at)}
                </span>
              </a>
            );
          })}
        </div>
      )}
    </>
  );
}

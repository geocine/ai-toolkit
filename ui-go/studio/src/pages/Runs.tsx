import { ErrorNote, NeedServer, Progress, Section, StatePill, useLoad } from '../components/ui';
import { get, type Items, type Job } from '../lib/api';
import { useServer } from '../lib/context';
import { ago, pad } from '../lib/format';
import { href } from '../lib/route';

export const totalSteps = (j: Job): number | null => j.total_steps ?? j.config?.config?.process?.[0]?.train?.steps ?? null;

export function Runs() {
  const server = useServer();
  const jobs = useLoad(server, s => get<Items<Job>>(s, '/v1/jobs'), [], 5000);
  if (!server) return <NeedServer />;
  const items = (jobs.data?.items ?? []).filter(j => j.type === 'train' || !j.type);
  return (
    <Section
      kicker={server.name}
      title="Runs"
      actions={
        <a className="btn btn-ink" href={href({ page: 'new' })}>
          New run
        </a>
      }
    >
      <ErrorNote error={jobs.error} />
      {jobs.data && !items.length && (
        <div className="empty">
          <p className="empty-title">Nothing trained here yet.</p>
          <a className="btn btn-ink" href={href({ page: 'new' })}>
            Start the first run
          </a>
        </div>
      )}
      <ul className="runs">
        {items.map((j, i) => {
          const total = totalSteps(j);
          const p = j.config?.config?.process?.[0] ?? {};
          return (
            <li key={j.id} style={{ animationDelay: `${Math.min(i, 12) * 40}ms` }}>
              <a className="run-card" href={href({ page: 'run', id: j.id })}>
                <span className="run-step mono">{pad(j.step)}</span>
                <span className="run-main">
                  <span className="run-name">{j.name}</span>
                  <span className="run-meta">
                    {p.trigger_word && <>trigger <b>{p.trigger_word}</b> · </>}
                    {p.model?.arch ?? '—'} · {p.network?.type ?? 'lora'} · updated {ago(j.updated_at)}
                  </span>
                  <Progress value={j.step} max={total} />
                </span>
                <span className="run-side">
                  <StatePill state={j.state} />
                  <span className="mono dim">
                    {j.step}/{total ?? '?'}
                  </span>
                </span>
              </a>
            </li>
          );
        })}
      </ul>
    </Section>
  );
}

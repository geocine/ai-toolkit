import { useMemo, useState } from 'react';
import { ListChecks } from 'lucide-react';
import { api, fmtAgo, plural } from '../api';
import { go, href } from '../router';
import type { Items, Job } from '../types';
import { Bar, Empty, Loading, Mono, StateBadge, usePoll } from '../ui';

const FILTERS = ['all', 'active', 'running', 'queued', 'completed', 'failed', 'interrupted', 'stopped', 'draft'] as const;

export function JobTableHead() {
  return (
    <thead>
      <tr>
        <th>Name</th>
        <th>State</th>
        <th style={{ minWidth: 140 }}>Progress</th>
        <th>Speed</th>
        <th>GPU</th>
        <th>Updated</th>
      </tr>
    </thead>
  );
}

export function JobRow({ job: j }: { job: Job }) {
  const total = j.total_steps ?? stepsOf(j);
  return (
    <tr className="clickable" onClick={() => go('jobs', j.id)}>
      <td>
        <a href={href('jobs', j.id)} onClick={e => e.stopPropagation()} style={{ fontWeight: 700 }}>
          {j.name}
        </a>
        <div className="dim small ellipsis" style={{ maxWidth: 360 }} title={j.info}>
          {j.info || j.type}
        </div>
      </td>
      <td>
        <StateBadge state={j.state} />
      </td>
      <td>
        <div className="mono small">
          {j.step}
          {total ? ` / ${total}` : ''}
        </div>
        {total ? <Bar value={j.step} max={total} /> : null}
      </td>
      <td className="mono small nowrap">{j.speed || '—'}</td>
      <td>
        <Mono>{j.gpu_ids}</Mono>
      </td>
      <td className="small muted nowrap">{fmtAgo(j.updated_at)}</td>
    </tr>
  );
}

/** Total steps from the config when the row has none (it is set by the trainer). */
export const stepsOf = (j: Job): number | null => {
  const s = j.config?.config?.process?.[0]?.train?.steps;
  return typeof s === 'number' ? s : null;
};

export function Jobs() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>('all');
  const [query, setQuery] = useState('');

  usePoll(async signal => {
    setJobs((await api.get<Items<Job>>('/v1/jobs', signal)).items);
  }, 4000);

  const shown = useMemo(() => {
    let list = jobs ?? [];
    if (filter === 'active') list = list.filter(j => ['queued', 'starting', 'running', 'stopping'].includes(j.state));
    else if (filter !== 'all') list = list.filter(j => j.state === filter || (filter === 'running' && j.state === 'starting'));
    if (query) list = list.filter(j => j.name.toLowerCase().includes(query.toLowerCase()));
    return list;
  }, [jobs, filter, query]);

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Jobs</h1>
          <div className="page-sub">{jobs ? plural(jobs.length, 'job') : ''}</div>
        </div>
        <div className="row-wrap">
          <input type="search" placeholder="Filter by name" value={query} onChange={e => setQuery(e.target.value)} style={{ width: 200 }} />
          <select value={filter} onChange={e => setFilter(e.target.value as any)} style={{ width: 150 }}>
            {FILTERS.map(f => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </div>
      </div>
      {!jobs ? (
        <Loading what="jobs" />
      ) : shown.length === 0 ? (
        <Empty icon={<ListChecks size={28} />} title={jobs.length ? 'No jobs match' : 'No jobs yet'}>
          Jobs are created by apps and agents through the API (POST /v1/jobs or MCP).
        </Empty>
      ) : (
        <div className="table-wrap">
          <table>
            <JobTableHead />
            <tbody>
              {shown.map(j => (
                <JobRow key={j.id} job={j} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

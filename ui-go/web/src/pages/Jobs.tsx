import { useMemo, useState } from 'react';
import { FileArchive, ListChecks, Search } from 'lucide-react';
import { api, fmtAgo, plural } from '../api';
import { ImportActivity, ImportDialog, type ImportStart } from '../bundleUi';
import { useTasks } from '../bundles';
import { go, href } from '../router';
import type { Items, Job } from '../types';
import { Bar, Empty, Loading, StateBadge, sentence, usePoll } from '../ui';

const FILTERS = ['all', 'active', 'running', 'queued', 'completed', 'failed', 'interrupted', 'stopped', 'draft'] as const;

export function JobTableHead() {
  return (
    <thead>
      <tr>
        <th>Name</th>
        <th className="col-state">State</th>
        <th className="col-progress">Progress</th>
        <th className="col-speed hide-sm">Speed</th>
        <th className="col-gpu hide-sm">GPU</th>
        <th className="col-updated hide-sm">Updated</th>
      </tr>
    </thead>
  );
}

export function JobRow({ job: j }: { job: Job }) {
  const total = j.total_steps ?? stepsOf(j);
  return (
    <tr className="clickable" onClick={() => go('jobs', j.id)}>
      <td>
        <a className="cell-title" href={href('jobs', j.id)} onClick={e => e.stopPropagation()}>
          {j.name}
        </a>
        <div className="cell-sub ellipsis" title={j.info}>
          {j.info || j.type}
        </div>
      </td>
      <td>
        <StateBadge state={j.state} />
      </td>
      <td>
        <div className="cell-progress">
          {total ? <Bar value={j.step} max={total} /> : null}
          <span className="num">
            {j.step}
            {total ? ` / ${total}` : ''}
          </span>
        </div>
      </td>
      <td className="small muted nowrap hide-sm">{j.speed || '—'}</td>
      <td className="small muted hide-sm">{j.gpu_ids}</td>
      <td className="small muted nowrap hide-sm">{fmtAgo(j.updated_at)}</td>
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
  const [importing, setImporting] = useState<ImportStart | null>(null);
  const imports = useTasks('import');

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
          <div className="page-sub">{jobs ? `${plural(jobs.length, 'job')} on this server` : 'Training jobs on this server'}</div>
        </div>
        <div className="page-actions">
          <label className="search">
            <Search size={14} />
            <input type="search" placeholder="Filter by name" value={query} onChange={e => setQuery(e.target.value)} style={{ width: 220 }} aria-label="Filter by name" />
          </label>
          <select value={filter} onChange={e => setFilter(e.target.value as any)} style={{ width: 150 }} aria-label="State">
            {FILTERS.map(f => (
              <option key={f} value={f}>
                {f === 'all' ? 'All states' : sentence(f)}
              </option>
            ))}
          </select>
          <button className="btn btn-primary" onClick={() => setImporting({})}>
            <FileArchive size={14} /> Import bundle
          </button>
        </div>
      </div>
      <ImportActivity tasks={imports} onLoad={setImporting} />
      {importing && <ImportDialog key={JSON.stringify(importing.source ?? null)} start={importing} tasks={imports} onClose={() => setImporting(null)} />}
      {!jobs ? (
        <Loading what="jobs" />
      ) : shown.length === 0 ? (
        <Empty
          icon={<ListChecks size={18} />}
          title={jobs.length ? 'No jobs match' : 'No jobs yet'}
          action={
            jobs.length ? (
              <button
                className="btn btn-sm"
                onClick={() => {
                  setQuery('');
                  setFilter('all');
                }}
              >
                Clear filters
              </button>
            ) : undefined
          }
        >
          {jobs.length ? 'Try another name or state.' : <>Apps and agents create jobs through the API (<span className="mono">POST /v1/jobs</span> or MCP).</>}
        </Empty>
      ) : (
        <div className="table-wrap">
          <table className="fixed">
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

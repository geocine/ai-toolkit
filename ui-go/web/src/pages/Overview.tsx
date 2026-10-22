import { useState } from 'react';
import { ArrowRight, Cpu, HardDrive, Layers, ListChecks, MonitorCog, Pause, Play } from 'lucide-react';
import { api, fmtBytes } from '../api';
import { href } from '../router';
import type { Items, Job, Queue, SystemInfo } from '../types';
import { Bar, Empty, Loading, Section, StateBadge, sentence, usePoll, useToast } from '../ui';
import { LifeNotice } from '../lifecycle';
import { JobRow, JobTableHead } from './Jobs';

const ACTIVE = new Set(['queued', 'starting', 'running', 'stopping']);

export function Overview() {
  const toast = useToast();
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [queues, setQueues] = useState<Queue[] | null>(null);
  const [jobs, setJobs] = useState<Job[] | null>(null);

  // nvidia-smi can take a moment; queues and jobs should not wait for it
  usePoll(async signal => setSys(await api.get<SystemInfo>('/v1/system', signal)), 5000);
  const refresh = usePoll(async signal => {
    const [q, j] = await Promise.all([api.get<Items<Queue>>('/v1/queues', signal), api.get<Items<Job>>('/v1/jobs', signal)]);
    setQueues(q.items);
    setJobs(j.items);
  }, 5000);

  const active = jobs?.filter(j => ACTIVE.has(j.state)) ?? [];
  const recent = jobs?.filter(j => !ACTIVE.has(j.state)).slice(0, 5) ?? [];

  const toggleQueue = async (q: Queue) => {
    try {
      await api.post(`/v1/queues/${encodeURIComponent(q.gpu_ids)}/${q.running ? 'stop' : 'start'}`);
      toast.success(`Queue ${q.gpu_ids} ${q.running ? 'stopped' : 'started'}`);
      refresh();
    } catch (e) {
      toast.error(e);
    }
  };

  const gb = (mb: number) => (mb / 1024).toFixed(1);

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Overview</h1>
          <div className="page-sub ellipsis">
            {sys ? (
              <>
                Machine, queues and jobs for <span className="mono">{sys.toolkit_root}</span>
              </>
            ) : (
              'Machine, queues and jobs on this server'
            )}
          </div>
        </div>
      </div>
      <LifeNotice />

      <Section title="Machine">
        {!sys ? (
          <Loading what="system" />
        ) : (
          <div className="grid grid-3">
            {sys.gpus.length === 0 && (
              <div className="card">
                <div className="stat-label">
                  <MonitorCog size={14} /> GPU
                </div>
                <div className="muted small" style={{ marginTop: 8 }}>
                  {sys.has_nvidia_smi ? 'No GPUs reported' : 'nvidia-smi not available'}
                </div>
              </div>
            )}
            {sys.gpus.map(g => (
              <div className="card" key={g.index}>
                <div className="stat-label">
                  <MonitorCog size={14} /> GPU {g.index}
                </div>
                <div className="stat-name ellipsis" title={g.name}>
                  {g.name ?? 'GPU'}
                </div>
                <div className="metric">
                  <div className="metric-row">
                    <span>Load</span>
                    <span>{g.utilization.gpu}%</span>
                  </div>
                  <Bar value={g.utilization.gpu} max={100} neutral />
                </div>
                <div className="metric">
                  <div className="metric-row">
                    <span>VRAM</span>
                    <span>
                      {gb(g.memory.used)} / {gb(g.memory.total)} GB
                    </span>
                  </div>
                  <Bar value={g.memory.used} max={g.memory.total} neutral />
                </div>
                <div className="metric metric-row">
                  <span>Temp · power</span>
                  <span>
                    {g.temperature}°C · {Math.round(g.power.draw)} / {Math.round(g.power.limit)} W
                  </span>
                </div>
              </div>
            ))}
            <div className="card">
              <div className="stat-label">
                <Cpu size={14} /> CPU
              </div>
              <div className="stat-name ellipsis" title={sys.cpu.name}>
                {sys.cpu.name || 'CPU'}
              </div>
              <div className="metric">
                <div className="metric-row">
                  <span>Load · {sys.cpu.cores} cores</span>
                  <span>{Math.round(sys.cpu.currentLoad)}%</span>
                </div>
                <Bar value={sys.cpu.currentLoad} max={100} neutral />
              </div>
              <div className="metric">
                <div className="metric-row">
                  <span>Memory</span>
                  <span>
                    {gb(sys.cpu.totalMemory - sys.cpu.availableMemory)} / {gb(sys.cpu.totalMemory)} GB
                  </span>
                </div>
                <Bar value={sys.cpu.totalMemory - sys.cpu.availableMemory} max={sys.cpu.totalMemory} neutral />
              </div>
            </div>
            <div className="card">
              <div className="stat-label">
                <HardDrive size={14} /> Disks
              </div>
              {sys.disks.map(d => (
                <div className="metric metric-tight" key={d.name}>
                  <div className="metric-row">
                    <span title={d.path}>{sentence(d.name)}</span>
                    <span>{fmtBytes(d.free)} free</span>
                  </div>
                  <Bar value={d.total - d.free} max={d.total} label={d.path} neutral />
                </div>
              ))}
            </div>
          </div>
        )}
      </Section>

      <Section title="Queues">
        {!queues ? (
          <Loading />
        ) : queues.length === 0 ? (
          <Empty icon={<Layers size={18} />} title="No queues yet">
            A GPU queue is created the first time a job is started on it.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>GPUs</th>
                  <th>State</th>
                  <th>Current job</th>
                  <th className="hide-sm">Waiting</th>
                  <th className="actions" />
                </tr>
              </thead>
              <tbody>
                {queues.map(q => {
                  const cur = jobs?.find(j => j.id === q.current_job_id);
                  return (
                    <tr key={q.gpu_ids}>
                      <td className="cell-title nowrap">GPU {q.gpu_ids}</td>
                      <td>
                        <StateBadge state={q.running ? 'running' : 'stopped'} />
                      </td>
                      <td>
                        {cur ? (
                          <a className="link" href={href('jobs', cur.id)}>
                            {cur.name}
                          </a>
                        ) : (
                          <span className="dim">—</span>
                        )}
                      </td>
                      <td className="hide-sm">{q.queued}</td>
                      <td className="actions">
                        <button className="btn btn-sm" onClick={() => toggleQueue(q)} title={q.running ? 'Stop queue' : 'Start queue'}>
                          {q.running ? <Pause size={14} /> : <Play size={14} />} <span className="hide-sm">{q.running ? 'Stop queue' : 'Start queue'}</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      <Section
        title="Active jobs"
        actions={
          <a className="btn btn-sm btn-ghost" href={href('jobs')}>
            All jobs <ArrowRight size={14} />
          </a>
        }
      >
        {!jobs ? (
          <Loading />
        ) : active.length === 0 ? (
          <Empty icon={<ListChecks size={18} />} title="Nothing is training">
            Queued and running jobs show up here.
          </Empty>
        ) : (
          <JobTable jobs={active} />
        )}
      </Section>

      {recent.length > 0 && (
        <Section title="Recent">
          <JobTable jobs={recent} />
        </Section>
      )}
    </>
  );
}

function JobTable({ jobs }: { jobs: Job[] }) {
  return (
    <div className="table-wrap">
      <table className="fixed">
        <JobTableHead />
        <tbody>
          {jobs.map(j => (
            <JobRow key={j.id} job={j} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

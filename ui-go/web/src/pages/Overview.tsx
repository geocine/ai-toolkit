import { useState } from 'react';
import { Cpu, HardDrive, MonitorCog, Pause, Play } from 'lucide-react';
import { api, fmtBytes } from '../api';
import { href } from '../router';
import type { Items, Job, Queue, SystemInfo } from '../types';
import { Bar, Empty, Loading, Section, StateBadge, usePoll, useToast } from '../ui';
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

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Overview</h1>
          <div className="page-sub mono">{sys?.toolkit_root}</div>
        </div>
      </div>

      <Section title="Machine">
        {!sys ? (
          <Loading what="system" />
        ) : (
          <div className="grid grid-3">
            {sys.gpus.length === 0 && (
              <div className="card">
                <div className="stat-label row">
                  <MonitorCog size={12} /> GPU
                </div>
                <div className="muted small" style={{ marginTop: 8 }}>
                  {sys.has_nvidia_smi ? 'No GPUs reported' : 'nvidia-smi not available'}
                </div>
              </div>
            )}
            {sys.gpus.map(g => (
              <div className="card" key={g.index}>
                <div className="stat-label row">
                  <MonitorCog size={12} /> GPU {g.index}
                </div>
                <div className="ellipsis" style={{ marginTop: 6, fontWeight: 700 }} title={g.name}>
                  {g.name ?? 'GPU'}
                </div>
                <div className="stat-row">
                  <span className="muted">Load</span>
                  <span className="mono">{g.utilization.gpu}%</span>
                </div>
                <Bar value={g.utilization.gpu} max={100} />
                <div className="stat-row">
                  <span className="muted">VRAM</span>
                  <span className="mono">
                    {(g.memory.used / 1024).toFixed(1)} / {(g.memory.total / 1024).toFixed(1)} GB
                  </span>
                </div>
                <Bar value={g.memory.used} max={g.memory.total} />
                <div className="stat-row">
                  <span className="muted">Temp · Power</span>
                  <span className="mono">
                    {g.temperature}°C · {Math.round(g.power.draw)}/{Math.round(g.power.limit)} W
                  </span>
                </div>
              </div>
            ))}
            <div className="card">
              <div className="stat-label row">
                <Cpu size={12} /> CPU
              </div>
              <div className="ellipsis" style={{ marginTop: 6, fontWeight: 700 }} title={sys.cpu.name}>
                {sys.cpu.name || 'CPU'}
              </div>
              <div className="stat-row">
                <span className="muted">Load · {sys.cpu.cores} cores</span>
                <span className="mono">{Math.round(sys.cpu.currentLoad)}%</span>
              </div>
              <Bar value={sys.cpu.currentLoad} max={100} />
              <div className="stat-row">
                <span className="muted">Memory</span>
                <span className="mono">
                  {((sys.cpu.totalMemory - sys.cpu.availableMemory) / 1024).toFixed(1)} / {(sys.cpu.totalMemory / 1024).toFixed(1)} GB
                </span>
              </div>
              <Bar value={sys.cpu.totalMemory - sys.cpu.availableMemory} max={sys.cpu.totalMemory} />
            </div>
            <div className="card">
              <div className="stat-label row">
                <HardDrive size={12} /> Disks
              </div>
              <div className="stack" style={{ marginTop: 8, gap: 10 }}>
                {sys.disks.map(d => (
                  <div key={d.name}>
                    <div className="stat-row" style={{ marginTop: 0 }}>
                      <span className="muted">{d.name}</span>
                      <span className="mono">{fmtBytes(d.free)} free</span>
                    </div>
                    <Bar value={d.total - d.free} max={d.total} label={d.path} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </Section>

      <Section title="Queues">
        {!queues ? (
          <Loading />
        ) : queues.length === 0 ? (
          <Empty title="No queues yet">A GPU queue is created the first time a job is started on it.</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>GPUs</th>
                  <th>State</th>
                  <th>Current job</th>
                  <th>Waiting</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {queues.map(q => {
                  const cur = jobs?.find(j => j.id === q.current_job_id);
                  return (
                    <tr key={q.gpu_ids}>
                      <td className="mono">{q.gpu_ids}</td>
                      <td>
                        <StateBadge state={q.running ? 'running' : 'stopped'} />
                      </td>
                      <td>{cur ? <a href={href('jobs', cur.id)}>{cur.name}</a> : <span className="dim">—</span>}</td>
                      <td className="mono">{q.queued}</td>
                      <td className="actions">
                        <button className="btn btn-sm" onClick={() => toggleQueue(q)}>
                          {q.running ? <Pause size={12} /> : <Play size={12} />} {q.running ? 'Stop' : 'Start'}
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

      <Section title="Active jobs" actions={<a className="btn btn-sm" href={href('jobs')}>All jobs</a>}>
        {!jobs ? (
          <Loading />
        ) : active.length === 0 ? (
          <Empty title="Nothing is training">Queued and running jobs show up here.</Empty>
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
      <table>
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

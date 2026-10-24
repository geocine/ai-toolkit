import { ScanFace } from 'lucide-react';
import { useEffect, useState } from 'react';
import { get, type DatasetDetail, type Job, type Server } from '../lib/api';
import { checkFaces, forgetPhotos } from '../lib/photos';
import type { PhotoReview } from '../lib/photoTypes';
import { Confirm } from './Dialog';
import { Callout, ErrorNote, Progress } from './ui';

// The face check (PHOTO-PLAN 13.1 item 4, section 0 decision 6): nothing
// biometric runs without this click and its consent line. It is a
// photo_faces job on the server's CPU queue; this follows the dataset's
// photo_job while it runs, then asks for the review again. Hidden until the
// server can run it (review.faces.available) and for sets that are not of a
// person.

const busyStates = new Set(['queued', 'starting', 'running', 'stopping']);

export function FaceCheck(p: { server: Server; dataset: string; review: PhotoReview; onChange(): void }) {
  const { server, dataset, review } = p;
  const faces = review.faces;
  const [job, setJob] = useState<Job | null>((faces.job as Job | null) ?? null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [ask, setAsk] = useState(false);
  const running = (!!job && busyStates.has(job.state)) || ((faces.state === 'queued' || faces.state === 'running') && !job);

  useEffect(() => setJob((faces.job as Job | null) ?? null), [faces.job]);

  // the job's progress, every 3 s like Auto caption; a finished job is read into the review
  useEffect(() => {
    if (!running) return;
    let stop = false;
    const id = setInterval(async () => {
      if (document.visibilityState === 'hidden') return;
      try {
        const d = await get<DatasetDetail>(server, `/v1/datasets/${encodeURIComponent(dataset)}?limit=1`);
        if (stop) return;
        const j = d.photo_job ?? null;
        setJob(j);
        if (!j || !busyStates.has(j.state)) p.onChange();
      } catch {
        /* the next tick tries again */
      }
    }, 3000);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [running, server, dataset]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!faces.available || review.kind !== 'person') return null;

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const j = await checkFaces(server, dataset);
      setWarnings(j.warnings ?? []);
      setJob(j);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  const forget = async () => {
    setBusy(true);
    setError(null);
    try {
      await forgetPhotos(server, dataset);
      setAsk(false);
      setJob(null);
      p.onChange();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  const forgetLink = (
    <button type="button" className="link" onClick={() => setAsk(true)} disabled={busy}>
      Forget analysis
    </button>
  );
  const confirm = ask && (
    <Confirm title="Forget the photo analysis?" confirm="Forget" onConfirm={forget} onClose={() => setAsk(false)} busy={busy}>
      Scores, face results, face-to-face similarities, the reference photo and your decisions on {dataset} are forgotten. The photos stay as they are.
    </Confirm>
  );

  let body;
  if (running) {
    body = (
      <div className="face-check-run">
        <p>
          Checking faces
          {job?.total_steps ? (
            <span className="muted num">
              {' '}
              · {job.step} of {job.total_steps}
            </span>
          ) : job?.state === 'queued' ? (
            <span className="muted"> · waiting for the CPU queue</span>
          ) : null}
        </p>
        <Progress value={job?.step ?? 0} max={job?.total_steps ?? null} indeterminate={!job?.total_steps} />
      </div>
    );
  } else if (faces.state === 'done' && faces.checked > 0) {
    body = faces.stale ? (
      <div className="face-check-line">
        <p>
          <span className="num">{faces.added}</span> {faces.added === 1 ? 'photo was' : 'photos were'} added after the face check.
        </p>
        <button type="button" className="btn btn-sm" onClick={start} disabled={busy}>
          {busy ? 'Starting…' : 'Check faces again'}
        </button>
        {forgetLink}
      </div>
    ) : (
      <p className="face-check-line quiet-note">
        <span>
          Faces checked in <span className="num">{faces.checked}</span> photos.
        </span>
        {forgetLink}
      </p>
    );
  } else {
    const failed = faces.state === 'failed' || job?.state === 'failed';
    body = (
      <Callout kind={failed ? 'warn' : 'info'}>
        <div className="face-check">
          <p className="face-check-title">
            <ScanFace aria-hidden />
            <strong>Check faces</strong> · finds other people, look-alikes, angles and framing.
          </p>
          {failed && <p>The last face check failed{job?.info ? `: ${job.info}` : ''}.</p>}
          <p>Runs on your server beside any training, about a minute for 30 photos (39 MB model download the first time).</p>
          <p className="muted">Only check photos of yourself or of someone who agreed. Faces are compared on your server; face templates are not stored.</p>
          <div className="face-check-foot">
            <button type="button" className="btn btn-sm btn-primary" onClick={start} disabled={busy}>
              {busy ? 'Starting…' : failed ? 'Check faces again' : 'Check faces'}
            </button>
            {forgetLink}
          </div>
        </div>
      </Callout>
    );
  }
  return (
    <div className="face-check-box">
      {body}
      {warnings.length > 0 && <p className="quiet-note">{warnings.join(' ')}</p>}
      {faces.skipped && faces.skipped.length > 0 && (
        <p className="quiet-note">
          Not checked: {faces.skipped.slice(0, 3).join(', ')}
          {faces.skipped.length > 3 ? ', …' : ''}
        </p>
      )}
      <ErrorNote error={error} />
      {confirm}
    </div>
  );
}

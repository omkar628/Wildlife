import { useEffect, useMemo, useState } from 'react'
import { api, type Job, type JobError } from '../api'

function selectedJobId(): number | null {
  const query = window.location.hash.split('?')[1] ?? ''
  const params = new URLSearchParams(query)
  const value = params.get('id')
  return value ? Number(value) : null
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [activeId, setActiveId] = useState<number | null>(selectedJobId())
  const [detail, setDetail] = useState<{ job: Job; errors: JobError[] } | null>(null)

  useEffect(() => {
    const load = () => api.jobs().then((body) => setJobs(body.jobs))
    load()
    const id = window.setInterval(load, 1500)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => {
    if (!activeId) {
      setDetail(null)
      return
    }
    const load = () => api.job(activeId).then(setDetail)
    load().catch(() => setDetail(null))
    const id = window.setInterval(() => load().catch(() => undefined), 1000)
    return () => window.clearInterval(id)
  }, [activeId])

  const job = detail?.job
  const percent = useMemo(() => {
    if (!job || !job.total_images) return 0
    const done = job.processed + job.skipped + job.duplicates + job.failed
    return Math.min(100, Math.round((done / job.total_images) * 100))
  }, [job])

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Processing</h2>
          <p>Long-running YOLO jobs stay on the backend so this page can keep polling.</p>
        </div>
      </div>
      <div className="grid two">
        <article className="card">
          <h3>Jobs</h3>
          {jobs.length === 0 ? <p className="empty">No jobs yet.</p> : (
            <table className="table">
              <thead>
                <tr><th>ID</th><th>Camera</th><th>Status</th><th>Progress</th></tr>
              </thead>
              <tbody>
                {jobs.map((item) => (
                  <tr key={item.job_id} onClick={() => setActiveId(item.job_id)} style={{ cursor: 'pointer' }}>
                    <td>#{item.job_id}</td>
                    <td>{item.camera_id}</td>
                    <td><span className={`badge ${item.status === 'completed' ? 'ok' : item.status === 'failed' ? 'bad' : 'warn'}`}>{item.status}</span></td>
                    <td>{item.processed}/{item.total_images}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
        <article className="card">
          <h3>Summary</h3>
          {!job ? <p className="empty">Select a job.</p> : (
            <>
              <p className="muted">{job.folder_path}</p>
              <div className="progress" style={{ margin: '12px 0 16px' }}>
                <span style={{ width: `${percent}%` }} />
              </div>
              <table className="table">
                <tbody>
                  <tr><td>Total images</td><td>{job.total_images}</td></tr>
                  <tr><td>Processed</td><td>{job.processed}</td></tr>
                  <tr><td>Duplicates</td><td>{job.duplicates}</td></tr>
                  <tr><td>Skipped</td><td>{job.skipped}</td></tr>
                  <tr><td>Failed</td><td>{job.failed}</td></tr>
                  <tr><td>Tiger / prey / rival / human</td><td>{job.tiger_count} / {job.prey_count} / {job.rival_count} / {job.human_count}</td></tr>
                  <tr><td>Low confidence / review</td><td>{job.low_confidence_count} / {job.review_count}</td></tr>
                  <tr><td>Threshold</td><td>{job.confidence_threshold}</td></tr>
                  <tr><td>Started</td><td>{job.started_at ?? '—'}</td></tr>
                  <tr><td>Finished</td><td>{job.finished_at ?? '—'}</td></tr>
                  {job.error_message ? <tr><td>Note</td><td>{job.error_message}</td></tr> : null}
                </tbody>
              </table>
              {detail && detail.errors.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h3>Failed files</h3>
                  {detail.errors.map((item) => (
                    <p key={item.error_id} className="error" style={{ fontSize: 13 }}>
                      {item.original_path}: {item.error_message}
                    </p>
                  ))}
                </div>
              )}
            </>
          )}
        </article>
      </div>
    </div>
  )
}

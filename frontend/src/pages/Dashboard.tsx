import { useEffect, useState } from 'react'
import { api, type Dashboard } from '../api'

export default function DashboardPage() {
  const [data, setData] = useState<Dashboard | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.dashboard().then(setData).catch((err: Error) => setError(err.message))
    const id = window.setInterval(() => {
      api.dashboard().then(setData).catch(() => undefined)
    }, 3000)
    return () => window.clearInterval(id)
  }, [])

  if (error) return <p className="error">{error}</p>
  if (!data) return <p className="muted">Loading dashboard…</p>

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Dashboard</h2>
          <p>Local camera-trap detections. Auto-accept threshold {Math.round(data.confidence_auto_accept * 100)}%.</p>
        </div>
        <a className="btn" href="#/import">Import folder</a>
      </div>

      <div className="grid stats">
        <article className="card stat gold">
          <div className="value">{data.images.total}</div>
          <div className="label">Images in database</div>
        </article>
        <article className="card stat tiger">
          <div className="value">{data.detections.tiger}</div>
          <div className="label">Tiger detections</div>
        </article>
        <article className="card stat prey">
          <div className="value">{data.detections.prey}</div>
          <div className="label">Prey detections</div>
        </article>
        <article className="card stat">
          <div className="value">{data.review.pending}</div>
          <div className="label">Waiting for review</div>
        </article>
      </div>

      <div className="grid two" style={{ marginTop: 16 }}>
        <article className="card">
          <h3>Class summary</h3>
          <table className="table">
            <thead>
              <tr><th>Class</th><th>Count</th></tr>
            </thead>
            <tbody>
              {['tiger', 'prey', 'rival', 'human'].map((name) => (
                <tr key={name}>
                  <td><span className={`badge ${name}`}>{name}</span></td>
                  <td>{data.detections.by_class[name] ?? 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </article>
        <article className="card">
          <h3>Recent jobs</h3>
          {data.recent_jobs.length === 0 ? (
            <p className="empty">No imports yet. Start with a camera folder.</p>
          ) : (
            <table className="table">
              <thead>
                <tr><th>Job</th><th>Camera</th><th>Status</th><th>Done</th></tr>
              </thead>
              <tbody>
                {data.recent_jobs.map((job) => (
                  <tr key={job.job_id}>
                    <td><a href={`#/jobs?id=${job.job_id}`}>#{job.job_id}</a></td>
                    <td>{job.camera_id}</td>
                    <td><span className={`badge ${job.status === 'completed' ? 'ok' : job.status === 'failed' ? 'bad' : 'warn'}`}>{job.status}</span></td>
                    <td>{job.processed}/{job.total_images}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
      </div>
    </div>
  )
}

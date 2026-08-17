import { useEffect, useMemo, useState } from 'react'
import { api, type CameraRow, type Dashboard, type GraphPayload, type TigerRow } from '../api'
import { UI_IMAGES } from '../ui/assets'
import { ErrorPanel, LoadingPanel } from '../ui/Status'

export default function DashboardPage() {
  const [data, setData] = useState<Dashboard | null>(null)
  const [cameras, setCameras] = useState<CameraRow[]>([])
  const [tigers, setTigers] = useState<TigerRow[]>([])
  const [graph, setGraph] = useState<GraphPayload | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const [dash, cam, tigerBody, graphBody] = await Promise.all([
          api.dashboard(),
          api.cameras(),
          api.tigers(),
          api.graph(),
        ])
        if (cancelled) return
        setData(dash)
        setCameras(cam.cameras)
        setTigers(tigerBody.tigers)
        setGraph(graphBody)
        setError(null)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      }
    }
    load()
    const id = window.setInterval(async () => {
      try {
        const dash = await api.dashboard()
        if (!cancelled) setData(dash)
      } catch {
        /* keep last successful snapshot */
      }
    }, 10000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  const activeCameras = cameras.filter((item) => item.enabled !== 0 && item.enabled !== false).length
  const latestPrediction = useMemo(
    () => (graph?.gnn.predictions ?? []).find((item) => item.available) ?? null,
    [graph],
  )
  const busiest = useMemo(() => {
    const stations = graph?.occupancy?.stations ?? []
    if (stations.length === 0) return null
    return [...stations].sort((a, b) => b.all_species_detections - a.all_species_detections)[0]
  }, [graph])
  const recentEvents = graph?.observation_graph.events.slice(-5).reverse() ?? []

  if (error && !data) return <ErrorPanel detail={error} />
  if (!data) return <LoadingPanel title="Reading field station…" detail="Loading cameras, detections, and recent activity." />

  return (
    <div>
      <article className="card hero">
        <img className="hero-photo" src={UI_IMAGES.mist} alt="" />
        <div className="hero-shade" />
        <div className="hero-copy">
          <p className="hero-kicker">Wildlife Intelligence</p>
          <h2>Camera-trap field station</h2>
          <p>Offline detection, local tiger identities, and observed movement on OpenStreetMap.</p>
          <div className="actions" style={{ marginTop: 16 }}>
            <a className="btn" href="#/import">Import camera folder</a>
            <a className="btn ghost" href="#/graph">Open movement map</a>
          </div>
        </div>
      </article>

      <div className="grid stats" style={{ marginTop: 16 }}>
        <article className="card stat gold">
          <div className="value">{cameras.length}</div>
          <div className="label">Total cameras</div>
        </article>
        <article className="card stat">
          <div className="value">{activeCameras}</div>
          <div className="label">Active cameras</div>
        </article>
        <article className="card stat tiger">
          <div className="value">{data.detections.total}</div>
          <div className="label">Wildlife detections</div>
        </article>
        <article className="card stat gold">
          <div className="value">{tigers.length || data.tigers.known}</div>
          <div className="label">Tigers identified</div>
        </article>
      </div>

      <div className="grid insight-grid" style={{ marginTop: 16 }}>
        <article className="card">
          <h3>Current tiger activity</h3>
          <table className="table">
            <tbody>
              <tr><td>Tiger detections</td><td>{data.detections.tiger}</td></tr>
              <tr><td>Review queue</td><td>{data.review.pending + (data.review.unidentified_tigers ?? 0)}</td></tr>
              <tr><td>Known field IDs</td><td>{data.tigers.known}</td></tr>
            </tbody>
          </table>
        </article>
        <article className="card">
          <h3>Highest observed activity</h3>
          {!busiest || busiest.all_species_detections === 0 ? (
            <p className="muted">No detections yet. Activity is counted from stored observations only.</p>
          ) : (
            <div>
              <p className="status-title" style={{ marginBottom: 6 }}>{busiest.camera_id}</p>
              <p className="muted">{busiest.all_species_detections} observations · {busiest.tiger_captures} tiger</p>
            </div>
          )}
        </article>
        <article className="card">
          <h3>Latest GNN prediction</h3>
          {!latestPrediction ? (
            <p className="muted">Insufficient data for prediction. A tiger needs five identified camera observations.</p>
          ) : (
            <div>
              <p className="status-title" style={{ marginBottom: 6 }}>
                {latestPrediction.tiger_id} → {latestPrediction.predicted_camera_id}
              </p>
              <p className="muted">Predicted movement · {Math.round((latestPrediction.confidence ?? 0) * 100)}% model confidence</p>
            </div>
          )}
        </article>
      </div>

      <div className="grid two" style={{ marginTop: 16 }}>
        <article className="card">
          <h3>Recent observations</h3>
          {recentEvents.length === 0 ? (
            <p className="muted">Identified tiger observations will appear here after review.</p>
          ) : (
            <table className="table">
              <thead>
                <tr><th>Tiger</th><th>Camera</th><th>Time</th></tr>
              </thead>
              <tbody>
                {recentEvents.map((event, index) => (
                  <tr key={`${event.tiger_id}-${event.camera_id}-${index}`}>
                    <td>{event.tiger_id ?? 'unidentified'}</td>
                    <td>{event.camera_id}</td>
                    <td>{event.timestamp ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
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
      </div>
    </div>
  )
}

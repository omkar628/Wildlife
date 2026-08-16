import { useEffect, useMemo, useState } from 'react'
import { api, type GraphPayload, type GnnPrediction } from '../api'

function percent(value: number | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}%`
}

function PredictionPanel({ prediction }: { prediction: GnnPrediction }) {
  if (!prediction.available) {
    return (
      <p className="empty">
        {prediction.detail || prediction.reason || 'Need 5 identified camera observations for GNN prediction.'}
      </p>
    )
  }

  return (
    <div>
      <p className="muted" style={{ marginTop: 0 }}>
        Predicted next camera{' '}
        <strong style={{ color: 'var(--gold)' }}>{prediction.predicted_camera_id}</strong>
        {' · '}
        confidence {percent(prediction.confidence)}
      </p>
      {prediction.feature_degraded ? (
        <p className="error" style={{ fontSize: 13 }}>
          Feature-degraded: some environmental fields used documented defaults. Ranking is still from the trained GNN.
        </p>
      ) : null}
      <table className="table">
        <thead>
          <tr><th>Rank</th><th>Camera</th><th>Confidence</th></tr>
        </thead>
        <tbody>
          {(prediction.ranked_candidates ?? []).map((item) => (
            <tr key={`${item.rank}-${item.camera_id}`}>
              <td>{item.rank}</td>
              <td>{item.camera_id}</td>
              <td>
                <div className="progress" style={{ minWidth: 80 }}>
                  <span style={{ width: `${Math.round(item.confidence * 100)}%` }} />
                </div>
                <span className="muted" style={{ marginLeft: 8 }}>{percent(item.confidence)}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3 style={{ marginTop: 18 }}>Recent movement</h3>
      <table className="table">
        <thead>
          <tr><th>#</th><th>Camera</th><th>Time</th></tr>
        </thead>
        <tbody>
          {(prediction.history ?? []).map((item, index) => (
            <tr key={`${item.camera_id}-${item.timestamp}-${index}`}>
              <td>{index + 1}</td>
              <td>{item.camera_id}</td>
              <td>{item.timestamp ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ marginTop: 12 }}>
        Predicted at {prediction.prediction_timestamp ?? '—'}
      </p>
    </div>
  )
}

export default function GraphPage() {
  const [data, setData] = useState<GraphPayload | null>(null)
  const [selected, setSelected] = useState<string>('')

  useEffect(() => {
    api.graph().then((payload) => {
      setData(payload)
      const first = payload.gnn.predictions?.[0]?.tiger_id
      if (first) setSelected(first)
    })
  }, [])

  const predictions = data?.gnn.predictions ?? []
  const current = useMemo(
    () => predictions.find((item) => item.tiger_id === selected) ?? predictions[0] ?? null,
    [predictions, selected],
  )

  if (!data) return <p className="muted">Loading graph…</p>
  const nodes = data.camera_graph.nodes
  const edges = data.camera_graph.edges
  const events = data.observation_graph.events
  const gnn = data.gnn

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Observation graph</h2>
          <p>Camera events and next-camera ranking from the local DistanceAware GNN.</p>
        </div>
      </div>

      <article className="card">
        <h3>GNN status</h3>
        <table className="table">
          <tbody>
            <tr>
              <td>Model</td>
              <td>
                <span className={`badge ${gnn.loaded ? 'ok' : 'warn'}`}>
                  {gnn.loaded ? 'Loaded' : 'Unavailable'}
                </span>
              </td>
            </tr>
            <tr><td>Device</td><td>{gnn.device || '—'}</td></tr>
            <tr><td>Version</td><td>{gnn.version || '—'}</td></tr>
            <tr><td>Weights</td><td className="muted">{gnn.path || '—'}</td></tr>
            {gnn.reason ? (
              <tr><td>Reason</td><td className="error">{gnn.reason}</td></tr>
            ) : null}
          </tbody>
        </table>
      </article>

      <article className="card" style={{ marginTop: 16 }}>
        <h3>GNN prediction</h3>
        {predictions.length === 0 ? (
          <p className="empty">Need 5 identified camera observations for GNN prediction.</p>
        ) : (
          <>
            <div className="field" style={{ maxWidth: 240, marginBottom: 16 }}>
              <label>Tiger</label>
              <select value={selected} onChange={(event) => setSelected(event.target.value)}>
                {predictions.map((item) => (
                  <option key={item.tiger_id} value={item.tiger_id}>
                    {item.tiger_id}
                  </option>
                ))}
              </select>
            </div>
            {current ? <PredictionPanel prediction={current} /> : null}
          </>
        )}
      </article>

      <article className="card" style={{ marginTop: 16 }}>
        <h3>Cameras</h3>
        {nodes.length === 0 ? (
          <p className="empty">Import at least one camera folder to see nodes.</p>
        ) : (
          <div className="graph-nodes">
            {nodes.map((node, index) => (
              <div key={node.camera_id} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div className="node">{node.camera_id}</div>
                {index < nodes.length - 1 && <div className="edge">─────→</div>}
              </div>
            ))}
          </div>
        )}
      </article>
      <div className="grid two" style={{ marginTop: 16 }}>
        <article className="card">
          <h3>Movement edges</h3>
          {edges.length === 0 ? (
            <p className="empty">Edges appear after the same tiger is identified at two cameras.</p>
          ) : (
            <table className="table">
              <thead><tr><th>From</th><th>To</th><th>Tiger</th><th>Weight</th></tr></thead>
              <tbody>
                {edges.map((edge, index) => (
                  <tr key={`${edge.source}-${edge.target}-${index}`}>
                    <td>{edge.source}</td>
                    <td>{edge.target}</td>
                    <td>{edge.tiger_id}</td>
                    <td>{edge.weight}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
        <article className="card">
          <h3>Recent events</h3>
          {events.length === 0 ? (
            <p className="empty">Tiger observations will land here as JSON events for the timeline.</p>
          ) : (
            <table className="table">
              <thead><tr><th>Tiger</th><th>Camera</th><th>Time</th></tr></thead>
              <tbody>
                {events.slice(0, 12).map((event, index) => (
                  <tr key={index}>
                    <td>{event.tiger_id ?? 'unidentified'}</td>
                    <td>{event.camera_id}</td>
                    <td>{event.timestamp ?? '—'}</td>
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

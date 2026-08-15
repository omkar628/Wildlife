import { useEffect, useState } from 'react'
import { api, type GraphPayload } from '../api'

export default function GraphPage() {
  const [data, setData] = useState<GraphPayload | null>(null)

  useEffect(() => {
    api.graph().then(setData)
  }, [])

  if (!data) return <p className="muted">Loading graph…</p>
  const nodes = data.camera_graph.nodes
  const edges = data.camera_graph.edges
  const events = data.observation_graph.events

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Observation graph</h2>
          <p>Structured camera events for later animation and the GNN. No model is running here yet.</p>
        </div>
      </div>
      <article className="card">
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
      <p className="muted" style={{ marginTop: 16 }}>
        GNN status: {data.gnn.implemented ? 'connected' : 'not implemented — drop the model into models/gnn later'}.
      </p>
    </div>
  )
}

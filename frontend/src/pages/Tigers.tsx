import { useEffect, useState } from 'react'
import { api, cropUrl, type GnnPrediction, type TigerDetail, type TigerRow } from '../api'

function percent(value: number | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}%`
}

export default function TigersPage() {
  const [items, setItems] = useState<TigerRow[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<TigerDetail | null>(null)
  const [prediction, setPrediction] = useState<GnnPrediction | null>(null)
  const [moveId, setMoveId] = useState('')
  const [error, setError] = useState<string | null>(null)

  async function loadList(preferred?: string | null) {
    const body = await api.tigers()
    setItems(body.tigers)
    const next = preferred && body.tigers.some((item) => item.tiger_id === preferred)
      ? preferred
      : body.tigers[0]?.tiger_id ?? null
    setSelected(next)
  }

  useEffect(() => {
    loadList().catch((err: Error) => setError(err.message))
  }, [])

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      setPrediction(null)
      return
    }
    setError(null)
    api.tiger(selected).then(setDetail).catch((err: Error) => setError(err.message))
    api.gnnPrediction(selected)
      .then(setPrediction)
      .catch((err: Error) => setError(err.message))
  }, [selected])

  async function reassign(observationId: number) {
    if (!moveId) return
    setError(null)
    try {
      await api.assignIdentity(observationId, { action: 'assign', tiger_id: moveId })
      await loadList(moveId)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Tigers</h2>
          <p>Human-confirmed field IDs (T001…). Encoder matching is off. ATRW gallery IDs are never used.</p>
        </div>
      </div>
      <div className="grid two">
        <article className="card">
          {items.length === 0 ? (
            <p className="empty">
              No field tigers yet. On Human review, assign or create a local ID for a tiger crop.
            </p>
          ) : (
            <table className="table">
              <thead>
                <tr><th>Tiger</th><th>First seen</th><th>Last seen</th><th>Observations</th></tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr
                    key={item.tiger_id}
                    onClick={() => setSelected(item.tiger_id)}
                    style={{ cursor: 'pointer', background: selected === item.tiger_id ? 'rgba(224,161,74,0.08)' : undefined }}
                  >
                    <td>{item.tiger_id}</td>
                    <td>{item.first_seen}</td>
                    <td>{item.last_seen}</td>
                    <td>{item.observation_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
        <article className="card">
          <h3>Identity history</h3>
          {!selected || !detail ? (
            <p className="empty">Select a field tiger to see cameras, times, and reference crops.</p>
          ) : (
            <div>
              <p className="muted" style={{ marginTop: 0 }}>
                {detail.tiger.tiger_id} · {detail.history.length} identified observations
              </p>
              {detail.references.length > 0 ? (
                <div className="thumb-grid" style={{ marginBottom: 16 }}>
                  {detail.references.map((ref) => (
                    <article className="thumb" key={ref.observation_id}>
                      <img src={cropUrl(ref.observation_id)} alt={ref.camera_id ?? ''} />
                      <div className="meta">
                        <strong>{ref.camera_id ?? '—'}</strong>
                        <div className="muted">{ref.timestamp ?? '—'}</div>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="muted">No reference crops stored for this ID.</p>
              )}
              <table className="table">
                <thead>
                  <tr><th>Camera</th><th>Time</th><th>Obs</th><th>Fix</th></tr>
                </thead>
                <tbody>
                  {detail.history.map((event) => (
                    <tr key={event.observation_id}>
                      <td>{event.camera_id}</td>
                      <td>{event.timestamp ?? '—'}</td>
                      <td>#{event.observation_id}</td>
                      <td>
                        <select value={moveId} onChange={(e) => setMoveId(e.target.value)}>
                          <option value="">Move to…</option>
                          {items.filter((item) => item.tiger_id !== selected).map((item) => (
                            <option key={item.tiger_id} value={item.tiger_id}>{item.tiger_id}</option>
                          ))}
                        </select>
                        <button
                          className="btn small ghost"
                          type="button"
                          style={{ marginLeft: 8 }}
                          disabled={!moveId}
                          onClick={() => reassign(event.observation_id)}
                        >
                          Reassign
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>
      </div>

      <article className="card" style={{ marginTop: 16 }}>
        <h3>GNN prediction</h3>
        {!selected ? (
          <p className="empty">Need 5 identified camera observations for GNN prediction.</p>
        ) : error ? (
          <p className="error">{error}</p>
        ) : !prediction ? (
          <p className="muted">Loading prediction…</p>
        ) : !prediction.available ? (
          <p className="empty">
            {prediction.detail || prediction.reason || 'Need 5 identified camera observations for GNN prediction.'}
          </p>
        ) : (
          <div>
            <p className="muted" style={{ marginTop: 0 }}>
              Next camera <strong style={{ color: 'var(--gold)' }}>{prediction.predicted_camera_id}</strong>
              {' · '}
              {percent(prediction.confidence)}
            </p>
            {prediction.feature_degraded ? (
              <p className="error" style={{ fontSize: 13 }}>
                Feature-degraded: documented environmental defaults were used.
              </p>
            ) : null}
            <table className="table">
              <thead>
                <tr><th>Rank</th><th>Camera</th><th>Conf</th></tr>
              </thead>
              <tbody>
                {(prediction.ranked_candidates ?? []).map((item) => (
                  <tr key={`${item.rank}-${item.camera_id}`}>
                    <td>{item.rank}</td>
                    <td>{item.camera_id}</td>
                    <td>{percent(item.confidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h3 style={{ marginTop: 18 }}>Recent 5-camera history</h3>
            <table className="table">
              <thead>
                <tr><th>Camera</th><th>Time</th></tr>
              </thead>
              <tbody>
                {(prediction.history ?? []).map((item, index) => (
                  <tr key={`${item.camera_id}-${index}`}>
                    <td>{item.camera_id}</td>
                    <td>{item.timestamp ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </article>
    </div>
  )
}

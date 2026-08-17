import { useCallback, useEffect, useState } from 'react'
import { api, cropUrl, type GnnPrediction, type MovementEdge, type TigerDetail, type TigerRoute, type TigerRow } from '../api'
import StationMap from './StationMap'
import { EmptyPanel, ErrorPanel, LoadingPanel } from '../ui/Status'

function percent(value: number | undefined | null): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}%`
}

export default function TigersPage() {
  const [items, setItems] = useState<TigerRow[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<TigerDetail | null>(null)
  const [prediction, setPrediction] = useState<GnnPrediction | null>(null)
  const [route, setRoute] = useState<TigerRoute | null>(null)
  const [loading, setLoading] = useState(true)
  const [moveId, setMoveId] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [selectedStationId, setSelectedStationId] = useState<string | null>(null)

  async function loadList(preferred?: string | null) {
    const body = await api.tigers()
    setItems(body.tigers)
    const next = preferred && body.tigers.some((item) => item.tiger_id === preferred)
      ? preferred
      : body.tigers[0]?.tiger_id ?? null
    setSelected(next)
  }

  useEffect(() => {
    loadList()
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      setPrediction(null)
      setRoute(null)
      return
    }
    setError(null)
    api.tiger(selected).then(setDetail).catch((err: Error) => setError(err.message))
    api.gnnPrediction(selected).then(setPrediction).catch((err: Error) => setError(err.message))
    api.tigerRoute(selected).then(setRoute).catch(() => setRoute(null))
  }, [selected])

  const onSelectStation = useCallback((cameraId: string) => {
    setSelectedStationId(cameraId)
  }, [])
  const onSelectEdge = useCallback(() => undefined, [])

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

  const currentCamera = route?.last_observed_station ?? detail?.last_camera ?? detail?.history.at(-1)?.camera_id ?? '—'
  const lastSeen = route?.last_observed_timestamp ?? detail?.last_seen ?? detail?.tiger.last_seen ?? '—'
  const camerasVisited = detail?.cameras_visited ?? route?.visited_stations ?? []
  const mostFrequent = detail?.most_frequent_camera ?? route?.most_frequent_camera ?? '—'
  const activityCameras = detail?.activity_area?.cameras ?? route?.activity_area?.cameras ?? []

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Tigers</h2>
          <p>Local field IDs (T001…). MegaDescriptor can auto-assign high-confidence matches. ATRW gallery IDs are never used.</p>
        </div>
      </div>

      {loading ? <LoadingPanel title="Loading tiger identities…" /> : null}

      <div className="grid two">
        <article className="card">
          {items.length === 0 && !loading ? (
            <EmptyPanel
              title="No field tigers yet"
              detail="On Review, assign or create a local ID for a tiger crop."
              action={<a className="btn" href="#/review">Open review</a>}
            />
          ) : (
            <table className="table selectable">
              <thead>
                <tr><th>Tiger</th><th>Last seen</th><th>Observations</th></tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr
                    key={item.tiger_id}
                    className={selected === item.tiger_id ? 'selected' : ''}
                    onClick={() => setSelected(item.tiger_id)}
                  >
                    <td>{item.tiger_id}</td>
                    <td>{item.last_seen}</td>
                    <td>{item.observation_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
        <article className="card">
          <h3>Profile</h3>
          {!selected || !detail ? (
            <p className="muted">Select a field tiger to see its current camera, route, and reference crops.</p>
          ) : (
            <div>
              <p className="status-title">{detail.tiger.tiger_id}</p>
              <div className="actions" style={{ marginBottom: 12 }}>
                <a className="btn small" href={`#/graph?tiger=${encodeURIComponent(detail.tiger.tiger_id)}`}>
                  View on Map
                </a>
              </div>
              <table className="table">
                <tbody>
                  <tr><td>Current / last camera</td><td>{currentCamera}</td></tr>
                  <tr><td>Last seen</td><td>{lastSeen}</td></tr>
                  <tr><td>Total observations</td><td>{detail.observation_count ?? detail.history.length}</td></tr>
                  <tr><td>Cameras visited</td><td>{camerasVisited.join(' → ') || '—'}</td></tr>
                  <tr>
                    <td>Most frequent camera</td>
                    <td>
                      {mostFrequent}
                      {detail.most_frequent_count ? ` · ${detail.most_frequent_count} observations` : ''}
                    </td>
                  </tr>
                  <tr>
                    <td>Observed activity area</td>
                    <td>
                      {activityCameras.length > 0
                        ? activityCameras.map((item) => `${item.camera_id} (${item.observation_count})`).join(', ')
                        : 'Not enough observations'}
                    </td>
                  </tr>
                  <tr>
                    <td>GNN predicted next camera</td>
                    <td>
                      {prediction?.available
                        ? `${prediction.predicted_camera_id} · ${percent(prediction.confidence)}`
                        : 'Insufficient data for prediction.'}
                    </td>
                  </tr>
                  <tr>
                    <td>Prediction confidence</td>
                    <td>{prediction?.available ? percent(prediction.confidence) : '—'}</td>
                  </tr>
                </tbody>
              </table>
              {detail.references.length > 0 ? (
                <div className="thumb-grid" style={{ marginTop: 16 }}>
                  {detail.references.map((ref) => (
                    <article className="thumb" key={ref.observation_id}>
                      <img src={cropUrl(ref.observation_id)} alt={ref.camera_id ?? ''} loading="lazy" />
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
            </div>
          )}
        </article>
      </div>

      {selected && route ? (
        <article className="card" style={{ marginTop: 16 }}>
          <h3>Observed route</h3>
          <p className="muted" style={{ marginTop: 0 }}>
            {route.observed_route.map((stop) => stop.camera_id).join(' → ') || 'No identified movement yet.'}
          </p>
          <StationMap
            compact
            stations={route.occupancy.stations}
            observedRoute={route.observed_route}
            movementEdges={route.observed_route.flatMap((stop, index, all) => {
              const next = all[index + 1]
              if (!next || stop.camera_id === next.camera_id) return []
              const edge: MovementEdge = {
                source: stop.camera_id,
                target: next.camera_id,
                tiger_id: route.tiger_id,
                weight: 1,
                animal_class: 'tiger',
                kind: 'observed',
              }
              return [edge]
            })}
            currentStationId={route.current_station?.camera_id ?? null}
            predictions={route.predictions.ranked_candidates ?? []}
            predictedCameraId={route.predictions.predicted_camera_id ?? null}
            predictionAvailable={Boolean(route.predictions.available)}
            homeRange={route.home_range}
            occupancyMode="selected_tiger"
            showOccupancy
            showHomeRange={false}
            showObserved
            showPredicted
            selectedStationId={selectedStationId}
            selectedEdgeKey={null}
            onSelectStation={onSelectStation}
            onSelectEdge={onSelectEdge}
            activityArea={route.activity_area ?? null}
            showLastSeen
            lastSeenCameraId={route.last_observed_station}
            tigerMarkers={[]}
            showTigerMarkers={false}
          />
        </article>
      ) : null}

      <article className="card" style={{ marginTop: 16 }}>
        <h3>Movement history</h3>
        {!selected || !detail ? (
          <p className="muted">Select a tiger to inspect cameras, times, and Re-ID confidence.</p>
        ) : (
          <table className="table">
            <thead>
              <tr><th>Camera</th><th>Time</th><th>Obs</th><th>Reassign</th></tr>
            </thead>
            <tbody>
              {detail.history.map((event) => (
                <tr key={event.observation_id}>
                  <td>{event.camera_id}</td>
                  <td>{event.timestamp ?? '—'}</td>
                  <td>#{event.observation_id}</td>
                  <td>
                    <select value={moveId} onChange={(e) => setMoveId(e.target.value)} aria-label="Reassign tiger">
                      <option value="">Move to…</option>
                      {items.filter((item) => item.tiger_id !== selected).map((item) => (
                        <option key={item.tiger_id} value={item.tiger_id}>{item.tiger_id}</option>
                      ))}
                    </select>
                    <button className="btn small ghost" type="button" style={{ marginLeft: 8 }} disabled={!moveId} onClick={() => reassign(event.observation_id)}>
                      Reassign
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </article>

      {error ? <ErrorPanel detail={error} /> : null}
    </div>
  )
}

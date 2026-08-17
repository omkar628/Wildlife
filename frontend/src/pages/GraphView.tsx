import { useEffect, useMemo, useState } from 'react'
import {
  api,
  type GraphPayload,
  type GnnPrediction,
  type OccupancyMode,
  type OccupancyStation,
  type ObservedStop,
  type TigerRoute,
  type TigerRow,
} from '../api'
import StationMap, { occupancyLevel } from './StationMap'

const PREDICTION_UNAVAILABLE = 'Prediction unavailable — insufficient data.'

function percent(value: number | undefined | null): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}%`
}

function zoneLabel(zone: string | null | undefined): string {
  if (zone === 'core') return 'Core zone'
  if (zone === 'buffer') return 'Buffer zone'
  if (zone === 'village-adjacent') return 'Village-adjacent zone'
  return 'Not recorded'
}

function occupancyLabel(level: string | undefined): string {
  if (level === 'high') return 'High occupancy'
  if (level === 'medium') return 'Medium occupancy'
  if (level === 'low') return 'Low occupancy'
  return 'No occupancy'
}

function PredictionPanel({ prediction }: { prediction: GnnPrediction }) {
  if (!prediction.available) {
    return (
      <p className="empty">
        {PREDICTION_UNAVAILABLE}
        {prediction.detail ? <span className="muted" style={{ display: 'block', marginTop: 8 }}>{prediction.detail}</span> : null}
      </p>
    )
  }

  return (
    <div>
      <p className="muted" style={{ marginTop: 0 }}>
        Predicted next station{' '}
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
          <tr><th>Rank</th><th>Station</th><th>Confidence</th></tr>
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
          <tr><th>#</th><th>Station</th><th>Time</th></tr>
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

function StationDetails({
  station,
  visit,
  occupancyMode,
}: {
  station: OccupancyStation
  visit: ObservedStop | null
  occupancyMode: OccupancyMode
}) {
  return (
    <table className="table">
      <tbody>
        <tr><td>Station ID</td><td>{station.camera_id}</td></tr>
        <tr>
          <td>Latitude / longitude</td>
          <td>
            {station.latitude != null && station.longitude != null
              ? `${station.latitude.toFixed(5)}, ${station.longitude.toFixed(5)}`
              : 'Not registered'}
          </td>
        </tr>
        <tr><td>Zone type</td><td>{zoneLabel(station.zone_type)}</td></tr>
        {station.habitat ? <tr><td>Habitat</td><td>{station.habitat}</td></tr> : null}
        <tr><td>Total tiger captures</td><td>{station.tiger_captures}</td></tr>
        <tr><td>Unique tigers detected</td><td>{station.unique_tigers}</td></tr>
        <tr>
          <td>Latest tiger detection</td>
          <td>
            {station.latest_tiger_id
              ? `${station.latest_tiger_id} · ${station.latest_tiger_timestamp ?? '—'}`
              : 'None'}
          </td>
        </tr>
        <tr><td>Occupancy level</td><td>{occupancyLabel(occupancyLevel(station, occupancyMode))}</td></tr>
        {station.capture_frequency_per_day != null ? (
          <tr>
            <td>Capture frequency</td>
            <td>{station.capture_frequency_per_day.toFixed(2)} / day</td>
          </tr>
        ) : null}
        {visit ? (
          <>
            <tr><td>Tiger ID</td><td>{visit.tiger_id}</td></tr>
            <tr><td>Observation time</td><td>{visit.timestamp ?? '—'}</td></tr>
            <tr><td>Observation confidence</td><td>{percent(visit.confidence)}</td></tr>
          </>
        ) : null}
      </tbody>
    </table>
  )
}

export default function GraphPage() {
  const [data, setData] = useState<GraphPayload | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [tigers, setTigers] = useState<TigerRow[]>([])
  const [selected, setSelected] = useState<string>('')
  const [route, setRoute] = useState<TigerRoute | null>(null)
  const [routeError, setRouteError] = useState<string | null>(null)
  const [selectedStationId, setSelectedStationId] = useState<string | null>(null)
  const [occupancyMode, setOccupancyMode] = useState<OccupancyMode>('tiger')
  const [showOccupancy, setShowOccupancy] = useState(true)
  const [showHomeRange, setShowHomeRange] = useState(true)

  useEffect(() => {
    api.graph().then((payload) => {
      setData(payload)
      setLoadError(null)
    }).catch((err: Error) => setLoadError(err.message))
    api.tigers().then((body) => {
      setTigers(body.tigers)
      setSelected((current) => current || body.tigers[0]?.tiger_id || '')
    }).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!selected) {
      setRoute(null)
      setRouteError(null)
      return
    }
    let cancelled = false
    setRouteError(null)
    api.tigerRoute(selected)
      .then((body) => {
        if (!cancelled) setRoute(body)
      })
      .catch((err: Error) => {
        if (cancelled) return
        setRoute(null)
        setRouteError(err.message)
      })
    return () => {
      cancelled = true
    }
  }, [selected])

  const predictions = data?.gnn.predictions ?? []
  const snapshotPrediction = useMemo(
    () => predictions.find((item) => item.tiger_id === selected) ?? null,
    [predictions, selected],
  )
  const currentPrediction = useMemo<GnnPrediction | null>(() => {
    if (!route) return snapshotPrediction
    return {
      available: route.predictions.available,
      reason: route.predictions.reason,
      detail: route.predictions.detail,
      tiger_id: route.predictions.tiger_id ?? selected,
      predicted_camera_id: route.predictions.predicted_camera_id,
      confidence: route.predictions.confidence,
      ranked_candidates: (route.predictions.ranked_candidates ?? []).map((item) => ({
        rank: item.rank,
        camera_id: item.camera_id,
        confidence: item.confidence ?? item.score ?? 0,
      })),
      history: route.predictions.history,
      prediction_timestamp: route.predictions.prediction_timestamp,
      feature_degraded: route.predictions.feature_degraded,
    }
  }, [route, snapshotPrediction, selected])

  const stations = useMemo(() => {
    if (route?.occupancy.stations) return route.occupancy.stations
    if (data?.occupancy?.stations) return data.occupancy.stations
    return (data?.camera_graph.nodes ?? []).map((node) => ({
      camera_id: node.camera_id,
      latitude: node.latitude ?? null,
      longitude: node.longitude ?? null,
      registered: true,
      missing_coordinates: node.latitude == null || node.longitude == null,
      zone_type: null,
      habitat: node.habitat ?? null,
      all_species_detections: 0,
      tiger_captures: node.observation_count,
      unique_tigers: 0,
      selected_tiger_captures: 0,
      latest_tiger_id: null,
      latest_tiger_timestamp: null,
      occupancy_level_all_species: 'none',
      occupancy_level_tiger: node.observation_count > 0 ? 'low' : 'none',
      occupancy_level_selected_tiger: 'none',
      capture_frequency_per_day: null,
      capture_span_days: null,
    }))
  }, [route, data])
  const observedRoute = route?.observed_route ?? []
  const currentStationId = route?.current_station?.camera_id ?? null
  const ranked = route?.predictions.ranked_candidates ?? []
  const predictionAvailable = Boolean(route?.predictions.available)
  const predictedCameraId = route?.predictions.predicted_camera_id ?? null

  const selectedStation = stations.find((item) => item.camera_id === selectedStationId) ?? null
  const selectedVisit = useMemo(() => {
    if (!selectedStationId) return null
    const matches = observedRoute.filter((stop) => stop.camera_id === selectedStationId)
    return matches[matches.length - 1] ?? null
  }, [observedRoute, selectedStationId])

  if (!data) {
    return <p className={loadError ? 'error' : 'muted'}>{loadError ?? 'Loading graph…'}</p>
  }
  const nodes = data.camera_graph.nodes
  const edges = data.camera_graph.edges
  const events = data.observation_graph.events
  const gnn = data.gnn
  const predictionMessage = route
    ? (route.predictions.available
      ? (route.predictions.summary || `Likely next station is ${route.predictions.predicted_camera_id}.`)
      : PREDICTION_UNAVAILABLE)
    : (selected ? 'Loading prediction…' : 'Select a tiger to see predicted movement.')

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Pench station map</h2>
          <p>Observed movement, current station, and predicted next station from registered cameras.</p>
        </div>
      </div>

      <article className="card">
        <div className="map-toolbar">
          <div className="field" style={{ minWidth: 180 }}>
            <label>Tiger</label>
            <select value={selected} onChange={(event) => setSelected(event.target.value)}>
              {tigers.length === 0 ? <option value="">No identified tigers</option> : null}
              {tigers.map((item) => (
                <option key={item.tiger_id} value={item.tiger_id}>{item.tiger_id}</option>
              ))}
            </select>
          </div>
          <div className="field" style={{ minWidth: 220 }}>
            <label>Area occupancy</label>
            <select value={occupancyMode} onChange={(event) => setOccupancyMode(event.target.value as OccupancyMode)}>
              <option value="all_species">All species occupancy</option>
              <option value="tiger">Tiger-only occupancy</option>
              <option value="selected_tiger" disabled={!selected}>Selected tiger occupancy</option>
            </select>
          </div>
          <label className="layer-toggle">
            <input type="checkbox" checked={showOccupancy} onChange={(event) => setShowOccupancy(event.target.checked)} />
            Area occupancy
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={showHomeRange} onChange={(event) => setShowHomeRange(event.target.checked)} />
            Estimated home range
          </label>
        </div>

        <StationMap
          stations={stations}
          observedRoute={observedRoute}
          currentStationId={currentStationId}
          predictions={ranked}
          predictedCameraId={predictedCameraId}
          predictionAvailable={predictionAvailable}
          homeRange={route?.home_range ?? null}
          occupancyMode={occupancyMode}
          showOccupancy={showOccupancy}
          showHomeRange={showHomeRange}
          selectedStationId={selectedStationId}
          onSelectStation={setSelectedStationId}
        />

        <div className="map-legend">
          <span><i className="swatch route-obs" /> Observed route</span>
          <span><i className="swatch current" /> Current station</span>
          <span><i className="swatch predicted" /> Predicted next station</span>
          <span><i className="swatch candidate" /> Predicted candidate</span>
          <span><i className="swatch station" /> Camera station</span>
          <span><i className="swatch high" /> High occupancy</span>
          <span><i className="swatch low" /> Low occupancy</span>
          <span><i className="swatch core" /> Core zone</span>
          <span><i className="swatch buffer" /> Buffer zone</span>
          <span><i className="swatch village" /> Village-adjacent zone</span>
          <span><i className="swatch range" /> Estimated home range</span>
        </div>
      </article>

      <div className="grid two" style={{ marginTop: 16 }}>
        <article className="card">
          <h3>Tiger information</h3>
          {!selected ? (
            <p className="empty">Select a tiger to see its observed route.</p>
          ) : !route ? (
            <p className="muted">{routeError ? `Map still available. ${PREDICTION_UNAVAILABLE}` : 'Loading tiger route…'}</p>
          ) : (
            <table className="table">
              <tbody>
                <tr><td>Tiger ID</td><td>{route.tiger_id}</td></tr>
                <tr><td>Total observations</td><td>{route.observation_count}</td></tr>
                <tr><td>Last observed station</td><td>{route.last_observed_station ?? '—'}</td></tr>
                <tr><td>Last observed timestamp</td><td>{route.last_observed_timestamp ?? '—'}</td></tr>
                <tr><td>Visited stations</td><td>{route.visited_stations.join(' → ') || '—'}</td></tr>
                <tr>
                  <td>Observed route</td>
                  <td>{route.observed_route.map((stop) => stop.camera_id).join(' → ') || '—'}</td>
                </tr>
                <tr>
                  <td>Estimated home range</td>
                  <td>
                    {route.home_range.available
                      ? `${route.home_range.label} · ${route.home_range.unique_stations} stations`
                      : route.home_range.reason || 'Not enough observations'}
                  </td>
                </tr>
                <tr>
                  <td>Predicted next station</td>
                  <td>
                    {route.predictions.available
                      ? `${route.predictions.predicted_camera_id} · ${percent(route.predictions.confidence)}`
                      : PREDICTION_UNAVAILABLE}
                  </td>
                </tr>
              </tbody>
            </table>
          )}
          {route?.predictions.available && (route.predictions.ranked_candidates?.length ?? 0) > 0 ? (
            <div style={{ marginTop: 12 }}>
              <p className="muted" style={{ marginTop: 0 }}>{route.predictions.summary}</p>
              <table className="table">
                <thead>
                  <tr><th>Rank</th><th>Predicted station</th><th>Score</th></tr>
                </thead>
                <tbody>
                  {route.predictions.ranked_candidates?.map((item) => (
                    <tr key={`${item.rank}-${item.camera_id}`}>
                      <td>{item.rank}</td>
                      <td>{item.camera_id}{item.rank === 1 ? ' · Predicted next station' : ''}</td>
                      <td>{percent(item.score ?? item.confidence)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : route && !route.predictions.available ? (
            <p className="empty" style={{ marginBottom: 0 }}>
              {PREDICTION_UNAVAILABLE}
              {route.predictions.detail ? (
                <span className="muted" style={{ display: 'block', marginTop: 8 }}>{route.predictions.detail}</span>
              ) : null}
            </p>
          ) : null}
        </article>

        <article className="card">
          <h3>Station information</h3>
          {!selectedStation ? (
            <p className="empty">Click a station on the map for camera ID, coordinates, and occupancy.</p>
          ) : (
            <StationDetails
              station={selectedStation}
              visit={selectedVisit}
              occupancyMode={occupancyMode}
            />
          )}
          <p className="muted" style={{ marginBottom: 0, marginTop: 14 }}>
            {predictionMessage}
          </p>
          {occupancyMode === 'selected_tiger' && !selected ? (
            <p className="empty">Select a tiger to see selected-tiger occupancy.</p>
          ) : null}
        </article>
      </div>

      <article className="card" style={{ marginTop: 16 }}>
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
        <h3>GNN ranking details</h3>
        {currentPrediction ? (
          <PredictionPanel prediction={currentPrediction} />
        ) : selected && route && !route.predictions.available ? (
          <p className="empty">{PREDICTION_UNAVAILABLE}</p>
        ) : (
          <p className="empty">{PREDICTION_UNAVAILABLE}</p>
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

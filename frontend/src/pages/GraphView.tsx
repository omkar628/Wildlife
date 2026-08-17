import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  api,
  cropUrl,
  imageUrl,
  type GraphPayload,
  type GnnPrediction,
  type MovementEdge,
  type OccupancyMode,
  type OccupancyStation,
  type ObservedStop,
  type TigerRoute,
  type TigerRow,
  type WildlifeClassNode,
} from '../api'
import StationMap, { occupancyLevel } from './StationMap'
import { ErrorPanel, LoadingPanel } from '../ui/Status'

const PREDICTION_UNAVAILABLE = 'Insufficient data for prediction.'
const INSUFFICIENT = 'Insufficient data for prediction.'

function percent(value: number | undefined | null): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}%`
}

function occupancyLabel(level: string | undefined): string {
  if (level === 'high') return 'High observed activity'
  if (level === 'medium') return 'Medium observed activity'
  if (level === 'low') return 'Low observed activity'
  return 'No observed activity'
}

function tigerFromHash(): string {
  const query = window.location.hash.split('?')[1] || ''
  return new URLSearchParams(query).get('tiger') || ''
}

function classLabel(value: string | null | undefined): string {
  if (value === 'tiger') return 'Tiger'
  if (value === 'prey') return 'Prey'
  if (value === 'rival') return 'Rival'
  if (value === 'human') return 'Human'
  return value || 'Unknown'
}

function PredictionPanel({ prediction }: { prediction: GnnPrediction }) {
  if (!prediction.available) {
    return (
      <p className="empty">
        {INSUFFICIENT}
        {prediction.detail ? <span className="muted" style={{ display: 'block', marginTop: 8 }}>{prediction.detail}</span> : null}
      </p>
    )
  }

  return (
    <div>
      <p className="muted" style={{ marginTop: 0 }}>
        GNN predicted movement to{' '}
        <strong style={{ color: '#8fd0ff' }}>{prediction.predicted_camera_id}</strong>
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
              <td>{item.camera_id}{item.rank === 1 ? ' · predicted' : ''}</td>
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
        <tr><td>Camera ID</td><td>{station.camera_id}</td></tr>
        <tr><td>Name</td><td>{station.name || station.camera_id}</td></tr>
        <tr><td>Status</td><td>{station.status || (station.enabled === false ? 'disabled' : 'enabled')}</td></tr>
        <tr>
          <td>Latitude / longitude</td>
          <td>
            {station.latitude != null && station.longitude != null
              ? `${station.latitude.toFixed(5)}, ${station.longitude.toFixed(5)}`
              : 'Not registered'}
          </td>
        </tr>
        {station.habitat ? <tr><td>Habitat</td><td>{station.habitat}</td></tr> : null}
        <tr><td>All observations</td><td>{station.all_species_detections}</td></tr>
        <tr><td>Tiger observations</td><td>{station.tiger_detections ?? station.tiger_captures}</td></tr>
        <tr><td>Prey observations</td><td>{station.prey_detections ?? 0}</td></tr>
        <tr><td>Rival observations</td><td>{station.rival_detections ?? 0}</td></tr>
        <tr><td>Human observations</td><td>{station.human_detections ?? 0}</td></tr>
        <tr><td>Unique tigers</td><td>{station.unique_tigers}</td></tr>
        <tr>
          <td>Latest tiger</td>
          <td>
            {station.latest_tiger_id
              ? `${station.latest_tiger_id} · ${station.latest_tiger_timestamp ?? '—'}`
              : 'None'}
          </td>
        </tr>
        <tr><td>Observed activity</td><td>{occupancyLabel(occupancyLevel(station, occupancyMode))}</td></tr>
        {visit ? (
          <>
            <tr><td>Selected tiger</td><td>{visit.tiger_id}</td></tr>
            <tr><td>Observation time</td><td>{visit.timestamp ?? '—'}</td></tr>
            <tr><td>Confidence</td><td>{percent(visit.confidence)}</td></tr>
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
  const [cameras, setCameras] = useState<string[]>([])
  const [selected, setSelected] = useState<string>(tigerFromHash())
  const [cameraFilter, setCameraFilter] = useState<string>('')
  const [route, setRoute] = useState<TigerRoute | null>(null)
  const [routeError, setRouteError] = useState<string | null>(null)
  const [selectedStationId, setSelectedStationId] = useState<string | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<MovementEdge | null>(null)
  const [selectedEdgeKey, setSelectedEdgeKey] = useState<string | null>(null)
  const [occupancyMode, setOccupancyMode] = useState<OccupancyMode>('tiger')
  const [showOccupancy, setShowOccupancy] = useState(true)
  const [showActivityRegion, setShowActivityRegion] = useState(true)
  const [showObserved, setShowObserved] = useState(true)
  const [showPredicted, setShowPredicted] = useState(true)
  const [showLastSeen, setShowLastSeen] = useState(true)
  const [classFilters, setClassFilters] = useState({ tiger: true, prey: true, rival: true, human: true })

  useEffect(() => {
    api.tigers().then((body) => {
      setTigers(body.tigers)
    }).catch(() => undefined)
    api.cameras().then((body) => {
      setCameras(body.cameras.map((item) => item.camera_id))
    }).catch(() => undefined)
    const onHash = () => {
      const fromHash = tigerFromHash()
      if (fromHash) setSelected(fromHash)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    let cancelled = false
    api.graph({
      tiger_id: selected || undefined,
      animal_class: selected ? 'tiger' : undefined,
      camera_id: cameraFilter || undefined,
    }).then((payload) => {
      if (cancelled) return
      setData(payload)
      setLoadError(null)
    }).catch((err: Error) => {
      if (!cancelled) setLoadError(err.message)
    })
    return () => {
      cancelled = true
    }
  }, [selected, cameraFilter])

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
    if (route?.occupancy.stations && !cameraFilter) {
      return route.occupancy.stations
    }
    if (data?.occupancy?.stations) return data.occupancy.stations
    return (data?.camera_graph.nodes ?? []).map((node) => ({
      camera_id: node.camera_id,
      name: node.name ?? node.camera_id,
      enabled: node.enabled,
      status: node.status,
      latitude: node.latitude ?? null,
      longitude: node.longitude ?? null,
      registered: true,
      missing_coordinates: node.latitude == null || node.longitude == null,
      zone_type: null,
      habitat: node.habitat ?? null,
      all_species_detections: 0,
      tiger_captures: node.observation_count,
      tiger_detections: node.tiger_count ?? 0,
      prey_detections: node.prey_count ?? 0,
      rival_detections: node.rival_count ?? 0,
      human_detections: node.human_count ?? 0,
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
  }, [route, data, cameraFilter])

  const observedRoute = route?.observed_route ?? []
  const movementEdges = useMemo(() => {
    const edges = data?.movement_edges ?? []
    return edges.filter((edge) => {
      if (selected) {
        return edge.tiger_id === selected && (edge.animal_class ?? 'tiger') === 'tiger'
      }
      const animal = (edge.animal_class || 'tiger') as keyof typeof classFilters
      if (animal in classFilters && !classFilters[animal]) return false
      return true
    })
  }, [data?.movement_edges, selected, classFilters])
  const wildlifeNodes = useMemo(() => {
    const entities = data?.wildlife_entities
    if (!entities) return [] as WildlifeClassNode[]
    const nodes: WildlifeClassNode[] = []
    if (!selected && classFilters.prey) nodes.push(...entities.prey)
    if (!selected && classFilters.rival) nodes.push(...entities.rival)
    if (!selected && classFilters.human) nodes.push(...entities.human)
    return nodes
  }, [data?.wildlife_entities, classFilters, selected])
  const tigerMarkers = useMemo(() => {
    const all = data?.wildlife_entities?.tigers ?? []
    if (!classFilters.tiger) return []
    return selected ? all.filter((item) => item.tiger_id === selected) : all
  }, [data?.wildlife_entities, selected, classFilters.tiger])
  const currentStationId = route?.current_station?.camera_id ?? null
  const lastSeenCameraId = route?.last_observed_station ?? currentStationId
  const focusPoint = useMemo(() => {
    const match = tigerMarkers.find((item) => item.tiger_id === selected)
    if (match?.latitude != null && match.longitude != null) {
      return { latitude: match.latitude, longitude: match.longitude }
    }
    return null
  }, [tigerMarkers, selected])
  const ranked = route?.predictions.ranked_candidates ?? []
  const predictionAvailable = Boolean(route?.predictions.available)
  const predictedCameraId = route?.predictions.predicted_camera_id ?? null

  const selectedStation = stations.find((item) => item.camera_id === selectedStationId) ?? null
  const selectedVisit = useMemo(() => {
    if (!selectedStationId) return null
    const matches = observedRoute.filter((stop) => stop.camera_id === selectedStationId)
    return matches[matches.length - 1] ?? null
  }, [observedRoute, selectedStationId])

  const onSelectStation = useCallback((cameraId: string) => {
    setSelectedStationId(cameraId)
    setSelectedEdge(null)
    setSelectedEdgeKey(null)
  }, [])
  const onSelectEdge = useCallback((edge: MovementEdge, key: string) => {
    setSelectedEdge(edge)
    setSelectedEdgeKey(key)
  }, [])

  if (loadError && !data) {
    return <ErrorPanel title="Movement graph unavailable" detail={loadError} />
  }
  if (!data) {
    return <LoadingPanel title="Building movement graph…" detail="Joining cameras, observations, and GNN status." />
  }
  const events = data.observation_graph.events
  const gnn = data.gnn

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Movement map</h2>
          <p>OpenStreetMap locked to Pench Tiger Reserve. Cameras use stored coordinates. Solid lines are observed movement; dashed lines are GNN predictions.</p>
        </div>
      </div>

      <article className="card">
        <div className="story-row" aria-label="How to read this map">
          <span className="story-chip">1. OpenStreetMap</span>
          <span className="story-chip">2. Camera network</span>
          <span className="story-chip">3. <strong>Solid</strong> observed movement</span>
          <span className="story-chip">4. Observed activity</span>
          <span className="story-chip">5. <strong>Dashed</strong> GNN prediction</span>
        </div>
        <div className="map-toolbar">
          <div className="chip-row" role="group" aria-label="Wildlife class filters">
            <button
              type="button"
              className={`filter-chip ${classFilters.tiger && classFilters.prey && classFilters.rival && classFilters.human ? 'active' : ''}`}
              onClick={() => setClassFilters({ tiger: true, prey: true, rival: true, human: true })}
            >
              All
            </button>
            {(['tiger', 'prey', 'rival', 'human'] as const).map((key) => (
              <label key={key} className={`filter-chip ${classFilters[key] ? 'active' : ''} ${key}`}>
                <input
                  type="checkbox"
                  checked={classFilters[key]}
                  onChange={(event) => setClassFilters((current) => ({ ...current, [key]: event.target.checked }))}
                />
                {key === 'tiger' ? 'Tigers' : key === 'prey' ? 'Prey' : key === 'rival' ? 'Rivals' : 'Humans'}
              </label>
            ))}
          </div>
          <div className="field" style={{ minWidth: 160 }}>
            <label>Select tiger</label>
            <select value={selected} onChange={(event) => setSelected(event.target.value)}>
              <option value="">All Tigers</option>
              {tigers.map((item) => (
                <option key={item.tiger_id} value={item.tiger_id}>{item.tiger_id}</option>
              ))}
            </select>
          </div>
          <div className="field" style={{ minWidth: 150 }}>
            <label>Camera</label>
            <select value={cameraFilter} onChange={(event) => setCameraFilter(event.target.value)}>
              <option value="">All cameras</option>
              {cameras.map((id) => (
                <option key={id} value={id}>{id}</option>
              ))}
            </select>
          </div>
          <div className="field" style={{ minWidth: 200 }}>
            <label>Observed activity</label>
            <select value={occupancyMode} onChange={(event) => setOccupancyMode(event.target.value as OccupancyMode)}>
              <option value="all_species">All classes</option>
              <option value="tiger">Tiger occupancy</option>
              <option value="prey">Prey occupancy</option>
              <option value="rival">Rival occupancy</option>
              <option value="human">Human activity</option>
              <option value="selected_tiger" disabled={!selected}>Selected tiger</option>
            </select>
          </div>
          <label className="layer-toggle">
            <input type="checkbox" checked={showObserved} onChange={(event) => setShowObserved(event.target.checked)} />
            Observed movement
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={showPredicted} onChange={(event) => setShowPredicted(event.target.checked)} />
            GNN prediction
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={showOccupancy} onChange={(event) => setShowOccupancy(event.target.checked)} />
            Activity area
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={showLastSeen} onChange={(event) => setShowLastSeen(event.target.checked)} />
            Last seen
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={showActivityRegion} onChange={(event) => setShowActivityRegion(event.target.checked)} />
            Activity region
          </label>
        </div>

        <StationMap
          stations={stations}
          observedRoute={observedRoute}
          movementEdges={movementEdges}
          currentStationId={currentStationId}
          predictions={ranked}
          predictedCameraId={predictedCameraId}
          predictionAvailable={predictionAvailable}
          homeRange={{
            available: Boolean(route?.activity_area?.region?.available),
            reason: route?.activity_area?.region?.reason ?? null,
            label: 'Observed Activity Area',
            polygon: route?.activity_area?.region?.polygon ?? [],
            point_count: route?.activity_area?.region?.point_count ?? 0,
            unique_stations: route?.activity_area?.region?.unique_stations ?? 0,
          }}
          occupancyMode={selected ? 'selected_tiger' : occupancyMode}
          showOccupancy={showOccupancy}
          showHomeRange={showActivityRegion && Boolean(selected)}
          showObserved={showObserved}
          showPredicted={showPredicted && Boolean(selected)}
          selectedStationId={selectedStationId}
          selectedEdgeKey={selectedEdgeKey}
          onSelectStation={onSelectStation}
          onSelectEdge={onSelectEdge}
          tigerMarkers={tigerMarkers}
          wildlifeNodes={wildlifeNodes}
          activityArea={selected ? route?.activity_area ?? null : null}
          showLastSeen={showLastSeen}
          lastSeenCameraId={showLastSeen ? lastSeenCameraId : null}
          showTigerMarkers
          classFilters={classFilters}
          focusPoint={focusPoint}
          onSelectTiger={(tigerId) => setSelected(tigerId)}
        />

        <div className="map-legend">
          <span><i className="swatch station" /> Camera</span>
          <span><i className="swatch route-tiger" /> Tiger</span>
          <span><i className="swatch route-prey" /> Prey</span>
          <span><i className="swatch route-rival" /> Rival</span>
          <span><i className="swatch route-human" /> Human</span>
          <span><i className="swatch route-obs" /> Observed route (solid)</span>
          <span><i className="swatch predicted-dash" /> GNN prediction (dashed)</span>
          <span><i className="swatch high" /> Observed activity area</span>
          <span><i className="swatch current" /> Last seen</span>
        </div>
        {selected && route?.last_observed_station ? (
          <div className="last-seen-banner">
            <strong>{selected}</strong>
            <span>LAST SEEN ● {route.last_observed_station}</span>
            <span>{route.last_observed_timestamp ?? '—'}</span>
            <span>Observations: {route.observation_count}</span>
          </div>
        ) : null}
      </article>

      <div className="grid two" style={{ marginTop: 16 }}>
        <article className="card">
          <h3>Future station</h3>
          {!selected ? (
            <p className="empty">Select a tiger to request a GNN next-station prediction.</p>
          ) : !route ? (
            <p className="muted">{routeError ? INSUFFICIENT : 'Loading prediction…'}</p>
          ) : (
            <table className="table">
              <tbody>
                <tr><td>Tiger</td><td>{route.tiger_id}</td></tr>
                <tr><td>Current camera</td><td>{route.last_observed_station ?? '—'}</td></tr>
                <tr>
                  <td>Predicted next camera</td>
                  <td>
                    {route.predictions.available
                      ? route.predictions.predicted_camera_id
                      : INSUFFICIENT}
                  </td>
                </tr>
                <tr>
                  <td>Confidence</td>
                  <td>{route.predictions.available ? percent(route.predictions.confidence) : '—'}</td>
                </tr>
                <tr><td>Prediction time</td><td>{route.predictions.prediction_timestamp ?? '—'}</td></tr>
                <tr>
                  <td>Last observations</td>
                  <td>{route.observed_route.map((stop) => stop.camera_id).join(' → ') || '—'}</td>
                </tr>
                <tr>
                  <td>Edge type</td>
                  <td>{route.predictions.available ? 'GNN predicted movement (not observed)' : 'No prediction'}</td>
                </tr>
              </tbody>
            </table>
          )}
        </article>

        <article className="card">
          <h3>{selectedEdge ? 'Movement edge' : 'Camera information'}</h3>
          {selectedEdge ? (
            <table className="table">
              <tbody>
                <tr><td>Type</td><td>Observed movement</td></tr>
                <tr><td>Class</td><td>{classLabel(selectedEdge.animal_class)}</td></tr>
                <tr><td>Identity</td><td>{selectedEdge.tiger_id || selectedEdge.identity || '—'}</td></tr>
                <tr><td>Source camera</td><td>{selectedEdge.source}</td></tr>
                <tr><td>Destination camera</td><td>{selectedEdge.target}</td></tr>
                <tr><td>Timestamp</td><td>{selectedEdge.last_timestamp ?? selectedEdge.first_timestamp ?? '—'}</td></tr>
                <tr>
                  <td>Distance</td>
                  <td>{selectedEdge.distance_km != null ? `${selectedEdge.distance_km.toFixed(2)} km` : '—'}</td>
                </tr>
                <tr><td>Supporting observations</td><td>{selectedEdge.weight}</td></tr>
                <tr><td>Confidence</td><td>{percent(selectedEdge.confidence)}</td></tr>
                <tr>
                  <td>Observation IDs</td>
                  <td>{(selectedEdge.observation_ids ?? []).join(', ') || '—'}</td>
                </tr>
              </tbody>
            </table>
          ) : !selectedStation ? (
            <p className="empty">Click a camera or an observed edge on the map.</p>
          ) : (
            <StationDetails
              station={selectedStation}
              visit={selectedVisit}
              occupancyMode={occupancyMode}
            />
          )}
        </article>
      </div>

      <article className="card" style={{ marginTop: 16 }}>
        <h3>Tiger history</h3>
        {!selected ? (
          <p className="empty">Select a tiger to see Camera_01 → Camera_02 → Camera_03 history.</p>
        ) : !route ? (
          <p className="muted">{routeError ?? 'Loading tiger history…'}</p>
        ) : route.observed_route.length === 0 ? (
          <p className="empty">No identified observations for {selected}.</p>
        ) : (
          <div>
            <p className="muted" style={{ marginTop: 0 }}>
              {route.tiger_id}: {route.observed_route.map((stop) => stop.camera_id).join(' → ')}
            </p>
            <div className="history-list">
              {route.observed_route.map((stop, index) => (
                <div key={`${stop.observation_id}-${index}`} className="history-stop">
                  {stop.image_id ? (
                    <img src={imageUrl(stop.image_id)} alt={stop.filename ?? stop.camera_id} />
                  ) : stop.observation_id ? (
                    <img src={cropUrl(stop.observation_id)} alt={stop.camera_id} />
                  ) : (
                    <div className="empty" style={{ padding: 16 }}>No image</div>
                  )}
                  <div>
                    <strong>{stop.camera_id}</strong>
                    <div className="muted">{stop.timestamp ?? '—'}</div>
                    <div>Confidence {percent(stop.confidence)}</div>
                    <div>Re-ID similarity {percent(stop.reid_confidence ?? stop.confidence)}</div>
                    <div>
                      Coordinates{' '}
                      {stop.latitude != null && stop.longitude != null
                        ? `${stop.latitude.toFixed(5)}, ${stop.longitude.toFixed(5)}`
                        : 'not registered'}
                    </div>
                  </div>
                  {index < route.observed_route.length - 1 ? <div className="history-arrow">↓</div> : null}
                </div>
              ))}
            </div>
          </div>
        )}
      </article>

      <div className="grid two" style={{ marginTop: 16 }}>
        <article className="card">
          <h3>GNN ranking</h3>
          {currentPrediction ? (
            <PredictionPanel prediction={currentPrediction} />
          ) : (
            <p className="empty">{PREDICTION_UNAVAILABLE}</p>
          )}
        </article>
        <article className="card">
          <h3>GNN status</h3>
          <table className="table">
            <tbody>
              <tr>
                <td>Model</td>
                <td>
                  <span className={`badge ${gnn.loaded ? 'ok' : 'warn'}`}>
                    {gnn.loaded ? 'Loaded V3.1' : 'Unavailable'}
                  </span>
                </td>
              </tr>
              <tr><td>Device</td><td>{gnn.device || '—'}</td></tr>
              <tr><td>Version</td><td>{gnn.version || '—'}</td></tr>
              <tr><td>Weights</td><td className="muted">{gnn.path || '—'}</td></tr>
              {gnn.reason ? <tr><td>Reason</td><td className="error">{gnn.reason}</td></tr> : null}
            </tbody>
          </table>
        </article>
      </div>

      <article className="card" style={{ marginTop: 16 }}>
        <h3>Observed movement edges</h3>
        {movementEdges.length === 0 ? (
          <p className="empty">Edges appear only from chronological observations. Cameras alone do not create a route.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>From</th>
                <th>To</th>
                <th>Class</th>
                <th>Identity</th>
                <th>Support</th>
                <th>Distance</th>
                <th>Kind</th>
              </tr>
            </thead>
            <tbody>
              {movementEdges.map((edge, index) => (
                <tr
                  key={`${edge.source}-${edge.target}-${edge.animal_class}-${index}`}
                  onClick={() => {
                    setSelectedEdge(edge)
                    setSelectedEdgeKey(`${edge.animal_class ?? 'unknown'}-${edge.source}-${edge.target}-${edge.tiger_id ?? ''}-${index}`)
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  <td>{edge.source}</td>
                  <td>{edge.target}</td>
                  <td>{classLabel(edge.animal_class)}</td>
                  <td>{edge.tiger_id || edge.identity || '—'}</td>
                  <td>{edge.weight}</td>
                  <td>{edge.distance_km != null ? `${edge.distance_km.toFixed(2)} km` : '—'}</td>
                  <td>Observed movement</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </article>

      <article className="card" style={{ marginTop: 16 }}>
        <h3>Recent events</h3>
        {events.length === 0 ? (
          <p className="empty">Tiger observations will land here as they are identified.</p>
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
  )
}

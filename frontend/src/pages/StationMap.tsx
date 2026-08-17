import { useMemo } from 'react'
import type {
  HomeRange,
  OccupancyMode,
  OccupancyStation,
  ObservedStop,
  RankedCandidate,
} from '../api'

const WIDTH = 920
const HEIGHT = 560
const PAD = 56

type Point = { x: number; y: number; lat: number; lon: number }

function hasCoords(item: {
  latitude?: number | null
  longitude?: number | null
}): item is { latitude: number; longitude: number } {
  return item.latitude != null && item.longitude != null && Number.isFinite(item.latitude) && Number.isFinite(item.longitude)
}

function zoneClass(zone: string | null | undefined): string {
  if (zone === 'core') return 'core'
  if (zone === 'buffer') return 'buffer'
  if (zone === 'village-adjacent') return 'village'
  return 'unknown'
}

function occupancyValue(station: OccupancyStation, mode: OccupancyMode): number {
  if (mode === 'all_species') return station.all_species_detections
  if (mode === 'selected_tiger') return station.selected_tiger_captures
  return station.tiger_captures
}

function occupancyLevel(station: OccupancyStation, mode: OccupancyMode): string {
  if (mode === 'all_species') return station.occupancy_level_all_species
  if (mode === 'selected_tiger') return station.occupancy_level_selected_tiger
  return station.occupancy_level_tiger
}

function project(
  lat: number,
  lon: number,
  bounds: { minLat: number; maxLat: number; minLon: number; maxLon: number },
): Point {
  const latSpan = Math.max(bounds.maxLat - bounds.minLat, 0.01)
  const lonSpan = Math.max(bounds.maxLon - bounds.minLon, 0.01)
  return {
    x: PAD + ((lon - bounds.minLon) / lonSpan) * (WIDTH - PAD * 2),
    y: PAD + ((bounds.maxLat - lat) / latSpan) * (HEIGHT - PAD * 2),
    lat,
    lon,
  }
}

function linePoints(points: Point[]): string {
  return points.map((point) => `${point.x},${point.y}`).join(' ')
}

type StationMapProps = {
  stations: OccupancyStation[]
  observedRoute: ObservedStop[]
  currentStationId: string | null
  predictions: RankedCandidate[]
  predictedCameraId: string | null
  predictionAvailable: boolean
  homeRange: HomeRange | null
  occupancyMode: OccupancyMode
  showOccupancy: boolean
  showHomeRange: boolean
  selectedStationId: string | null
  onSelectStation: (cameraId: string) => void
}

export default function StationMap({
  stations,
  observedRoute,
  currentStationId,
  predictions,
  predictedCameraId,
  predictionAvailable,
  homeRange,
  occupancyMode,
  showOccupancy,
  showHomeRange,
  selectedStationId,
  onSelectStation,
}: StationMapProps) {
  const plottable = useMemo(() => {
    const coords: Array<{ lat: number; lon: number }> = []
    for (const station of stations) {
      if (hasCoords(station)) {
        coords.push({ lat: station.latitude, lon: station.longitude })
      }
    }
    for (const stop of observedRoute) {
      if (hasCoords(stop)) {
        coords.push({ lat: stop.latitude, lon: stop.longitude })
      }
    }
    for (const candidate of predictions) {
      if (hasCoords(candidate)) {
        coords.push({ lat: candidate.latitude, lon: candidate.longitude })
      }
    }
    return coords
  }, [stations, observedRoute, predictions])

  const bounds = useMemo(() => {
    if (plottable.length === 0) {
      return { minLat: 0, maxLat: 1, minLon: 0, maxLon: 1 }
    }
    const lats = plottable.map((item) => item.lat)
    const lons = plottable.map((item) => item.lon)
    let minLat = Math.min(...lats)
    let maxLat = Math.max(...lats)
    let minLon = Math.min(...lons)
    let maxLon = Math.max(...lons)
    const latPad = Math.max((maxLat - minLat) * 0.18, 0.02)
    const lonPad = Math.max((maxLon - minLon) * 0.18, 0.02)
    return {
      minLat: minLat - latPad,
      maxLat: maxLat + latPad,
      minLon: minLon - lonPad,
      maxLon: maxLon + lonPad,
    }
  }, [plottable])

  const stationPoints = useMemo(() => {
    const map = new Map<string, Point>()
    for (const station of stations) {
      if (!hasCoords(station)) continue
      map.set(station.camera_id, project(station.latitude, station.longitude, bounds))
    }
    return map
  }, [stations, bounds])

  const observedLine = useMemo(() => {
    const points: Point[] = []
    for (const stop of observedRoute) {
      const point = stationPoints.get(stop.camera_id)
      if (!point) continue
      const previous = points[points.length - 1]
      if (previous && previous.x === point.x && previous.y === point.y) continue
      points.push(point)
    }
    return points
  }, [observedRoute, stationPoints])

  const predictedSegments = useMemo(() => {
    if (!predictionAvailable || !currentStationId) return []
    const origin = stationPoints.get(currentStationId)
    if (!origin) return []
    return predictions
      .filter((item) => hasCoords(item) && stationPoints.has(item.camera_id))
      .map((item) => ({
        candidate: item,
        from: origin,
        to: stationPoints.get(item.camera_id)!,
        strongest: item.camera_id === predictedCameraId || item.rank === 1,
      }))
  }, [predictionAvailable, currentStationId, predictions, predictedCameraId, stationPoints])

  const hullPoints = useMemo(() => {
    if (!homeRange?.available) return []
    return homeRange.polygon
      .filter((point) => hasCoords(point))
      .map((point) => project(point.latitude, point.longitude, bounds))
  }, [homeRange, bounds])

  const maxOccupancy = Math.max(1, ...stations.map((station) => occupancyValue(station, occupancyMode)))
  const missingCoords = stations.filter((station) => !hasCoords(station))

  if (stations.length === 0) {
    return <p className="empty">Import at least one camera folder to see the station map.</p>
  }

  if (plottable.length === 0) {
    return (
      <p className="empty">
        No registered station coordinates. Add latitude and longitude on Camera Trap Import to plot stations.
      </p>
    )
  }

  return (
    <div className="pench-map">
      <svg
        className="pench-map-canvas"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Pench station map"
      >
        <defs>
          <radialGradient id="penchGlow" cx="50%" cy="42%" r="70%">
            <stop offset="0%" stopColor="#1d3328" />
            <stop offset="55%" stopColor="#121c17" />
            <stop offset="100%" stopColor="#0b100e" />
          </radialGradient>
          <pattern id="penchHatch" width="28" height="28" patternUnits="userSpaceOnUse">
            <path d="M0 28 L28 0" stroke="rgba(63,143,109,0.08)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width={WIDTH} height={HEIGHT} fill="url(#penchGlow)" />
        <rect width={WIDTH} height={HEIGHT} fill="url(#penchHatch)" />
        <text x="24" y="30" className="map-kicker">Pench station map</text>
        <text x="24" y="50" className="map-sub">Registered camera coordinates only</text>

        {showOccupancy &&
          stations.map((station) => {
            const point = stationPoints.get(station.camera_id)
            if (!point) return null
            const value = occupancyValue(station, occupancyMode)
            if (value <= 0) return null
            const radius = 16 + (value / maxOccupancy) * 28
            const level = occupancyLevel(station, occupancyMode)
            return (
              <circle
                key={`occ-${station.camera_id}`}
                cx={point.x}
                cy={point.y}
                r={radius}
                className={`occupancy-disk ${level}`}
              />
            )
          })}

        {showHomeRange && hullPoints.length >= 3 && (
          <polygon
            points={linePoints(hullPoints)}
            className="home-range"
          />
        )}
        {showHomeRange && hullPoints.length >= 3 && (
          <text
            x={hullPoints.reduce((sum, point) => sum + point.x, 0) / hullPoints.length}
            y={hullPoints.reduce((sum, point) => sum + point.y, 0) / hullPoints.length - 18}
            className="map-label range"
          >
            Estimated home range
          </text>
        )}

        {observedLine.length >= 2 && (
          <polyline points={linePoints(observedLine)} className="route observed" />
        )}

        {predictedSegments.map((segment) => (
          <line
            key={`pred-${segment.candidate.camera_id}`}
            x1={segment.from.x}
            y1={segment.from.y}
            x2={segment.to.x}
            y2={segment.to.y}
            className={`route predicted ${segment.strongest ? 'strong' : 'weak'}`}
          />
        ))}

        {stations.map((station) => {
          const point = stationPoints.get(station.camera_id)
          if (!point) return null
          const isCurrent = station.camera_id === currentStationId
          const isPredicted = station.camera_id === predictedCameraId && predictionAvailable
          const isCandidate = predictions.some((item) => item.camera_id === station.camera_id)
          const selected = station.camera_id === selectedStationId
          return (
            <g
              key={station.camera_id}
              className="station-hit"
              onClick={() => onSelectStation(station.camera_id)}
            >
              {isCurrent && <circle cx={point.x} cy={point.y} r="18" className="station-ring current" />}
              {isPredicted && <circle cx={point.x} cy={point.y} r="16" className="station-ring predicted" />}
              <circle
                cx={point.x}
                cy={point.y}
                r={isCurrent || isPredicted ? 8 : 6}
                className={`station-dot ${zoneClass(station.zone_type)} ${selected ? 'selected' : ''} ${isCandidate && !isPredicted ? 'candidate' : ''}`}
              />
              <text x={point.x} y={point.y - 22} className="map-label station">
                {station.camera_id}
              </text>
              {isCurrent && (
                <text x={point.x} y={point.y + 28} className="map-label current">
                  Current station
                </text>
              )}
              {isPredicted && (
                <text x={point.x} y={point.y + (isCurrent ? 44 : 28)} className="map-label predicted">
                  Predicted next station
                </text>
              )}
            </g>
          )
        })}
      </svg>
      {missingCoords.length > 0 && (
        <p className="muted" style={{ margin: '10px 0 0' }}>
          Not plotted (no coordinates): {missingCoords.map((item) => item.camera_id).join(', ')}
        </p>
      )}
    </div>
  )
}

export { occupancyLevel, occupancyValue }

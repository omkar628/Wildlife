import { useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import type {
  ActivityArea,
  HomeRange,
  MovementEdge,
  OccupancyMode,
  OccupancyStation,
  ObservedStop,
  RankedCandidate,
  TigerMapEntity,
  WildlifeClassNode,
} from '../api'
import { EmptyPanel, ErrorPanel, LoadingPanel } from '../ui/Status'
import {
  PENCH_BOUNDS,
  PENCH_CENTER,
  PENCH_MAP_LABEL,
  PENCH_MAX_ZOOM,
  PENCH_MIN_ZOOM,
  PENCH_ZOOM,
} from '../ui/penchMap'

const CLASS_COLORS: Record<string, string> = {
  tiger: '#d4893a',
  prey: '#7eb36a',
  rival: '#c47a9a',
  human: '#6aa8c4',
}
const TIGER_PALETTE = ['#d4893a', '#e0a14a', '#c46a2a', '#f0c27a', '#b86b3a', '#df8a4a', '#a85c28', '#f2b45a']
const PREDICTED_COLOR = '#8fd0ff'

export function tigerColor(tigerId: string): string {
  let hash = 0
  for (let i = 0; i < tigerId.length; i += 1) {
    hash = (hash * 31 + tigerId.charCodeAt(i)) >>> 0
  }
  return TIGER_PALETTE[hash % TIGER_PALETTE.length]
}
const OSM_TILES = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

type LocatedStation = OccupancyStation & { latitude: number; longitude: number }

function hasCoords(item: {
  latitude?: number | null
  longitude?: number | null
}): item is { latitude: number; longitude: number } {
  return item.latitude != null && item.longitude != null && Number.isFinite(item.latitude) && Number.isFinite(item.longitude)
}

function isLocatedStation(station: OccupancyStation): station is LocatedStation {
  return hasCoords(station)
}

function occupancyValue(station: OccupancyStation, mode: OccupancyMode): number {
  if (mode === 'all_species') return station.all_species_detections
  if (mode === 'selected_tiger') return station.selected_tiger_captures
  if (mode === 'prey') return station.prey_detections ?? 0
  if (mode === 'rival') return station.rival_detections ?? 0
  if (mode === 'human') return station.human_detections ?? 0
  return station.tiger_captures
}

function occupancyLevel(station: OccupancyStation, mode: OccupancyMode): string {
  if (mode === 'all_species') return station.occupancy_level_all_species
  if (mode === 'selected_tiger') return station.occupancy_level_selected_tiger
  if (mode === 'prey') return station.occupancy_level_prey ?? 'none'
  if (mode === 'rival') return station.occupancy_level_rival ?? 'none'
  if (mode === 'human') return station.occupancy_level_human ?? 'none'
  return station.occupancy_level_tiger
}

function occupancyColor(level: string): string {
  if (level === 'high') return 'rgba(212, 137, 58, 0.38)'
  if (level === 'medium') return 'rgba(224, 161, 74, 0.28)'
  if (level === 'low') return 'rgba(63, 143, 109, 0.24)'
  return 'rgba(63, 143, 109, 0.08)'
}

function edgeKey(edge: MovementEdge, index: number): string {
  return `${edge.animal_class ?? 'unknown'}-${edge.source}-${edge.target}-${edge.tiger_id ?? ''}-${index}`
}

function resetLeafletContainer(el: HTMLElement) {
  const marked = el as HTMLElement & { _leaflet_id?: number }
  if (marked._leaflet_id) {
    delete marked._leaflet_id
  }
  el.replaceChildren()
}

type StationMapProps = {
  stations: OccupancyStation[]
  observedRoute: ObservedStop[]
  movementEdges: MovementEdge[]
  currentStationId: string | null
  predictions: RankedCandidate[]
  predictedCameraId: string | null
  predictionAvailable: boolean
  homeRange: HomeRange | null
  occupancyMode: OccupancyMode
  showOccupancy: boolean
  showHomeRange: boolean
  showObserved: boolean
  showPredicted: boolean
  selectedStationId: string | null
  selectedEdgeKey: string | null
  onSelectStation: (cameraId: string) => void
  onSelectEdge: (edge: MovementEdge, key: string) => void
  compact?: boolean
  tigerMarkers?: TigerMapEntity[]
  wildlifeNodes?: WildlifeClassNode[]
  activityArea?: ActivityArea | null
  showLastSeen?: boolean
  lastSeenCameraId?: string | null
  showTigerMarkers?: boolean
  classFilters?: { tiger: boolean; prey: boolean; rival: boolean; human: boolean }
  focusPoint?: { latitude: number; longitude: number } | null
  onSelectTiger?: (tigerId: string) => void
}

export default function StationMap({
  stations,
  movementEdges,
  currentStationId,
  predictions,
  predictedCameraId,
  predictionAvailable,
  homeRange,
  occupancyMode,
  showOccupancy,
  showHomeRange,
  showObserved,
  showPredicted,
  selectedStationId,
  selectedEdgeKey,
  onSelectStation,
  onSelectEdge,
  compact = false,
  tigerMarkers = [],
  wildlifeNodes = [],
  activityArea = null,
  showLastSeen = true,
  lastSeenCameraId = null,
  showTigerMarkers = true,
  classFilters = { tiger: true, prey: true, rival: true, human: true },
  focusPoint = null,
  onSelectTiger,
}: StationMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<L.Map | null>(null)
  const overlaysRef = useRef<L.LayerGroup | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [mapError, setMapError] = useState<string | null>(null)

  const plottable = useMemo(() => stations.filter(isLocatedStation), [stations])
  const stationLookup = useMemo(() => {
    const map = new Map<string, OccupancyStation>()
    for (const station of stations) map.set(station.camera_id, station)
    return map
  }, [stations])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    let cancelled = false
    let map: L.Map | null = null
    let resize: ResizeObserver | null = null

    const frame = window.requestAnimationFrame(() => {
      if (cancelled || !containerRef.current) return
      try {
        resetLeafletContainer(el)
        map = L.map(el, {
          zoomControl: true,
          attributionControl: true,
          scrollWheelZoom: true,
          center: PENCH_CENTER,
          zoom: PENCH_ZOOM,
          minZoom: PENCH_MIN_ZOOM,
          maxZoom: PENCH_MAX_ZOOM,
          maxBounds: L.latLngBounds(PENCH_BOUNDS[0], PENCH_BOUNDS[1]),
          maxBoundsViscosity: 1,
          worldCopyJump: false,
        })
        map.setView(PENCH_CENTER, PENCH_ZOOM)
        L.tileLayer(OSM_TILES, {
          attribution: OSM_ATTRIBUTION,
          maxZoom: PENCH_MAX_ZOOM,
          detectRetina: true,
        }).addTo(map)
        const overlays = L.layerGroup().addTo(map)
        mapRef.current = map
        overlaysRef.current = overlays
        setMapReady(true)
        setMapError(null)
        map.invalidateSize()
        resize = new ResizeObserver(() => {
          map?.invalidateSize()
        })
        resize.observe(el)
      } catch (err) {
        setMapError(err instanceof Error ? err.message : String(err))
        setMapReady(false)
      }
    })

    return () => {
      cancelled = true
      window.cancelAnimationFrame(frame)
      resize?.disconnect()
      if (map) {
        map.remove()
      }
      mapRef.current = null
      overlaysRef.current = null
      setMapReady(false)
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    const overlays = overlaysRef.current
    if (!map || !overlays || !mapReady) return
    overlays.clearLayers()
    if (plottable.length === 0) {
      window.setTimeout(() => map.invalidateSize(), 50)
      return
    }

    const maxValue = Math.max(1, ...stations.map((station) => occupancyValue(station, occupancyMode)))
    const activityCameras = activityArea?.cameras ?? []
    const maxActivity = Math.max(1, ...activityCameras.map((item) => item.observation_count))

    if (showOccupancy && activityCameras.length > 0) {
      for (const item of activityCameras) {
        if (item.latitude == null || item.longitude == null || item.observation_count <= 0) continue
        L.circle([item.latitude, item.longitude], {
          radius: 200 + (item.observation_count / maxActivity) * 520,
          color: 'transparent',
          fillColor: occupancyColor(item.intensity >= 0.67 ? 'high' : item.intensity >= 0.34 ? 'medium' : 'low'),
          fillOpacity: 0.82,
          interactive: false,
        }).addTo(overlays)
      }
    } else if (showOccupancy) {
      for (const station of plottable) {
        const value = occupancyValue(station, occupancyMode)
        if (value <= 0) continue
        L.circle([station.latitude, station.longitude], {
          radius: 180 + (value / maxValue) * 420,
          color: 'transparent',
          fillColor: occupancyColor(occupancyLevel(station, occupancyMode)),
          fillOpacity: 0.85,
          interactive: false,
        }).addTo(overlays)
      }
    }

    if (showHomeRange && homeRange?.available && homeRange.polygon.length >= 3) {
      const ring = homeRange.polygon
        .filter(hasCoords)
        .map((point) => [point.latitude, point.longitude] as L.LatLngTuple)
      if (ring.length >= 3) {
        L.polygon(ring, {
          color: 'rgba(224, 161, 74, 0.8)',
          weight: 1.5,
          dashArray: '6 5',
          fillColor: 'rgba(224, 161, 74, 0.12)',
          interactive: false,
        }).addTo(overlays)
      }
    }

    if (showObserved) {
      movementEdges.forEach((edge, index) => {
        const animal = (edge.animal_class || 'tiger') as keyof typeof classFilters
        if (animal in classFilters && !classFilters[animal]) return
        const source = stationLookup.get(edge.source)
        const target = stationLookup.get(edge.target)
        if (!source || !target || !hasCoords(source) || !hasCoords(target)) return
        const key = edgeKey(edge, index)
        const line = L.polyline(
          [
            [source.latitude, source.longitude],
            [target.latitude, target.longitude],
          ],
          {
            color: CLASS_COLORS[edge.animal_class || 'tiger'] || CLASS_COLORS.tiger,
            weight: selectedEdgeKey === key ? 6 : Math.min(5, 2 + edge.weight),
            opacity: selectedEdgeKey === key ? 1 : 0.9,
          },
        )
        line.on('click', (event) => {
          L.DomEvent.stopPropagation(event)
          onSelectEdge(edge, key)
        })
        const identity = edge.tiger_id ? ` ${edge.tiger_id}` : ''
        line.bindTooltip(
          `Observed${identity}: ${edge.source} → ${edge.target} · ${edge.weight} transition${edge.weight === 1 ? '' : 's'}`,
          { sticky: true },
        )
        line.addTo(overlays)
      })
    }

    if (showPredicted && predictionAvailable && currentStationId && predictedCameraId) {
      const origin = stationLookup.get(currentStationId)
      const dest = stationLookup.get(predictedCameraId)
      if (origin && dest && hasCoords(origin) && hasCoords(dest)) {
        L.polyline(
          [
            [origin.latitude, origin.longitude],
            [dest.latitude, dest.longitude],
          ],
          {
            color: PREDICTED_COLOR,
            weight: 4,
            dashArray: '10 8',
            opacity: 0.95,
          },
        )
          .bindTooltip('GNN PREDICTION — not confirmed movement', { sticky: true })
          .addTo(overlays)
        const mid: L.LatLngTuple = [
          (origin.latitude + dest.latitude) / 2,
          (origin.longitude + dest.longitude) / 2,
        ]
        L.marker(mid, {
          icon: L.divIcon({
            className: 'osm-predict-label',
            html: '<span>GNN PREDICTION</span>',
            iconSize: [0, 0],
            iconAnchor: [0, 0],
          }),
          interactive: false,
          keyboard: false,
        }).addTo(overlays)
      }
      for (const candidate of predictions) {
        if (candidate.camera_id === predictedCameraId) continue
        const destCandidate = stationLookup.get(candidate.camera_id)
        if (!origin || !destCandidate || !hasCoords(origin) || !hasCoords(destCandidate)) continue
        L.polyline(
          [
            [origin.latitude, origin.longitude],
            [destCandidate.latitude, destCandidate.longitude],
          ],
          {
            color: PREDICTED_COLOR,
            weight: 2,
            dashArray: '4 8',
            opacity: 0.3,
          },
        ).addTo(overlays)
      }
    }

    for (const station of plottable) {
      const isLastSeen = showLastSeen && station.camera_id === (lastSeenCameraId || currentStationId)
      const isPredicted = station.camera_id === predictedCameraId && predictionAvailable
      const selected = station.camera_id === selectedStationId
      const marker = L.circleMarker([station.latitude, station.longitude], {
        radius: isLastSeen || isPredicted ? 10 : 7,
        color: isPredicted ? PREDICTED_COLOR : isLastSeen ? '#e0a14a' : '#e7eee9',
        weight: selected || isLastSeen ? 3 : 2,
        fillColor: station.enabled === false ? '#627068' : '#151d19',
        fillOpacity: 1,
      })
      marker.on('click', () => onSelectStation(station.camera_id))
      const status = station.enabled === false ? 'disabled' : 'enabled'
      marker.bindTooltip(
        `${station.camera_id} · ${status} · ${station.all_species_detections} observations`,
        { direction: 'top', offset: [0, -8] },
      )
      marker.addTo(overlays)
      const labels = [station.camera_id]
      if (isLastSeen) labels.push('LAST SEEN')
      if (isPredicted) labels.push('Predicted next')
      L.marker([station.latitude, station.longitude], {
        icon: L.divIcon({
          className: `osm-camera-label${isLastSeen ? ' last-seen' : ''}`,
          html: `<span>${labels.join(' · ')}</span>`,
          iconSize: [0, 0],
          iconAnchor: [0, 18],
        }),
        interactive: false,
        keyboard: false,
      }).addTo(overlays)
    }

    if (classFilters.prey || classFilters.rival || classFilters.human) {
      const classOffset: Record<string, [number, number]> = {
        prey: [-0.004, -0.004],
        rival: [0.004, -0.004],
        human: [0.0, 0.005],
      }
      for (const node of wildlifeNodes) {
        const kind = node.animal_class
        if ((kind === 'prey' && !classFilters.prey)
          || (kind === 'rival' && !classFilters.rival)
          || (kind === 'human' && !classFilters.human)) continue
        if (node.latitude == null || node.longitude == null) continue
        const offset = classOffset[kind] ?? [0, 0]
        const lat = node.latitude + offset[0]
        const lon = node.longitude + offset[1]
        const marker = L.circleMarker([lat, lon], {
          radius: 7,
          color: CLASS_COLORS[kind] || '#e7eee9',
          weight: 2,
          fillColor: CLASS_COLORS[kind] || '#e7eee9',
          fillOpacity: 0.85,
        })
        marker.bindPopup(
          `<strong>${kind.toUpperCase()}</strong><br/>${node.camera_id}<br/>Detections: ${node.detection_count}<br/>Last seen: ${node.last_seen ?? '—'}`,
        )
        marker.addTo(overlays)
      }
    }

    if (showTigerMarkers && classFilters.tiger) {
      const grouped = new Map<string, TigerMapEntity[]>()
      for (const tiger of tigerMarkers) {
        if (tiger.latitude == null || tiger.longitude == null) continue
        const key = `${tiger.latitude.toFixed(5)},${tiger.longitude.toFixed(5)}`
        const bucket = grouped.get(key) ?? []
        bucket.push(tiger)
        grouped.set(key, bucket)
      }
      for (const group of grouped.values()) {
        group.forEach((tiger, index) => {
          if (tiger.latitude == null || tiger.longitude == null) return
          const angle = (index / Math.max(group.length, 1)) * Math.PI * 2
          const radius = group.length > 1 ? 0.004 : 0.0022
          const lat = tiger.latitude + Math.sin(angle) * radius
          const lon = tiger.longitude + Math.cos(angle) * radius
          const color = tigerColor(tiger.tiger_id)
          const marker = L.circleMarker([lat, lon], {
            radius: 8,
            color,
            weight: 2,
            fillColor: color,
            fillOpacity: 0.95,
          })
          marker.bindPopup(
            `<strong>${tiger.tiger_id}</strong><br/>Last seen: ${tiger.last_camera ?? '—'}<br/>Last seen: ${tiger.last_seen ?? '—'}<br/>Observations: ${tiger.observation_count}`,
          )
          marker.on('click', () => onSelectTiger?.(tiger.tiger_id))
          marker.addTo(overlays)
          L.marker([lat, lon], {
            icon: L.divIcon({
              className: 'osm-tiger-label',
              html: `<span style="--tiger:${color}">${tiger.tiger_id} ●</span>`,
              iconSize: [0, 0],
              iconAnchor: [0, -10],
            }),
            interactive: false,
            keyboard: false,
          }).addTo(overlays)
        })
      }
    }

    window.setTimeout(() => map.invalidateSize(), 50)
  }, [
    mapReady,
    plottable,
    stations,
    movementEdges,
    currentStationId,
    predictions,
    predictedCameraId,
    predictionAvailable,
    homeRange,
    occupancyMode,
    showOccupancy,
    showHomeRange,
    showObserved,
    showPredicted,
    selectedStationId,
    selectedEdgeKey,
    stationLookup,
    onSelectStation,
    onSelectEdge,
    tigerMarkers,
    wildlifeNodes,
    activityArea,
    showLastSeen,
    lastSeenCameraId,
    showTigerMarkers,
    classFilters,
    onSelectTiger,
  ])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReady || !focusPoint) return
    map.flyTo([focusPoint.latitude, focusPoint.longitude], Math.max(map.getZoom(), 13), {
      duration: 0.6,
    })
  }, [focusPoint, mapReady])

  const missingCoords = stations.filter((station) => !hasCoords(station))

  return (
    <div className={`osm-map ${compact ? 'compact' : ''}`}>
      <div
        ref={containerRef}
        className="osm-map-canvas"
        role="application"
        aria-label="OpenStreetMap of Pench Tiger Reserve"
      />
      <div className="osm-map-label">{PENCH_MAP_LABEL}</div>
      {!mapReady && !mapError ? (
        <div className="osm-map-overlay">
          <LoadingPanel title="Building movement graph…" detail="Loading OpenStreetMap tiles and registered cameras." />
        </div>
      ) : null}
      {mapError ? (
        <div className="osm-map-overlay">
          <ErrorPanel title="Map failed to start" detail={mapError} />
        </div>
      ) : null}
      {mapReady && stations.length === 0 ? (
        <div className="osm-map-overlay">
          <EmptyPanel
            title="No cameras on the map yet"
            detail="Register cameras with latitude and longitude, then import detections."
          />
        </div>
      ) : null}
      {mapReady && stations.length > 0 && plottable.length === 0 ? (
        <div className="osm-map-overlay">
          <EmptyPanel
            title="No plottable coordinates"
            detail="Add valid latitude and longitude on the Cameras page. Coordinates are never invented from images."
          />
        </div>
      ) : null}
      {missingCoords.length > 0 && plottable.length > 0 ? (
        <p className="muted osm-map-note">
          Not plotted (invalid or missing coordinates): {missingCoords.map((item) => item.camera_id).join(', ')}
        </p>
      ) : null}
    </div>
  )
}

export { occupancyLevel, occupancyValue, edgeKey }

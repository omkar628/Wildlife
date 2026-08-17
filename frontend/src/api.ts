const API = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail ?? JSON.stringify(body)
    } catch {
      detail = await response.text()
    }
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/health'),
  dashboard: () => request<Dashboard>('/dashboard'),
  settings: () => request<AppSettings>('/settings'),
  updateSettings: (payload: Partial<AppSettings>) =>
    request<AppSettings>('/settings', { method: 'PUT', body: JSON.stringify(payload) }),
  importFolder: (payload: ImportPayload) =>
    request<Job>('/import', { method: 'POST', body: JSON.stringify(payload) }),
  previewImport: (folder_path: string) =>
    request<ImportPreview>('/import/preview', { method: 'POST', body: JSON.stringify({ folder_path }) }),
  importBatch: (cameras: ImportPayload[]) =>
    request<{ jobs: Job[] }>('/import/batch', { method: 'POST', body: JSON.stringify({ cameras }) }),
  jobs: () => request<{ jobs: Job[] }>('/jobs'),
  job: (id: number) => request<{ job: Job; errors: JobError[] }>(`/jobs/${id}`),
  detections: (className?: string) =>
    request<{ detections: Detection[] }>(`/detections${className ? `?class_name=${className}` : ''}`),
  reviews: () => request<{ reviews: ReviewItem[]; pending: number }>('/reviews'),
  decide: (id: number, human_class: string) =>
    request(`/reviews/${id}/decide`, { method: 'POST', body: JSON.stringify({ human_class }) }),
  unidentifiedObservations: () =>
    request<UnidentifiedPayload>('/observations/unidentified'),
  assignIdentity: (observationId: number, payload: { action: 'assign' | 'create' | 'keep'; tiger_id?: string }) =>
    request<{ tiger_id: string | null; created: boolean; kept_unidentified?: boolean }>(
      `/observations/${observationId}/identity`,
      { method: 'POST', body: JSON.stringify(payload) },
    ),
  images: () => request<{ images: ImageRow[] }>('/images'),
  tigers: () => request<{ tigers: TigerRow[]; catalog: TigerCatalogItem[]; next_tiger_id: string }>('/tigers'),
  tiger: (tigerId: string) =>
    request<TigerDetail>(`/tigers/${encodeURIComponent(tigerId)}`),
  cameras: () => request<{ cameras: CameraRow[] }>('/cameras'),
  camera: (cameraId: string) =>
    request<CameraRow>(`/cameras/${encodeURIComponent(cameraId)}`),
  createCamera: (payload: CameraPayload) =>
    request<CameraRow>('/cameras', { method: 'POST', body: JSON.stringify(payload) }),
  updateCamera: (cameraId: string, payload: CameraUpdatePayload) =>
    request<CameraRow>(`/cameras/${encodeURIComponent(cameraId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  setCameraEnabled: (cameraId: string, enabled: boolean) =>
    request<CameraRow>(`/cameras/${encodeURIComponent(cameraId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
    }),
  deleteCamera: (cameraId: string) =>
    request<{ ok: boolean; camera_id: string }>(`/cameras/${encodeURIComponent(cameraId)}`, {
      method: 'DELETE',
    }),
  graph: (params?: GraphFilters) => {
    const query = new URLSearchParams()
    if (params?.tiger_id) query.set('tiger_id', params.tiger_id)
    if (params?.animal_class) query.set('animal_class', params.animal_class)
    if (params?.camera_id) query.set('camera_id', params.camera_id)
    if (params?.time_from) query.set('time_from', params.time_from)
    if (params?.time_to) query.set('time_to', params.time_to)
    const suffix = query.toString() ? `?${query.toString()}` : ''
    return request<GraphPayload>(`/graph${suffix}`)
  },
  gnnPrediction: (tigerId: string) =>
    request<GnnPrediction>(`/graph/predictions?tiger_id=${encodeURIComponent(tigerId)}`),
  tigerRoute: (tigerId: string) =>
    request<TigerRoute>(`/graph/tigers/${encodeURIComponent(tigerId)}/route`),
}

export function imageUrl(imageId: number): string {
  return `${API}/media/images/${imageId}`
}

export function cropUrl(observationId: number): string {
  return `${API}/media/crops/${observationId}`
}

export type Health = {
  status: string
  offline: boolean
  detector: { available: boolean; path: string; device: string }
  reid: {
    implemented?: boolean
    loaded?: boolean
    device?: string
    backend?: string
    reason?: string | null
    uses_atrw_gallery?: boolean
    match_threshold?: number
    review_threshold?: number
    local_identity?: {
      encoder_enabled?: boolean
      assigns_atrw_ids?: boolean
      id_namespace?: string
    }
  }
  gnn: GnnStatus
  confidence: { auto_accept: number; detect_min: number }
}

export type Dashboard = {
  images: { total: number; by_status: Record<string, number> }
  detections: {
    total: number
    by_class: Record<string, number>
    tiger: number
    prey: number
    rival: number
    human: number
  }
  review: { pending: number; unidentified_tigers?: number }
  tigers: { known: number }
  recent_jobs: Job[]
  confidence_auto_accept: number
}

export type AppSettings = {
  confidence_auto_accept: number
  confidence_detect_min: number
  detector_batch_size: number
  detector_device: string
  class_map: Record<string, string>
}

export type ImportPayload = {
  folder_path: string
  camera_id: string
  latitude?: number | null
  longitude?: number | null
  elevation?: number | null
  habitat?: string | null
  create_if_missing?: boolean
  name?: string | null
}

export type DiscoveredCameraFolder = {
  folder_path: string
  folder_name: string
  suggested_camera_id: string
  image_count: number
  sample_names: string[]
  camera_exists: boolean
  match_status?: 'matched' | 'unknown'
  unknown_camera_folder?: boolean
}

export type ImportPreview = {
  folder_path: string
  total_images: number
  camera_count: number
  camera_folders: DiscoveredCameraFolder[]
  known_cameras: string[]
}

export type Job = {
  job_id: number
  folder_path: string
  camera_id: string
  status: string
  total_images: number
  processed: number
  skipped: number
  duplicates: number
  failed: number
  tiger_count: number
  prey_count: number
  rival_count: number
  human_count: number
  low_confidence_count: number
  review_count: number
  started_at: string | null
  finished_at: string | null
  error_message: string | null
  confidence_threshold: number | null
}

export type JobError = {
  error_id: number
  original_path: string
  error_message: string
}

export type Detection = {
  detection_id: number
  image_id: number
  class_name: string
  final_class_name: string | null
  confidence: number
  bbox_x: number
  bbox_y: number
  bbox_width: number
  bbox_height: number
  camera_id: string | null
  filename: string
  timestamp: string | null
  tiger_id?: string | null
  review_status: string
  accepted: number
}

export type ReviewItem = {
  review_id: number
  detection_id: number
  image_id: number
  predicted_class: string
  predicted_confidence: number
  filename: string
  camera_id: string | null
  timestamp: string | null
  bbox_x: number
  bbox_y: number
  bbox_width: number
  bbox_height: number
  image_width: number | null
  image_height: number | null
}

export type ImageRow = {
  image_id: number
  filename: string
  camera_id: string | null
  timestamp: string | null
  timestamp_source: string | null
  processing_status: string
  detection_count: number
  original_path: string
}

export type TigerRow = {
  tiger_id: string
  first_seen: string | null
  last_seen: string | null
  observation_count: number
  last_camera?: string | null
}

export type IdentityReference = {
  observation_id: number
  camera_id: string | null
  timestamp: string | null
  crop_path: string | null
  tiger_id?: string | null
}

export type TigerCatalogItem = TigerRow & {
  references: IdentityReference[]
}

export type ReidCandidate = {
  tiger_id: string
  similarity: number
  support?: number
  mean_similarity?: number
}

export type ReidSuggestion = {
  matched: boolean
  tiger_id: string | null
  suggested_tiger_id: string | null
  similarity: number | null
  needs_review: boolean
  decision: string | null
  candidates: ReidCandidate[]
  reason: string | null
  deferred?: boolean
}

export type UnidentifiedObservation = {
  observation_id: number
  detection_id: number
  image_id: number
  camera_id: string | null
  filename: string
  timestamp: string | null
  image_timestamp: string | null
  crop_path: string | null
  reid?: ReidSuggestion | null
}

export type UnidentifiedPayload = {
  observations: UnidentifiedObservation[]
  pending: number
  tigers: TigerCatalogItem[]
  next_tiger_id: string
}

export type ActivityCamera = {
  camera_id: string
  observation_count: number
  intensity: number
  last_seen?: string | null
  latitude?: number | null
  longitude?: number | null
  missing_coordinates?: boolean
}

export type ActivityArea = {
  label: string
  tiger_id?: string
  cameras: ActivityCamera[]
  strongest_camera: string | null
  strongest_count: number
  region?: HomeRange & { label?: string }
}

export type TigerDetail = {
  tiger: TigerRow
  history: Array<{
    tiger_id: string | null
    camera_id: string | null
    timestamp: string | null
    observation_id: number
    detection_id: number
  }>
  references: IdentityReference[]
  last_camera?: string | null
  last_seen?: string | null
  cameras_visited?: string[]
  observation_count?: number
  most_frequent_camera?: string | null
  most_frequent_count?: number
  activity_area?: ActivityArea
  current_station?: ObservedStop | null
}

export type CameraRow = {
  camera_id: string
  name?: string | null
  latitude: number | null
  longitude: number | null
  elevation?: number | null
  habitat: string | null
  metadata?: string | null
  enabled?: boolean | number
  status?: string
  image_count?: number
  observation_count?: number
  tiger_count?: number
  prey_count?: number
  rival_count?: number
  human_count?: number
  missing_coordinates?: boolean
}

export type CameraPayload = {
  camera_id: string
  name?: string | null
  latitude?: number | null
  longitude?: number | null
  elevation?: number | null
  habitat?: string | null
  enabled?: boolean
}

export type CameraUpdatePayload = Partial<CameraPayload>

export type GraphFilters = {
  tiger_id?: string
  animal_class?: string
  camera_id?: string
  time_from?: string
  time_to?: string
}

export type OccupancyMode = 'all_species' | 'tiger' | 'prey' | 'rival' | 'human' | 'selected_tiger'

export type OccupancyStation = {
  camera_id: string
  name?: string | null
  enabled?: boolean
  status?: string
  latitude: number | null
  longitude: number | null
  registered: boolean
  missing_coordinates: boolean
  zone_type: string | null
  habitat: string | null
  all_species_detections: number
  tiger_captures: number
  tiger_detections?: number
  prey_detections?: number
  rival_detections?: number
  human_detections?: number
  unique_tigers: number
  selected_tiger_captures: number
  latest_tiger_id: string | null
  latest_tiger_timestamp: string | null
  occupancy_level_all_species: string
  occupancy_level_tiger: string
  occupancy_level_prey?: string
  occupancy_level_rival?: string
  occupancy_level_human?: string
  occupancy_level_selected_tiger: string
  capture_frequency_per_day: number | null
  capture_span_days: number | null
}

export type OccupancyPayload = {
  label?: string
  stations: OccupancyStation[]
  supported_modes: OccupancyMode[]
  selected_tiger_id?: string | null
}

export type ObservedStop = {
  camera_id: string
  timestamp: string | null
  latitude: number | null
  longitude: number | null
  confidence: number | null
  reid_confidence?: number | null
  observation_id: number
  detection_id?: number
  image_id?: number | null
  tiger_id: string
  class_name?: string | null
  crop_path?: string | null
  filename?: string | null
  embedding_available?: boolean
  bbox_x?: number | null
  bbox_y?: number | null
  bbox_width?: number | null
  bbox_height?: number | null
  missing_coordinates: boolean
  registered: boolean
  zone_type: string | null
  habitat: string | null
}

export type MovementEdge = {
  source: string
  target: string
  tiger_id: string | null
  weight: number
  first_timestamp?: string | null
  last_timestamp?: string | null
  animal_class?: string | null
  identity?: string | null
  distance_km?: number | null
  confidence?: number | null
  observation_ids?: number[]
  detection_ids?: number[]
  source_observation_id?: number | null
  destination_observation_id?: number | null
  kind?: string
}

export type RankedCandidate = {
  rank: number
  camera_id: string
  score: number | null
  confidence: number | null
  latitude: number | null
  longitude: number | null
  registered: boolean
  missing_coordinates: boolean
  zone_type: string | null
}

export type RoutePredictions = {
  available: boolean
  reason?: string
  detail?: string
  tiger_id?: string
  predicted_camera_id?: string
  confidence?: number
  summary?: string
  latitude?: number | null
  longitude?: number | null
  missing_coordinates?: boolean
  ranked_candidates?: RankedCandidate[]
  feature_degraded?: boolean
  history?: Array<{ camera_id: string; timestamp: string | null; observation_id?: number }>
  prediction_timestamp?: string
}

export type HomeRange = {
  available: boolean
  reason: string | null
  label: string
  polygon: Array<{ latitude: number; longitude: number; camera_id?: string }>
  point_count: number
  unique_stations: number
}

export type TigerRoute = {
  tiger_id: string
  observed_route: ObservedStop[]
  current_station: ObservedStop | null
  predictions: RoutePredictions
  occupancy: OccupancyPayload
  home_range: HomeRange
  activity_area?: ActivityArea
  visited_stations: string[]
  observation_count: number
  last_observed_station: string | null
  last_observed_timestamp: string | null
  most_frequent_camera?: string | null
  most_frequent_count?: number
}

export type TigerMapEntity = {
  tiger_id: string
  last_camera: string | null
  last_seen: string | null
  observation_count: number
  cameras_visited: string[]
  most_frequent_camera: string | null
  most_frequent_count: number
  latitude: number | null
  longitude: number | null
  missing_coordinates: boolean
  registered: boolean
  activity_area?: ActivityArea
}

export type WildlifeClassNode = {
  animal_class: 'prey' | 'rival' | 'human' | string
  camera_id: string
  detection_count: number
  last_seen: string | null
  confidence?: number | null
  detection_id?: number
  image_id?: number | null
  latitude: number | null
  longitude: number | null
  missing_coordinates: boolean
  registered: boolean
}

export type WildlifeEntities = {
  tigers: TigerMapEntity[]
  prey: WildlifeClassNode[]
  rival: WildlifeClassNode[]
  human: WildlifeClassNode[]
}

export type GnnStatus = {
  implemented?: boolean
  loaded: boolean
  device: string
  path?: string
  version?: string | null
  reason?: string | null
  history_len_required?: number
  predictions?: GnnPrediction[]
}

export type GnnCandidate = {
  rank: number
  camera_id: string
  confidence: number
}

export type GnnPrediction = {
  available: boolean
  reason?: string
  detail?: string
  tiger_id?: string
  predicted_camera_id?: string
  confidence?: number
  ranked_candidates?: GnnCandidate[]
  history?: Array<{ camera_id: string; timestamp: string | null; observation_id?: number }>
  prediction_timestamp?: string
  feature_degraded?: boolean
  feature_notes?: string[]
  observation_count?: number
  model?: { version?: string | null; device?: string; path?: string }
}

export type GraphPayload = {
  camera_graph: {
    nodes: Array<{
      camera_id: string
      name?: string | null
      latitude?: number | null
      longitude?: number | null
      habitat?: string | null
      observation_count: number
      image_count: number
      enabled?: boolean
      status?: string
      tiger_count?: number
      prey_count?: number
      rival_count?: number
      human_count?: number
    }>
    edges: Array<{ source: string; target: string; tiger_id: string | null; weight: number }>
  }
  observation_graph: {
    events: Array<{
      tiger_id: string | null
      camera_id: string | null
      timestamp: string | null
      confidence: number | null
    }>
  }
  movement_edges?: MovementEdge[]
  occupancy?: OccupancyPayload
  wildlife_entities?: WildlifeEntities
  gnn: GnnStatus
}

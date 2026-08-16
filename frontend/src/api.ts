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
  assignIdentity: (observationId: number, payload: { action: 'assign' | 'create'; tiger_id?: string }) =>
    request<{ tiger_id: string; created: boolean }>(
      `/observations/${observationId}/identity`,
      { method: 'POST', body: JSON.stringify(payload) },
    ),
  images: () => request<{ images: ImageRow[] }>('/images'),
  tigers: () => request<{ tigers: TigerRow[]; catalog: TigerCatalogItem[]; next_tiger_id: string }>('/tigers'),
  tiger: (tigerId: string) =>
    request<TigerDetail>(`/tigers/${encodeURIComponent(tigerId)}`),
  cameras: () => request<{ cameras: CameraRow[] }>('/cameras'),
  graph: () => request<GraphPayload>('/graph'),
  gnnPrediction: (tigerId: string) =>
    request<GnnPrediction>(`/graph/predictions?tiger_id=${encodeURIComponent(tigerId)}`),
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
  reid: { implemented: boolean; reason: string }
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
}

export type DiscoveredCameraFolder = {
  folder_path: string
  folder_name: string
  suggested_camera_id: string
  image_count: number
  sample_names: string[]
  camera_exists: boolean
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

export type UnidentifiedObservation = {
  observation_id: number
  detection_id: number
  image_id: number
  camera_id: string | null
  filename: string
  timestamp: string | null
  image_timestamp: string | null
  crop_path: string | null
}

export type UnidentifiedPayload = {
  observations: UnidentifiedObservation[]
  pending: number
  tigers: TigerCatalogItem[]
  next_tiger_id: string
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
}

export type CameraRow = {
  camera_id: string
  latitude: number | null
  longitude: number | null
  habitat: string | null
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
    nodes: Array<{ camera_id: string; observation_count: number; image_count: number }>
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
  gnn: GnnStatus
}

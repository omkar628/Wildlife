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
  jobs: () => request<{ jobs: Job[] }>('/jobs'),
  job: (id: number) => request<{ job: Job; errors: JobError[] }>(`/jobs/${id}`),
  detections: (className?: string) =>
    request<{ detections: Detection[] }>(`/detections${className ? `?class_name=${className}` : ''}`),
  reviews: () => request<{ reviews: ReviewItem[]; pending: number }>('/reviews'),
  decide: (id: number, human_class: string) =>
    request(`/reviews/${id}/decide`, { method: 'POST', body: JSON.stringify({ human_class }) }),
  images: () => request<{ images: ImageRow[] }>('/images'),
  tigers: () => request<{ tigers: TigerRow[] }>('/tigers'),
  cameras: () => request<{ cameras: CameraRow[] }>('/cameras'),
  graph: () => request<GraphPayload>('/graph'),
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
  review: { pending: number }
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

export type CameraRow = {
  camera_id: string
  latitude: number | null
  longitude: number | null
  habitat: string | null
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
  gnn: { implemented: boolean }
}

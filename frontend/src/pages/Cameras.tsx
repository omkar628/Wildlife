import { useEffect, useMemo, useState } from 'react'
import { api, type CameraPayload, type CameraRow, type OccupancyStation } from '../api'
import { EmptyPanel, ErrorPanel, LoadingPanel } from '../ui/Status'

const emptyForm: CameraPayload = {
  camera_id: '',
  name: '',
  latitude: null,
  longitude: null,
  elevation: null,
  habitat: '',
  enabled: true,
}

function numberOrNull(value: string): number | null {
  const trimmed = value.trim()
  if (!trimmed) return null
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : null
}

function isEnabled(camera: CameraRow): boolean {
  return camera.enabled !== 0 && camera.enabled !== false
}

function formatCoord(value: number | null | undefined): string {
  return value == null ? '—' : value.toFixed(5)
}

export default function CamerasPage() {
  const [cameras, setCameras] = useState<CameraRow[]>([])
  const [occupancy, setOccupancy] = useState<OccupancyStation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)
  const [form, setForm] = useState<CameraPayload>(emptyForm)
  const [latText, setLatText] = useState('')
  const [lonText, setLonText] = useState('')
  const [elevText, setElevText] = useState('')

  async function load() {
    const [body, graph] = await Promise.all([api.cameras(), api.graph()])
    setCameras(body.cameras)
    setOccupancy(graph.occupancy?.stations ?? [])
  }

  useEffect(() => {
    load()
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  const occupancyById = useMemo(() => {
    const map = new Map<string, OccupancyStation>()
    for (const station of occupancy) map.set(station.camera_id, station)
    return map
  }, [occupancy])

  function startCreate() {
    setEditing(null)
    setForm(emptyForm)
    setLatText('')
    setLonText('')
    setElevText('')
    setError(null)
  }

  function startEdit(camera: CameraRow) {
    setEditing(camera.camera_id)
    setForm({
      camera_id: camera.camera_id,
      name: camera.name || camera.camera_id,
      latitude: camera.latitude,
      longitude: camera.longitude,
      elevation: camera.elevation ?? null,
      habitat: camera.habitat || '',
      enabled: isEnabled(camera),
    })
    setLatText(camera.latitude == null ? '' : String(camera.latitude))
    setLonText(camera.longitude == null ? '' : String(camera.longitude))
    setElevText(camera.elevation == null ? '' : String(camera.elevation))
    setError(null)
  }

  async function save() {
    const cameraId = (form.camera_id || '').trim()
    if (!cameraId) {
      setError('Camera ID is required and must be unique.')
      return
    }
    const payload: CameraPayload = {
      camera_id: cameraId,
      name: (form.name || cameraId).trim(),
      latitude: numberOrNull(latText),
      longitude: numberOrNull(lonText),
      elevation: numberOrNull(elevText),
      habitat: form.habitat || null,
      enabled: form.enabled !== false,
    }
    if ((latText && payload.latitude == null) || (lonText && payload.longitude == null)) {
      setError('Latitude and longitude must be valid numbers.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      if (editing) {
        await api.updateCamera(editing, payload)
      } else {
        await api.createCamera(payload)
      }
      startCreate()
      await load()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function toggle(camera: CameraRow) {
    setError(null)
    try {
      await api.setCameraEnabled(camera.camera_id, !isEnabled(camera))
      await load()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function remove(camera: CameraRow) {
    if (!window.confirm(`Delete camera ${camera.camera_id}? This only works if no images are linked.`)) {
      return
    }
    setError(null)
    try {
      await api.deleteCamera(camera.camera_id)
      if (editing === camera.camera_id) startCreate()
      await load()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Cameras</h2>
          <p>Register field cameras. Import folder names must match these IDs. Coordinates always come from this record.</p>
        </div>
      </div>

      {loading ? <LoadingPanel title="Loading camera network…" /> : null}

      <div className="grid two">
        <article className="card">
          <h3>Registered cameras</h3>
          {!loading && cameras.length === 0 ? (
            <EmptyPanel
              title="No cameras yet"
              detail="Add Camera_01, Camera_02, Camera_03 before importing folders."
            />
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Camera ID</th>
                  <th>Location</th>
                  <th>Habitat</th>
                  <th>Status</th>
                  <th>Obs</th>
                  <th>Last observation</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {cameras.map((camera) => {
                  const station = occupancyById.get(camera.camera_id)
                  return (
                    <tr key={camera.camera_id}>
                      <td>
                        <strong>{camera.camera_id}</strong>
                        <div className="muted" style={{ fontSize: 12 }}>{camera.name || camera.camera_id}</div>
                      </td>
                      <td>
                        {formatCoord(camera.latitude)}, {formatCoord(camera.longitude)}
                        {camera.missing_coordinates ? <div className="error">Invalid or missing</div> : null}
                      </td>
                      <td>{camera.habitat || '—'}</td>
                      <td>
                        <span className={`badge ${isEnabled(camera) ? 'ok' : 'warn'}`}>
                          {isEnabled(camera) ? 'enabled' : 'disabled'}
                        </span>
                      </td>
                      <td>{camera.observation_count ?? 0}</td>
                      <td>{station?.latest_tiger_timestamp ?? '—'}</td>
                      <td>
                        <div className="actions">
                          <button className="btn small ghost" type="button" onClick={() => startEdit(camera)}>Edit</button>
                          <button className="btn small ghost" type="button" onClick={() => toggle(camera)}>
                            {isEnabled(camera) ? 'Disable' : 'Enable'}
                          </button>
                          <button className="btn small ignore" type="button" onClick={() => remove(camera)}>Delete</button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </article>

        <article className="card form">
          <h3>{editing ? `Edit ${editing}` : 'Add camera'}</h3>
          <div className="field">
            <label htmlFor="camera-id">Camera ID</label>
            <input id="camera-id" value={form.camera_id} onChange={(event) => setForm({ ...form, camera_id: event.target.value })} placeholder="Camera_01" />
          </div>
          <div className="field">
            <label htmlFor="camera-name">Name</label>
            <input id="camera-name" value={form.name ?? ''} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="Ridge camera" />
          </div>
          <div className="row-2">
            <div className="field">
              <label htmlFor="camera-lat">Latitude</label>
              <input id="camera-lat" value={latText} onChange={(event) => setLatText(event.target.value)} placeholder="21.65000" />
            </div>
            <div className="field">
              <label htmlFor="camera-lon">Longitude</label>
              <input id="camera-lon" value={lonText} onChange={(event) => setLonText(event.target.value)} placeholder="79.24000" />
            </div>
          </div>
          <div className="field">
            <label htmlFor="camera-elev">Elevation (optional)</label>
            <input id="camera-elev" value={elevText} onChange={(event) => setElevText(event.target.value)} placeholder="420" />
          </div>
          <div className="field">
            <label htmlFor="camera-habitat">Habitat / metadata</label>
            <input id="camera-habitat" value={form.habitat ?? ''} onChange={(event) => setForm({ ...form, habitat: event.target.value })} placeholder="dry deciduous, core zone" />
          </div>
          <label className="layer-toggle" style={{ paddingBottom: 0 }}>
            <input type="checkbox" checked={form.enabled !== false} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} />
            Enabled
          </label>
          <div className="actions">
            <button className="btn" type="button" onClick={save} disabled={busy}>
              {busy ? 'Saving…' : editing ? 'Save camera' : 'Add camera'}
            </button>
            {editing ? <button className="btn ghost" type="button" onClick={startCreate}>Cancel</button> : null}
          </div>
        </article>
      </div>
      {error ? <ErrorPanel detail={error} /> : null}
    </div>
  )
}

import { useEffect, useState, type FormEvent } from 'react'
import { api, type AppSettings } from '../api'

export default function ImportPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [folder, setFolder] = useState('')
  const [cameraId, setCameraId] = useState('C01')
  const [habitat, setHabitat] = useState('')
  const [latitude, setLatitude] = useState('')
  const [longitude, setLongitude] = useState('')
  const [elevation, setElevation] = useState('')
  const [threshold, setThreshold] = useState('0.60')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.settings().then((value) => {
      setSettings(value)
      setThreshold(String(value.confidence_auto_accept))
    })
  }, [])

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      const nextThreshold = Number(threshold)
      if (!Number.isNaN(nextThreshold)) {
        await api.updateSettings({ confidence_auto_accept: nextThreshold })
      }
      const job = await api.importFolder({
        folder_path: folder,
        camera_id: cameraId,
        habitat: habitat || null,
        latitude: latitude === '' ? null : Number(latitude),
        longitude: longitude === '' ? null : Number(longitude),
        elevation: elevation === '' ? null : Number(elevation),
      })
      setMessage(`Job #${job.job_id} started for camera ${job.camera_id}.`)
      window.location.hash = `#/jobs?id=${job.job_id}`
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Import folder</h2>
          <p>Point at an SD card or camera folder. Subfolders are scanned. Originals are never modified.</p>
        </div>
      </div>
      <div className="grid two">
        <form className="card form" onSubmit={onSubmit}>
          <div className="field">
            <label>Folder path</label>
            <input
              value={folder}
              onChange={(e) => setFolder(e.target.value)}
              placeholder="D:\CameraTrap\C01"
              required
            />
          </div>
          <div className="row-2">
            <div className="field">
              <label>Camera ID</label>
              <input value={cameraId} onChange={(e) => setCameraId(e.target.value)} required />
            </div>
            <div className="field">
              <label>Habitat</label>
              <input value={habitat} onChange={(e) => setHabitat(e.target.value)} placeholder="dry deciduous" />
            </div>
          </div>
          <div className="row-2">
            <div className="field">
              <label>Latitude</label>
              <input value={latitude} onChange={(e) => setLatitude(e.target.value)} />
            </div>
            <div className="field">
              <label>Longitude</label>
              <input value={longitude} onChange={(e) => setLongitude(e.target.value)} />
            </div>
          </div>
          <div className="row-2">
            <div className="field">
              <label>Elevation (m)</label>
              <input value={elevation} onChange={(e) => setElevation(e.target.value)} />
            </div>
            <div className="field">
              <label>Auto-accept threshold</label>
              <input value={threshold} onChange={(e) => setThreshold(e.target.value)} />
            </div>
          </div>
          <button className="btn" type="submit" disabled={busy}>
            {busy ? 'Starting…' : 'Start processing'}
          </button>
          {message && <p className="ok">{message}</p>}
          {error && <p className="error">{error}</p>}
        </form>
        <article className="card">
          <h3>How this works</h3>
          <p className="muted">
            The backend walks the folder recursively for `.jpg`, `.jpeg`, and `.png`.
            Each file is hashed so duplicates are skipped and a stopped job can resume.
          </p>
          <p className="muted">
            Current detector min confidence: {settings ? settings.confidence_detect_min : '…'}.
            Predictions below the auto-accept threshold go to Human review.
          </p>
          <p className="muted">
            Re-ID is not run yet. Tiger crops are saved so identity can be attached later.
          </p>
        </article>
      </div>
    </div>
  )
}

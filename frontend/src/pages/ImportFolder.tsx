import { useEffect, useMemo, useState } from 'react'
import { api, type ImportPreview, type Job } from '../api'
import { isDesktopApp, selectCameraFolder } from '../desktop'

type CameraRow = {
  folder_path: string
  folder_name: string
  camera_id: string
  image_count: number
  sample_names: string[]
  included: boolean
  latitude: string
  longitude: string
  elevation: string
  habitat: string
}

function jobDone(job: Job): boolean {
  return ['completed', 'failed', 'cancelled'].includes(job.status)
}

function jobPercent(job: Job): number {
  if (!job.total_images) return jobDone(job) ? 100 : 0
  const done = job.processed + job.skipped + job.duplicates + job.failed
  return Math.min(100, Math.round((done / job.total_images) * 100))
}

export default function ImportPage() {
  const [folder, setFolder] = useState('')
  const [preview, setPreview] = useState<ImportPreview | null>(null)
  const [rows, setRows] = useState<CameraRow[]>([])
  const [known, setKnown] = useState<string[]>([])
  const [picking, setPicking] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [busy, setBusy] = useState(false)
  const [jobs, setJobs] = useState<Job[]>([])
  const [error, setError] = useState<string | null>(null)
  const desktop = isDesktopApp()

  const selectedCount = rows.filter((row) => row.included && row.camera_id.trim()).length
  const selectedImages = rows
    .filter((row) => row.included && row.camera_id.trim())
    .reduce((sum, row) => sum + row.image_count, 0)
  const allDone = jobs.length > 0 && jobs.every(jobDone)

  useEffect(() => {
    if (jobs.length === 0 || allDone) return
    const ids = new Set(jobs.map((job) => job.job_id))
    const tick = () => {
      api.jobs()
        .then((body) => setJobs(body.jobs.filter((job) => ids.has(job.job_id))))
        .catch(() => undefined)
    }
    const id = window.setInterval(tick, 1200)
    return () => window.clearInterval(id)
  }, [jobs, allDone])

  useEffect(() => {
    if (allDone) {
      window.location.hash = '#/review'
    }
  }, [allDone])

  async function onSelectFolder() {
    setError(null)
    setJobs([])
    setPicking(true)
    try {
      const selected = await selectCameraFolder()
      if (!selected) return
      setFolder(selected)
      setScanning(true)
      const result = await api.previewImport(selected)
      setPreview(result)
      setKnown(result.known_cameras)
      setRows(
        result.camera_folders.map((item) => ({
          folder_path: item.folder_path,
          folder_name: item.folder_name,
          camera_id: item.suggested_camera_id,
          image_count: item.image_count,
          sample_names: item.sample_names,
          included: true,
          latitude: '',
          longitude: '',
          elevation: '',
          habitat: '',
        })),
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setPicking(false)
      setScanning(false)
    }
  }

  function updateRow(path: string, patch: Partial<CameraRow>) {
    setRows((current) => current.map((row) => (row.folder_path === path ? { ...row, ...patch } : row)))
  }

  async function startImport() {
    const cameras = rows
      .filter((row) => row.included && row.camera_id.trim())
      .map((row) => ({
        folder_path: row.folder_path,
        camera_id: row.camera_id.trim(),
        habitat: row.habitat || null,
        latitude: row.latitude === '' ? null : Number(row.latitude),
        longitude: row.longitude === '' ? null : Number(row.longitude),
        elevation: row.elevation === '' ? null : Number(row.elevation),
      }))
    if (cameras.length === 0) {
      setError('Include at least one camera folder and give it a camera ID.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const body = await api.importBatch(cameras)
      setJobs(body.jobs)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const totals = useMemo(() => {
    if (!preview) return null
    return { images: preview.total_images, cameras: preview.camera_count }
  }, [preview])

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Camera Trap Import</h2>
          <p>Select a local folder. Each camera subfolder is mapped to a camera ID, then sent through YOLO.</p>
        </div>
      </div>

      <article className="card form">
        <div className="field">
          <label>Local camera-trap folder</label>
          <button className="btn" type="button" onClick={onSelectFolder} disabled={picking || scanning}>
            {picking ? 'Opening folder picker…' : scanning ? 'Scanning images…' : 'Select Camera Trap Folder'}
          </button>
          <div className={`path-display ${folder ? 'ready' : ''}`}>
            {folder || 'No folder selected'}
          </div>
          {!desktop && (
            <p className="error">
              The Windows folder picker needs the desktop window. From the project folder run <code>npm run dev</code>.
            </p>
          )}
        </div>
        {totals && (
          <p className="muted" style={{ margin: 0 }}>
            {totals.images} images · {totals.cameras} camera folder{totals.cameras === 1 ? '' : 's'} discovered
          </p>
        )}
      </article>

      {rows.length > 0 && (
        <article className="card" style={{ marginTop: 16 }}>
          <h3>Discovered camera folders</h3>
          <p className="muted">
            Map each folder to a camera ID. New IDs are created in SQLite. Existing IDs reuse that camera.
            Selected: {selectedCount} cameras / {selectedImages} images.
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Import</th>
                <th>Folder</th>
                <th>Images</th>
                <th>Camera ID</th>
                <th>Existing</th>
                <th>Lat / Lon / Habitat</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const exists = known.includes(row.camera_id.trim())
                return (
                  <tr key={row.folder_path}>
                    <td>
                      <input
                        type="checkbox"
                        checked={row.included}
                        onChange={(event) => updateRow(row.folder_path, { included: event.target.checked })}
                      />
                    </td>
                    <td>
                      <strong>{row.folder_name}</strong>
                      <div className="muted" style={{ fontSize: 12 }}>{row.folder_path}</div>
                      {row.sample_names.length > 0 && (
                        <div className="muted" style={{ fontSize: 12 }}>{row.sample_names.join(', ')}</div>
                      )}
                    </td>
                    <td>{row.image_count}</td>
                    <td>
                      <input
                        value={row.camera_id}
                        onChange={(event) => updateRow(row.folder_path, { camera_id: event.target.value })}
                        list="known-cameras"
                        placeholder="Camera_01"
                      />
                    </td>
                    <td>
                      <span className={`badge ${exists ? 'ok' : 'warn'}`}>
                        {exists ? 'maps to existing' : 'will create'}
                      </span>
                    </td>
                    <td>
                      <div className="row-2">
                        <input
                          value={row.latitude}
                          onChange={(event) => updateRow(row.folder_path, { latitude: event.target.value })}
                          placeholder="lat"
                        />
                        <input
                          value={row.longitude}
                          onChange={(event) => updateRow(row.folder_path, { longitude: event.target.value })}
                          placeholder="lon"
                        />
                      </div>
                      <input
                        style={{ marginTop: 6 }}
                        value={row.habitat}
                        onChange={(event) => updateRow(row.folder_path, { habitat: event.target.value })}
                        placeholder="habitat (optional)"
                      />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <datalist id="known-cameras">
            {known.map((id) => (
              <option key={id} value={id} />
            ))}
          </datalist>
          <div className="actions" style={{ marginTop: 16 }}>
            <button className="btn" type="button" onClick={startImport} disabled={busy || selectedCount === 0}>
              {busy ? 'Starting YOLO…' : `Import ${selectedCount} camera folder${selectedCount === 1 ? '' : 's'}`}
            </button>
          </div>
        </article>
      )}

      {jobs.length > 0 && (
        <article className="card" style={{ marginTop: 16 }}>
          <h3>Import progress</h3>
          {jobs.map((job) => (
            <div key={job.job_id} style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                <strong>{job.camera_id}</strong>
                <span className={`badge ${job.status === 'completed' ? 'ok' : job.status === 'failed' ? 'bad' : 'warn'}`}>
                  {job.status} · {job.processed}/{job.total_images}
                </span>
              </div>
              <div className="progress" style={{ marginTop: 8 }}>
                <span style={{ width: `${jobPercent(job)}%` }} />
              </div>
            </div>
          ))}
          {allDone ? (
            <p className="ok">Import finished. Opening Human review for unidentified tiger crops…</p>
          ) : (
            <p className="muted">YOLO is running locally. Unidentified tiger crops will land on Human review.</p>
          )}
        </article>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  )
}

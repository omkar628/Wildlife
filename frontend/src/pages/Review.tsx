import { useEffect, useState } from 'react'
import { api, cropUrl, imageUrl, type ReviewItem, type TigerCatalogItem, type UnidentifiedObservation } from '../api'

const CHOICES = [
  { id: 'tiger', label: 'Tiger', className: 'btn tiger' },
  { id: 'prey', label: 'Prey', className: 'btn prey' },
  { id: 'rival', label: 'Rival', className: 'btn rival' },
  { id: 'human', label: 'Human', className: 'btn human' },
  { id: 'other', label: 'Other', className: 'btn ghost' },
  { id: 'ignore', label: 'Ignore', className: 'btn ignore' },
]

export default function ReviewPage() {
  const [items, setItems] = useState<ReviewItem[]>([])
  const [pending, setPending] = useState(0)
  const [unidentified, setUnidentified] = useState<UnidentifiedObservation[]>([])
  const [identityPending, setIdentityPending] = useState(0)
  const [catalog, setCatalog] = useState<TigerCatalogItem[]>([])
  const [nextId, setNextId] = useState('T001')
  const [selectedTiger, setSelectedTiger] = useState('')
  const [error, setError] = useState<string | null>(null)
  const current = items[0]
  const identityCurrent = unidentified[0]

  async function load() {
    const [classBody, identityBody] = await Promise.all([
      api.reviews(),
      api.unidentifiedObservations(),
    ])
    setItems(classBody.reviews)
    setPending(classBody.pending)
    setUnidentified(identityBody.observations)
    setIdentityPending(identityBody.pending)
    setCatalog(identityBody.tigers)
    setNextId(identityBody.next_tiger_id)
    const firstExisting = identityBody.tigers[0]?.tiger_id ?? ''
    setSelectedTiger((current) => current && identityBody.tigers.some((item) => item.tiger_id === current)
      ? current
      : firstExisting)
  }

  useEffect(() => {
    load().catch((err: Error) => setError(err.message))
  }, [])

  async function decide(humanClass: string) {
    if (!current) return
    setError(null)
    try {
      await api.decide(current.review_id, humanClass)
      await load()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function assignExisting() {
    if (!identityCurrent || !selectedTiger) return
    setError(null)
    try {
      await api.assignIdentity(identityCurrent.observation_id, {
        action: 'assign',
        tiger_id: selectedTiger,
      })
      await load()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function createNew() {
    if (!identityCurrent) return
    setError(null)
    try {
      await api.assignIdentity(identityCurrent.observation_id, { action: 'create' })
      await load()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const boxStyle = current && current.image_width && current.image_height
    ? {
        left: `${(current.bbox_x / current.image_width) * 100}%`,
        top: `${(current.bbox_y / current.image_height) * 100}%`,
        width: `${(current.bbox_width / current.image_width) * 100}%`,
        height: `${(current.bbox_height / current.image_height) * 100}%`,
      }
    : undefined

  const selected = catalog.find((item) => item.tiger_id === selectedTiger)

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Human review</h2>
          <p>
            {pending} low-confidence classes · {identityPending} unidentified tiger crops.
            Original files stay untouched.
          </p>
        </div>
      </div>

      {!current ? (
        <article className="card"><p className="empty">Class review queue is empty.</p></article>
      ) : (
        <div className="review-layout">
          <div className="frame">
            <img src={imageUrl(current.image_id)} alt={current.filename} />
            {boxStyle && <div className="box" style={boxStyle} />}
          </div>
          <article className="card">
            <h3>Predicted class</h3>
            <p style={{ fontSize: 28, margin: '0 0 8px', textTransform: 'capitalize' }}>{current.predicted_class}</p>
            <p className="muted">Confidence {Math.round(current.predicted_confidence * 100)}%</p>
            <p className="muted">{current.filename} · {current.camera_id ?? 'no camera'}</p>
            <p className="muted">{current.timestamp ?? 'no timestamp'}</p>
            <div className="actions" style={{ marginTop: 18 }}>
              {CHOICES.map((choice) => (
                <button key={choice.id} className={choice.className} type="button" onClick={() => decide(choice.id)}>
                  {choice.label}
                </button>
              ))}
            </div>
          </article>
        </div>
      )}

      <article className="card" style={{ marginTop: 16 }}>
        <h3>Tiger identity</h3>
        {!identityCurrent ? (
          <p className="empty">No unidentified tiger crops. Confirm a detection as tiger, or import a folder.</p>
        ) : (
          <div className="review-layout">
            <div className="frame">
              {identityCurrent.crop_path ? (
                <img src={cropUrl(identityCurrent.observation_id)} alt={`crop ${identityCurrent.observation_id}`} />
              ) : (
                <img src={imageUrl(identityCurrent.image_id)} alt={identityCurrent.filename} />
              )}
            </div>
            <div>
              <p className="muted" style={{ marginTop: 0 }}>
                Observation #{identityCurrent.observation_id} · {identityCurrent.camera_id ?? 'no camera'} · {identityCurrent.timestamp || identityCurrent.image_timestamp || 'no time'}
              </p>
              <div className="field">
                <label>Assign existing tiger</label>
                <select
                  value={selectedTiger}
                  onChange={(event) => setSelectedTiger(event.target.value)}
                  disabled={catalog.length === 0}
                >
                  {catalog.length === 0 ? <option value="">No field tigers yet</option> : null}
                  {catalog.map((item) => (
                    <option key={item.tiger_id} value={item.tiger_id}>
                      {item.tiger_id} · {item.observation_count} obs
                    </option>
                  ))}
                </select>
              </div>
              {selected && selected.references.length > 0 ? (
                <div style={{ marginTop: 12 }}>
                  <p className="muted">Previous reference crops for {selected.tiger_id}</p>
                  <div className="thumb-grid">
                    {selected.references.map((ref) => (
                      <article className="thumb" key={ref.observation_id}>
                        <img src={cropUrl(ref.observation_id)} alt={ref.tiger_id ?? ''} />
                        <div className="meta">
                          <strong>{ref.camera_id ?? '—'}</strong>
                          <div className="muted">{ref.timestamp ?? '—'}</div>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="muted">No reference crops for this ID yet.</p>
              )}
              <div className="actions" style={{ marginTop: 16 }}>
                <button className="btn" type="button" onClick={assignExisting} disabled={!selectedTiger}>
                  Assign existing tiger
                </button>
                <button className="btn ghost" type="button" onClick={createNew}>
                  Create new tiger ({nextId})
                </button>
              </div>
            </div>
          </div>
        )}
        {error && <p className="error">{error}</p>}
      </article>
    </div>
  )
}

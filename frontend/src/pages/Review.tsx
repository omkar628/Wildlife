import { useEffect, useState } from 'react'
import { api, imageUrl, type ReviewItem } from '../api'

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
  const [error, setError] = useState<string | null>(null)
  const current = items[0]

  async function load() {
    const body = await api.reviews()
    setItems(body.reviews)
    setPending(body.pending)
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

  const boxStyle = current && current.image_width && current.image_height
    ? {
        left: `${(current.bbox_x / current.image_width) * 100}%`,
        top: `${(current.bbox_y / current.image_height) * 100}%`,
        width: `${(current.bbox_width / current.image_width) * 100}%`,
        height: `${(current.bbox_height / current.image_height) * 100}%`,
      }
    : undefined

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Human review</h2>
          <p>{pending} low-confidence detections waiting. Original files stay untouched.</p>
        </div>
      </div>
      {!current ? (
        <article className="card"><p className="empty">Review queue is empty.</p></article>
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
            {error && <p className="error">{error}</p>}
          </article>
        </div>
      )}
    </div>
  )
}

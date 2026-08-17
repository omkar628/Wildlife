import { useEffect, useState } from 'react'
import { api, imageUrl, type ImageRow } from '../api'
import { EmptyPanel, LoadingPanel } from '../ui/Status'

export default function ImagesPage() {
  const [items, setItems] = useState<ImageRow[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.images()
      .then((body) => setItems(body.images))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Recent images</h2>
          <p>Processed files with detection counts. Source images are only read, never rewritten.</p>
        </div>
      </div>
      {loading ? <LoadingPanel title="Loading camera-trap images…" /> : null}
      {!loading && items.length === 0 ? (
        <article className="card">
          <EmptyPanel title="No images imported yet" detail="Select a camera-trap folder to begin analysis." />
        </article>
      ) : null}
      {!loading && items.length > 0 ? (
        <div className="thumb-grid">
          {items.map((item) => (
            <article className="thumb" key={item.image_id}>
              <img src={imageUrl(item.image_id)} alt={item.filename} />
              <div className="meta">
                <strong>{item.filename}</strong>
                <div className="muted">{item.camera_id} · {item.detection_count} boxes</div>
                <div className="muted">{item.timestamp_source}: {item.timestamp ?? '—'}</div>
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </div>
  )
}

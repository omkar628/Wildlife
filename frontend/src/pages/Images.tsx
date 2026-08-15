import { useEffect, useState } from 'react'
import { api, imageUrl, type ImageRow } from '../api'

export default function ImagesPage() {
  const [items, setItems] = useState<ImageRow[]>([])

  useEffect(() => {
    api.images().then((body) => setItems(body.images))
  }, [])

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Recent images</h2>
          <p>Processed files with detection counts. Source images are only read, never rewritten.</p>
        </div>
      </div>
      {items.length === 0 ? (
        <article className="card"><p className="empty">No images imported yet.</p></article>
      ) : (
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
      )}
    </div>
  )
}

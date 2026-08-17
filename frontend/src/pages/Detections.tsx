import { useEffect, useState } from 'react'
import { api, type Detection } from '../api'
import { EmptyPanel, LoadingPanel } from '../ui/Status'

export default function DetectionsPage() {
  const [items, setItems] = useState<Detection[]>([])
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    api.detections(filter || undefined)
      .then((body) => setItems(body.detections))
      .finally(() => setLoading(false))
  }, [filter])

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Detection summary</h2>
          <p>
            Every saved box, with the confidence used for auto-accept vs review.
            {!loading ? ` ${items.length} detection${items.length === 1 ? '' : 's'}.` : ''}
          </p>
        </div>
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="">All classes</option>
          <option value="tiger">Tiger</option>
          <option value="prey">Prey</option>
          <option value="rival">Rival</option>
          <option value="human">Human</option>
        </select>
      </div>
      <article className="card">
        {loading ? <LoadingPanel title="Identifying wildlife…" detail="Reading stored detections." /> : null}
        {!loading && items.length === 0 ? <EmptyPanel title="No detections yet" detail="Import a camera folder to run YOLO." /> : null}
        {!loading && items.length > 0 ? (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th><th>Class</th><th>Camera</th><th>Time</th><th>Tiger</th><th>Conf</th><th>Review</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.detection_id}>
                  <td>{item.detection_id}</td>
                  <td>
                    <span className={`badge ${item.final_class_name || item.class_name}`}>
                      {item.final_class_name || item.class_name}
                    </span>
                  </td>
                  <td>{item.camera_id ?? '—'}</td>
                  <td>{item.timestamp ?? '—'}</td>
                  <td>{item.tiger_id ?? '—'}</td>
                  <td>{Math.round(item.confidence * 100)}%</td>
                  <td>{item.review_status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </article>
    </div>
  )
}

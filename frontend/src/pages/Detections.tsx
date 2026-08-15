import { useEffect, useState } from 'react'
import { api, type Detection } from '../api'

export default function DetectionsPage() {
  const [items, setItems] = useState<Detection[]>([])
  const [filter, setFilter] = useState('')

  useEffect(() => {
    api.detections(filter || undefined).then((body) => setItems(body.detections))
  }, [filter])

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Detection summary</h2>
          <p>Every saved box, with the confidence used for auto-accept vs review.</p>
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
        {items.length === 0 ? <p className="empty">No detections yet.</p> : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th><th>File</th><th>Camera</th><th>Class</th><th>Conf</th><th>Review</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.detection_id}>
                  <td>{item.detection_id}</td>
                  <td>{item.filename}</td>
                  <td>{item.camera_id}</td>
                  <td>
                    <span className={`badge ${item.final_class_name || item.class_name}`}>
                      {item.final_class_name || item.class_name}
                    </span>
                  </td>
                  <td>{Math.round(item.confidence * 100)}%</td>
                  <td>{item.review_status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </article>
    </div>
  )
}

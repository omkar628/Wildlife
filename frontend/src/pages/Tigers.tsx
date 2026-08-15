import { useEffect, useState } from 'react'
import { api, type TigerRow } from '../api'

export default function TigersPage() {
  const [items, setItems] = useState<TigerRow[]>([])

  useEffect(() => {
    api.tigers().then((body) => setItems(body.tigers))
  }, [])

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Tigers</h2>
          <p>Individual IDs appear here after Re-ID is connected. Crops are already being stored.</p>
        </div>
      </div>
      <article className="card">
        {items.length === 0 ? (
          <p className="empty">
            No identified tigers yet. The Re-ID adapter is in place, but inference code was not in this repository.
          </p>
        ) : (
          <table className="table">
            <thead>
              <tr><th>Tiger</th><th>First seen</th><th>Last seen</th><th>Observations</th></tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.tiger_id}>
                  <td>{item.tiger_id}</td>
                  <td>{item.first_seen}</td>
                  <td>{item.last_seen}</td>
                  <td>{item.observation_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </article>
    </div>
  )
}

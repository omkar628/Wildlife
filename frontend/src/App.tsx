import { useEffect, useMemo, useState } from 'react'
import { api, type Health } from './api'
import { isDesktopApp } from './desktop'
import DashboardPage from './pages/Dashboard'
import ImportPage from './pages/ImportFolder'
import JobsPage from './pages/Jobs'
import DetectionsPage from './pages/Detections'
import ReviewPage from './pages/Review'
import ImagesPage from './pages/Images'
import TigersPage from './pages/Tigers'
import GraphPage from './pages/GraphView'
import CamerasPage from './pages/Cameras'

const ROUTES = [
  { hash: '#/', label: 'Dashboard' },
  { hash: '#/cameras', label: 'Cameras' },
  { hash: '#/import', label: 'Import' },
  { hash: '#/jobs', label: 'Processing' },
  { hash: '#/detections', label: 'Detections' },
  { hash: '#/review', label: 'Review' },
  { hash: '#/images', label: 'Images' },
  { hash: '#/tigers', label: 'Tigers' },
  { hash: '#/graph', label: 'Movement map' },
]

export default function App() {
  const [hash, setHash] = useState(window.location.hash || '#/')
  const [health, setHealth] = useState<Health | null>(null)
  const [pending, setPending] = useState(0)

  useEffect(() => {
    const onHash = () => setHash(window.location.hash || '#/')
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
    const tick = () => {
      api.dashboard()
        .then((d) => setPending(d.review.pending + (d.review.unidentified_tigers ?? 0)))
        .catch(() => undefined)
    }
    tick()
    const id = window.setInterval(tick, 6000)
    return () => window.clearInterval(id)
  }, [])

  const page = useMemo(() => hash.split('?')[0], [hash])

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="mark">W</div>
          <div>
            <h1>Wildlife Intelligence</h1>
            <p>Offline field station</p>
          </div>
        </div>
        <nav className="nav">
          {ROUTES.map((route) => (
            <a
              key={route.hash}
              href={route.hash}
              className={page === route.hash ? 'active' : ''}
            >
              <span>{route.label}</span>
              {route.hash === '#/review' && pending > 0 ? <span className="badge warn">{pending}</span> : null}
            </a>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div>
            <span className={`dot ${isDesktopApp() ? 'ok' : 'warn'}`} />
            {isDesktopApp() ? 'Desktop app' : 'Browser preview'}
          </div>
          <div style={{ marginTop: 8 }}>
            <span className={`dot ${health?.detector.available ? 'ok' : 'warn'}`} />
            Detector {health?.detector.available ? 'ready' : 'weights missing'}
          </div>
          <div style={{ marginTop: 8 }}>
            <span className={`dot ${health?.reid?.loaded ? 'ok' : 'warn'}`} />
            Identity {health?.reid?.loaded ? `MegaDescriptor ${health.reid.device ?? ''}` : 'human confirm'}
          </div>
          <div style={{ marginTop: 8 }}>
            <span className={`dot ${health?.gnn?.loaded ? 'ok' : 'warn'}`} />
            GNN {health?.gnn?.loaded ? `${health.gnn.device}` : 'unavailable'}
          </div>
        </div>
      </aside>
      <main className="main">
        {page === '#/' && <DashboardPage />}
        {page === '#/cameras' && <CamerasPage />}
        {page === '#/import' && <ImportPage />}
        {page === '#/jobs' && <JobsPage />}
        {page === '#/detections' && <DetectionsPage />}
        {page === '#/review' && <ReviewPage />}
        {page === '#/images' && <ImagesPage />}
        {page === '#/tigers' && <TigersPage />}
        {page === '#/graph' && <GraphPage />}
      </main>
    </div>
  )
}

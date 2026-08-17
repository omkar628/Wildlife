import type { ReactNode } from 'react'
import { UI_IMAGES } from './assets'

export function LoadingPanel({
  title,
  detail,
}: {
  title: string
  detail?: string
}) {
  return (
    <div className="status-panel" role="status" aria-live="polite">
      <div className="status-spinner" aria-hidden="true" />
      <div>
        <p className="status-title">{title}</p>
        {detail ? <p className="muted">{detail}</p> : null}
      </div>
    </div>
  )
}

export function EmptyPanel({
  title,
  detail,
  action,
  image = UI_IMAGES.path,
}: {
  title: string
  detail?: string
  action?: ReactNode
  image?: string
}) {
  return (
    <div className="status-panel empty-panel">
      <img className="status-photo" src={image} alt="" loading="lazy" />
      <div>
        <p className="status-title">{title}</p>
        {detail ? <p className="muted">{detail}</p> : null}
        {action ? <div className="actions" style={{ marginTop: 12 }}>{action}</div> : null}
      </div>
    </div>
  )
}

export function ErrorPanel({
  title = 'Something went wrong',
  detail,
}: {
  title?: string
  detail?: string
}) {
  return (
    <div className="status-panel error-panel" role="alert">
      <p className="status-title">{title}</p>
      {detail ? <p className="error">{detail}</p> : null}
    </div>
  )
}

export function SkeletonBlock({ rows = 4 }: { rows?: number }) {
  return (
    <div className="skeleton-stack" aria-hidden="true">
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="skeleton-line" style={{ width: `${86 - index * 8}%` }} />
      ))}
    </div>
  )
}

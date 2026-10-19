import { Plus, Trash, X } from '@phosphor-icons/react'
import type { SessionSummary } from '../types/session'

interface SessionListProps {
  sessions: SessionSummary[]
  currentSessionId: string | null
  disabled: boolean
  onCreate: () => void
  onSelect: (sessionId: string) => void
  onDelete: (sessionId: string) => void
  onClose: () => void
}

function sessionGroup(date: string) {
  const time = new Date(date)
  if (Number.isNaN(time.getTime())) return '更早'
  const today = new Date()
  const yesterday = new Date(today.getFullYear(), today.getMonth(), today.getDate() - 1)
  if (time.toDateString() === today.toDateString()) return '今天'
  if (time.toDateString() === yesterday.toDateString()) return '昨天'
  return '更早'
}

export function SessionList({ sessions, currentSessionId, disabled, onCreate, onSelect, onDelete, onClose }: SessionListProps) {
  const groups = ['今天', '昨天', '更早'].map((name) => ({ name, sessions: sessions.filter((session) => sessionGroup(session.updated_at) === name) }))
  return (
    <aside className="session-panel" aria-label="会话列表" id="session-panel">
      <div className="brand"><span className="logo">PW</span><div><b>PickWise</b><small>选购助手</small></div><button className="icon-button panel-close" type="button" onClick={onClose} title="关闭会话列表" aria-label="关闭会话列表"><X size={18} /></button></div>
      <button className="new-session-button" type="button" onClick={onCreate} disabled={disabled}><Plus size={17} weight="bold" aria-hidden="true" /> 新建会话</button>
      <nav className="session-list" aria-label="历史会话">
        {!sessions.length && <p className="empty-state">还没有会话</p>}
        {groups.filter((group) => group.sessions.length > 0).map((group) => <div key={group.name}><h2 className="session-group">{group.name}</h2>{group.sessions.map((session) => (
          <div className={`session-item ${session.session_id === currentSessionId ? 'is-active' : ''}`} key={session.session_id}>
            <button className="session-select" type="button" disabled={disabled} aria-current={session.session_id === currentSessionId ? 'page' : undefined} onClick={() => onSelect(session.session_id)} title={session.title || '新会话'}>
              <strong>{session.title || '新会话'}</strong><span>{session.message_count} 条可见消息</span>
            </button>
            <button className="session-delete icon-button" type="button" disabled={disabled} onClick={() => onDelete(session.session_id)} aria-label={`删除${session.title || '会话'}`} title="删除会话"><Trash size={14} aria-hidden="true" /></button>
          </div>
        ))}</div>)}
      </nav>
      <div className="session-footer"><span>决策记录</span><strong>{String(sessions.length).padStart(2, '0')}</strong></div>
    </aside>
  )
}

import type { AnonymousIdentity, SessionSummary, VisibleMessage } from '../types/session'
import type { RecommendedProduct } from '../types/product'
import { requestJson, setCsrfToken } from './client'

export function initializeIdentity(): Promise<AnonymousIdentity> {
  return requestJson<AnonymousIdentity>('/api/auth/anonymous', { method: 'POST' }).then((identity) => {
    setCsrfToken(identity.csrf_token)
    return identity
  })
}

export function listSessions(): Promise<{ sessions: SessionSummary[] }> {
  return requestJson('/api/sessions')
}

export function createSession(title = '新会话'): Promise<{ session_id: string; title: string }> {
  return requestJson('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({ title }),
  })
}

export async function loadMessages(sessionId: string): Promise<{
  session_id: string
  messages: VisibleMessage[]
  recommendations: RecommendedProduct[]
}> {
  const result = await requestJson<{
    session_id: string
    messages: VisibleMessage[]
    recommendations?: RecommendedProduct[]
  }>(`/api/sessions/${encodeURIComponent(sessionId)}/messages`)
  return { ...result, recommendations: Array.isArray(result.recommendations) ? result.recommendations : [] }
}

export function deleteSession(sessionId: string): Promise<{ session_id: string; deleted: boolean }> {
  return requestJson(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
}

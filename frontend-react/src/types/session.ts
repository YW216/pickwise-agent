export type MessageRole = 'user' | 'assistant'

export interface VisibleMessage {
  message_id?: string
  role: MessageRole
  content: string
  created_at?: string | null
}

export interface SessionSummary {
  session_id: string
  title: string
  status: string
  message_count: number
  created_at: string
  updated_at: string
  last_message_at: string | null
}

export interface AnonymousIdentity {
  user_id: string
  csrf_token: string
  expires_at: string
}

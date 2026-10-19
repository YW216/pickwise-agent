export type PhaseState = 'running' | 'done' | 'partial' | 'failed'

export interface ExecutionDetail {
  key: string
  title: string
  state: PhaseState
  message: string
  count?: number
}

export interface ExecutionPhase {
  key: string
  title: string
  state: PhaseState
  message: string
  count?: number
  details: ExecutionDetail[]
}

export interface ChatStreamEvent {
  event: string
  data: Record<string, unknown>
}

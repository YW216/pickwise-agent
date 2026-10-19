import type { ChatStreamEvent, ExecutionDetail, ExecutionPhase, PhaseState } from '../types/chat'
import type { SessionSummary, VisibleMessage } from '../types/session'

export interface ChatState {
  sessions: SessionSummary[]
  currentSessionId: string | null
  messages: VisibleMessage[]
  initialized: boolean
  loading: boolean
  streaming: boolean
  execution: ExecutionPhase[]
  error: string | null
}

export const initialChatState: ChatState = {
  sessions: [],
  currentSessionId: null,
  messages: [],
  initialized: false,
  loading: true,
  streaming: false,
  execution: [],
  error: null,
}

type Action =
  | { type: 'INITIALIZED' }
  | { type: 'SESSIONS_LOADED'; sessions: SessionSummary[] }
  | { type: 'SESSION_SELECTED'; sessionId: string }
  | { type: 'MESSAGES_LOADED'; messages: VisibleMessage[] }
  | { type: 'STREAM_STARTED'; message: string }
  | { type: 'STREAM_EVENT'; event: ChatStreamEvent }
  | { type: 'STREAM_STOPPED'; message: string }
  | { type: 'ERROR'; message: string }
  | { type: 'CLEAR_ERROR' }

const phaseTitles: Record<string, string> = {
  requirements: '理解需求',
  products: '筛选商品',
  knowledge: '查询资料',
  synthesis: '生成建议',
  persist: '保存会话',
}

function text(data: Record<string, unknown>, key: string, fallback = ''): string {
  const value = data[key]
  return typeof value === 'string' ? value : fallback
}

function number(data: Record<string, unknown>, key: string): number | undefined {
  const value = data[key]
  return typeof value === 'number' ? value : undefined
}

function phaseFor(data: Record<string, unknown>): string {
  const explicit = text(data, 'phase')
  if (phaseTitles[explicit]) return explicit
  const stage = text(data, 'stage')
  if (stage === 'routing') return 'requirements'
  if (stage === 'result') return 'synthesis'
  if (stage === 'save') return 'persist'
  const tool = text(data, 'tool')
  if (tool === 'retrieve_knowledge') return 'knowledge'
  if (tool) return 'products'
  if (text(data, 'agent') === 'consult') return 'knowledge'
  if (text(data, 'agent') === 'presale') return 'products'
  return 'synthesis'
}

function eventState(event: ChatStreamEvent): PhaseState {
  if (event.event === 'tool.started') return 'running'
  if (event.event === 'tool.finished') return 'done'
  if (event.event === 'tool.failed') {
    return text(event.data, 'state', 'failed') as PhaseState
  }
  const state = text(event.data, 'state', 'running') as PhaseState
  return ['running', 'done', 'partial', 'failed'].includes(state) ? state : 'running'
}

function updateExecution(
  execution: ExecutionPhase[],
  streamEvent: ChatStreamEvent,
): ExecutionPhase[] {
  if (!['status', 'tool.started', 'tool.finished', 'tool.failed'].includes(streamEvent.event)) {
    return execution
  }
  const data = streamEvent.data
  const key = phaseFor(data)
  const state = eventState(streamEvent)
  const message = text(data, 'message', text(data, 'summary', '处理中'))
  const count = number(data, 'count')
  const phaseIndex = execution.findIndex((item) => item.key === key)
  const phase: ExecutionPhase = phaseIndex >= 0
    ? { ...execution[phaseIndex], state, message, ...(count === undefined ? {} : { count }) }
    : {
        key,
        title: text(data, 'title', phaseTitles[key] || '处理步骤'),
        state,
        message,
        ...(count === undefined ? {} : { count }),
        details: [],
      }

  if (streamEvent.event.startsWith('tool.')) {
    const detailKey = text(data, 'call_id', `${key}:${text(data, 'tool', message)}`)
    const detail: ExecutionDetail = {
      key: detailKey,
      title: text(data, 'tool_title', text(data, 'tool', '查询')),
      state,
      message,
      ...(count === undefined ? {} : { count }),
    }
    const detailIndex = phase.details.findIndex((item) => item.key === detailKey)
    const details = detailIndex >= 0
      ? phase.details.map((item, index) => (index === detailIndex ? detail : item))
      : [...phase.details, detail]
    phase.details = details
  }

  if (phaseIndex >= 0) return execution.map((item, index) => (index === phaseIndex ? phase : item))
  return [...execution, phase]
}

function replaceLastAssistant(messages: VisibleMessage[], content: string): VisibleMessage[] {
  const last = messages.length - 1
  if (last < 0 || messages[last].role !== 'assistant') return messages
  return messages.map((message, index) => (index === last ? { ...message, content } : message))
}

export function chatReducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case 'INITIALIZED':
      return { ...state, initialized: true, error: null }
    case 'SESSIONS_LOADED':
      return { ...state, sessions: action.sessions }
    case 'SESSION_SELECTED':
      return { ...state, currentSessionId: action.sessionId, messages: [], execution: [], loading: true, error: null }
    case 'MESSAGES_LOADED':
      return { ...state, messages: action.messages, loading: false, error: null }
    case 'STREAM_STARTED':
      return {
        ...state,
        messages: [
          ...state.messages,
          { role: 'user', content: action.message },
          { role: 'assistant', content: '' },
        ],
        streaming: true,
        execution: [],
        error: null,
      }
    case 'STREAM_EVENT': {
      const { event, data } = action.event
      if (event === 'answer.delta') {
        return {
          ...state,
          messages: replaceLastAssistant(
            state.messages,
            `${state.messages[state.messages.length - 1]?.content || ''}${text(data, 'text')}`,
          ),
        }
      }
      if (event === 'done') {
        return {
          ...state,
          messages: replaceLastAssistant(state.messages, text(data, 'reply')),
          streaming: false,
        }
      }
      if (event === 'error') {
        return { ...state, streaming: false, error: text(data, 'message', '本轮处理失败') }
      }
      return { ...state, execution: updateExecution(state.execution, action.event) }
    }
    case 'STREAM_STOPPED':
      return { ...state, streaming: false, error: action.message }
    case 'ERROR':
      return { ...state, loading: false, streaming: false, error: action.message }
    case 'CLEAR_ERROR':
      return { ...state, error: null }
    default:
      return state
  }
}

export type { Action as ChatAction }

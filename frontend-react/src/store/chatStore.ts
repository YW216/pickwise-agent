import type { ChatStreamEvent, ExecutionDetail, ExecutionPhase, PhaseState } from '../types/chat'
import type { SessionSummary, VisibleMessage } from '../types/session'
import type { RecommendedProduct } from '../types/product'

export interface ChatState {
  sessions: SessionSummary[]
  currentSessionId: string | null
  messages: VisibleMessage[]
  initialized: boolean
  loading: boolean
  streaming: boolean
  execution: ExecutionPhase[]
  error: string | null
  recommendations: RecommendedProduct[]
  selectedRefs: string[]
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
  recommendations: [],
  selectedRefs: [],
}

type Action =
  | { type: 'INITIALIZED' }
  | { type: 'SESSIONS_LOADED'; sessions: SessionSummary[] }
  | { type: 'SESSION_SELECTED'; sessionId: string }
  | { type: 'MESSAGES_LOADED'; messages: VisibleMessage[]; recommendations: RecommendedProduct[] }
  | { type: 'STREAM_STARTED'; message: string }
  | { type: 'STREAM_EVENT'; event: ChatStreamEvent }
  | { type: 'TOGGLE_COMPARE'; ref: string }
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

function normalizeProducts(items: unknown[]): RecommendedProduct[] {
  return items.filter((item): item is RecommendedProduct => {
    if (!item || typeof item !== 'object') return false
    const value = item as Record<string, unknown>
    return typeof value.ref === 'string' && typeof value.display_no === 'number'
      && typeof value.name === 'string'
  }).map((item) => ({
    ref: String(item.ref),
    display_no: Number(item.display_no),
    name: String(item.name),
    brand: String(item.brand || ''),
    category: String(item.category || '商品'),
    price: Number(item.price || 0),
    summary: String(item.summary || ''),
    reason: String(item.reason || '本轮回答中提到'),
    image: typeof item.image === 'string' ? item.image : undefined,
    best: item.best === true,
  }))
}

function recommendations(data: Record<string, unknown>): RecommendedProduct[] {
  return Array.isArray(data.products) ? normalizeProducts(data.products) : []
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
      return { ...state, currentSessionId: action.sessionId, messages: [], execution: [], recommendations: [], selectedRefs: [], loading: true, error: null }
    case 'MESSAGES_LOADED':
      return { ...state, messages: action.messages, recommendations: action.recommendations, loading: false, error: null }
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
        selectedRefs: [],
        error: null,
      }
    case 'STREAM_EVENT': {
      const { event, data } = action.event
      if (event === 'recommendations') {
        const incoming = recommendations(data)
        // 同一商品可能在不同轮次以不同 ref 出现；名称是用户看到的唯一身份。
        const seenNames = new Set<string>()
        const merged = [...incoming, ...state.recommendations].filter((item) => {
          const key = item.name.replace(/\s+/g, '').toLowerCase()
          if (seenNames.has(key)) return false
          seenNames.add(key)
          return true
        })
        return { ...state, recommendations: merged }
      }
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
    case 'TOGGLE_COMPARE': {
      const selected = state.selectedRefs.includes(action.ref)
        ? state.selectedRefs.filter((ref) => ref !== action.ref)
        : state.selectedRefs.length < 3 ? [...state.selectedRefs, action.ref] : state.selectedRefs
      return { ...state, selectedRefs: selected }
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

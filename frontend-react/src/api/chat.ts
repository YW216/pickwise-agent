import { csrfHeaders } from './client'
import type { ChatStreamEvent } from '../types/chat'

export interface ChatRequest {
  session_id: string
  message: string
  /** 右侧推荐区的选中引用；服务端凭此还原 product_id，前端不接触内部 ID。 */
  product_refs?: string[]
}

export type StreamEventHandler = (event: ChatStreamEvent) => void

function streamError(payload: unknown, status: number): Error {
  if (payload && typeof payload === 'object' && 'error' in payload) {
    const error = (payload as { error?: { message?: string } | string }).error
    if (typeof error === 'string') return new Error(error)
    if (error?.message) return new Error(error.message)
  }
  return new Error(`请求失败（${status}）`)
}

function dispatchFrame(eventName: string, dataLines: string[], onEvent: StreamEventHandler) {
  if (!dataLines.length) return
  const data = JSON.parse(dataLines.join('\n')) as Record<string, unknown>
  onEvent({ event: eventName || 'message', data })
}

export async function streamChat(
  body: ChatRequest,
  idempotencyKey: string,
  signal: AbortSignal,
  onEvent: StreamEventHandler,
) {
  const headers = csrfHeaders({
    Accept: 'text/event-stream',
    'Content-Type': 'application/json',
    'Idempotency-Key': idempotencyKey,
  })
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers,
    credentials: 'same-origin',
    signal,
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw streamError(payload, response.status)
  }
  if (!response.body) throw new Error('服务未返回事件流')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let eventName = ''
  let dataLines: string[] = []
  let completed = false

  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      let lineEnd = buffer.indexOf('\n')
      while (lineEnd >= 0) {
        const line = buffer.slice(0, lineEnd).replace(/\r$/, '')
        buffer = buffer.slice(lineEnd + 1)
        if (!line) {
          dispatchFrame(eventName, dataLines, onEvent)
          if (eventName === 'done') completed = true
          eventName = ''
          dataLines = []
        } else if (line.startsWith(':')) {
          // SSE 心跳注释。
        } else if (line.startsWith('event:')) {
          eventName = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).replace(/^ /, ''))
        }
        lineEnd = buffer.indexOf('\n')
      }
      if (done) break
    }
  } finally {
    await reader.cancel().catch(() => undefined)
    reader.releaseLock()
  }

  if (!completed) throw new Error('连接中断；请刷新会话查看后台结果')
}

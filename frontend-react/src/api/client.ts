export interface ApiEnvelope<T> {
  success: boolean
  data?: T
  error?: { code?: string; message?: string } | string | null
  request_id?: string
}

export class ApiError extends Error {
  status: number
  code?: string
  requestId?: string

  constructor(message: string, status: number, code?: string, requestId?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.requestId = requestId
  }
}

let csrfToken = ''

export function setCsrfToken(token: string) {
  csrfToken = token
}

export function csrfHeaders(headers: HeadersInit = {}): Headers {
  const result = new Headers(headers)
  if (csrfToken) result.set('X-Pickwise-CSRF', csrfToken)
  return result
}

function errorMessage(payload: ApiEnvelope<unknown>, status: number): string {
  if (typeof payload.error === 'string') return payload.error
  return payload.error?.message || `请求失败（${status}）`
}

export async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = (init.method || 'GET').toUpperCase()
  const headers = csrfHeaders(init.headers)
  headers.set('Accept', 'application/json')
  if (method !== 'GET' && method !== 'HEAD' && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(path, {
    ...init,
    headers,
    credentials: 'same-origin',
  })
  const payload = (await response.json().catch(() => ({}))) as ApiEnvelope<T>
  if (!response.ok || payload.success !== true) {
    throw new ApiError(
      errorMessage(payload, response.status),
      response.status,
      typeof payload.error === 'object' ? payload.error?.code : undefined,
      payload.request_id,
    )
  }
  return payload.data as T
}

import { requestJson, setCsrfToken } from './client'

export interface AuthUser {
  user_id: string
  email: string
  display_name: string
}

export interface AuthState {
  authenticated: boolean
  user: AuthUser | null
  csrf_token?: string
  expires_at?: number
}

export interface AuthInput {
  email: string
  password: string
  display_name?: string
}

export async function loadAuthState(): Promise<AuthState> {
  const result = await requestJson<AuthState>('/api/auth/me')
  setCsrfToken(result.csrf_token || '')
  return result
}

export async function login(input: AuthInput): Promise<AuthState> {
  const result = await requestJson<AuthState>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify(input),
  })
  setCsrfToken(result.csrf_token || '')
  return result
}

export async function register(input: AuthInput): Promise<AuthState> {
  const result = await requestJson<AuthState>('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify(input),
  })
  setCsrfToken(result.csrf_token || '')
  return result
}

export async function logout(): Promise<void> {
  await requestJson('/api/auth/logout', { method: 'POST' })
  setCsrfToken('')
}

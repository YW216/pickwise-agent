import { useState, type FormEvent } from 'react'
import { X } from '@phosphor-icons/react'
import { login, register, type AuthState } from '../api/auth'

interface AuthModalProps {
  open: boolean
  onClose: () => void
  onAuthenticated: (state: AuthState) => void
}

export function AuthModal({ open, onClose, onAuthenticated }: AuthModalProps) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  if (!open) return null

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const result = mode === 'login'
        ? await login({ email, password })
        : await register({ email, password, display_name: displayName })
      onAuthenticated(result)
      setPassword('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '认证失败，请重试')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <section className="auth-dialog" role="dialog" aria-modal="true" aria-labelledby="auth-title">
        <header className="auth-dialog-header">
          <div><span>PickWise 账号</span><h2 id="auth-title">{mode === 'login' ? '登录后继续' : '创建账号'}</h2></div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="关闭登录窗口"><X size={20} /></button>
        </header>
        <div className="auth-tabs" role="tablist" aria-label="认证方式">
          <button type="button" role="tab" aria-selected={mode === 'login'} className={mode === 'login' ? 'is-active' : ''} onClick={() => { setMode('login'); setError('') }}>登录</button>
          <button type="button" role="tab" aria-selected={mode === 'register'} className={mode === 'register' ? 'is-active' : ''} onClick={() => { setMode('register'); setError('') }}>注册</button>
        </div>
        <form className="auth-form" onSubmit={(event) => void submit(event)}>
          {mode === 'register' && <label>昵称<input autoComplete="nickname" value={displayName} maxLength={80} onChange={(event) => setDisplayName(event.target.value)} required /></label>}
          <label>邮箱<input type="email" autoComplete="email" value={email} maxLength={254} onChange={(event) => setEmail(event.target.value)} required /></label>
          <label>密码<input type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} value={password} minLength={mode === 'login' ? 1 : 10} maxLength={128} onChange={(event) => setPassword(event.target.value)} required /></label>
          {error && <p className="auth-error" role="alert">{error}</p>}
          <button className="auth-submit" type="submit" disabled={busy}>{busy ? '正在处理…' : mode === 'login' ? '登录' : '创建账号'}</button>
        </form>
      </section>
    </div>
  )
}

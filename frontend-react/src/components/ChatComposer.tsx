import { Stop, ArrowUpRight } from '@phosphor-icons/react'
import type { FormEvent } from 'react'

interface ChatComposerProps {
  disabled: boolean
  streaming: boolean
  value: string
  onChange: (value: string) => void
  onSubmit: (message: string) => void
  onStop: () => void
}

export function ChatComposer({ disabled, streaming, value, onChange, onSubmit, onStop }: ChatComposerProps) {
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (disabled || streaming || !value.trim()) return
    onSubmit(value.trim())
    onChange('')
  }
  return (
    <form className="composer" onSubmit={submit}>
      <div className="composer-input">
        <label className="sr-only" htmlFor="message-input">输入你的需求</label>
        <textarea id="message-input" rows={1} maxLength={8000} value={value} onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }}
          placeholder="说说你的预算和用途" disabled={disabled || streaming} />
        {streaming ? <button className="send-button is-stop" type="button" onClick={onStop} title="停止接收" aria-label="停止接收"><Stop size={18} weight="fill" /><span>停止</span></button>
          : <button className="send-button" type="submit" disabled={disabled || !value.trim()}><span>发送</span><ArrowUpRight size={16} weight="bold" aria-hidden="true" /></button>}
      </div>
    </form>
  )
}

import { useState } from 'react'
import { Check, Copy } from '@phosphor-icons/react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { VisibleMessage } from '../types/session'
import type { ExecutionPhase } from '../types/chat'
import { ExecutionTimeline } from './ExecutionTimeline'

interface ChatMessageProps {
  message: VisibleMessage
  execution?: ExecutionPhase[]
  streaming?: boolean
}

export function ChatMessage({ message, execution = [], streaming = false }: ChatMessageProps) {
  const [copied, setCopied] = useState(false)
  const [copyError, setCopyError] = useState(false)
  async function copy() {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setCopyError(false)
    } catch {
      setCopyError(true)
    }
  }
  return (
    <article className={`message message-${message.role}`}>
      {message.role === 'assistant' && <span className="assistant-avatar" aria-hidden="true">PW</span>}
      <div className="message-content">
        <span className="message-label">{message.role === 'user' ? '你' : 'PickWise'}</span>
        {message.role === 'assistant' ? (
          <>
            <ExecutionTimeline key={streaming ? 'streaming' : 'done'} phases={execution} streaming={streaming} />
            <div className="message-markdown">
              {message.content ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml components={{
                  table: ({ children, ...props }) => <div className="table-scroll" tabIndex={0} role="region" aria-label="商品对比表"><table {...props}>{children}</table></div>,
                  a: ({ children, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer">{children}</a>,
                }}>{message.content}</ReactMarkdown>
              ) : <div className="answer-loading" role="status"><span /><span /><span /><span className="sr-only">正在处理</span></div>}
            </div>
            {message.content && !streaming && <div className="message-actions"><button className="icon-button" type="button" onClick={() => void copy()} aria-label={copied ? '已复制回答' : '复制回答'} title={copied ? '已复制' : '复制回答'}>{copied ? <Check size={16} /> : <Copy size={16} />}</button>{copyError && <span role="status">无法访问剪贴板</span>}</div>}
          </>
        ) : <p>{message.content}</p>}
      </div>
    </article>
  )
}

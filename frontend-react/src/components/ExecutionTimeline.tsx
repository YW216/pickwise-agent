import { useState } from 'react'
import type { ExecutionPhase } from '../types/chat'

interface ExecutionTimelineProps {
  phases: ExecutionPhase[]
  streaming: boolean
}

function summary(phases: ExecutionPhase[]): string {
  return phases.map((phase) => {
    if (phase.count !== undefined && phase.key === 'products') return `筛选 ${phase.count} 款商品`
    if (phase.count !== undefined && phase.key === 'knowledge') return `参考 ${phase.count} 条资料`
    return phase.title
  }).join(' · ') || '正在处理'
}

export function ExecutionTimeline({ phases, streaming }: ExecutionTimelineProps) {
  const [open, setOpen] = useState(streaming)
  if (!phases.length) return null

  return (
    <details className="execution-details" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>{streaming ? '正在处理 · ' : ''}{summary(phases)}</summary>
      <ol className="execution-phases">
        {phases.map((phase) => (
          <li className="execution-phase" data-state={phase.state} key={phase.key}>
            <div className="execution-phase-head">
              <strong>{phase.title}</strong>
              <span>{phase.state === 'done' ? '已完成' : phase.state === 'partial' ? '部分完成' : phase.state === 'failed' ? '未完成' : '进行中'}</span>
            </div>
            <div className="execution-phase-message">{phase.message}</div>
            {phase.details.length > 0 && (
              <ul className="execution-phase-details">
                {phase.details.map((detail) => (
                  <li data-state={detail.state} key={detail.key}>
                    {detail.title}：{detail.message}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ol>
    </details>
  )
}

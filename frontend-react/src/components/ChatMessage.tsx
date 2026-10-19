import { useMemo, useState, type ReactNode } from 'react'
import { Check, Copy } from '@phosphor-icons/react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { VisibleMessage } from '../types/session'
import type { ExecutionPhase } from '../types/chat'
import type { RecommendedProduct } from '../types/product'
import { ExecutionTimeline } from './ExecutionTimeline'

interface ChatMessageProps {
  message: VisibleMessage
  execution?: ExecutionPhase[]
  streaming?: boolean
  /** 本会话出现过的商品：用于把回复里的商品渲染成卡片。 */
  products?: RecommendedProduct[]
  selectedRefs?: string[]
  onDetail?: (product: RecommendedProduct) => void
  onToggle?: (ref: string) => void
}

/** 单元格里可能嵌着 **加粗**，先把所有文本节点拼起来再匹配。 */
function nodeText(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(nodeText).join('')
  if (node && typeof node === 'object' && 'props' in node) {
    return nodeText((node as { props?: { children?: ReactNode } }).props?.children)
  }
  return ''
}

function normalize(value: string): string {
  return value.replace(/\s+/g, '').toLowerCase()
}

function elementChildren(node: ReactNode): ReactNode {
  if (Array.isArray(node)) return node
  return node && typeof node === 'object' && 'props' in node
    ? (node as { props?: { children?: ReactNode } }).props?.children
    : undefined
}

/** 把 markdown 表格的 children 拆成行 × 单元格，保留原始节点（加粗等仍生效）。 */
function tableRows(node: ReactNode): ReactNode[][] {
  const rows: ReactNode[][] = []
  function visit(child: ReactNode) {
    if (Array.isArray(child)) {
      child.forEach(visit)
      return
    }
    if (!child || typeof child !== 'object' || !('props' in child)) return
    const element = child as { type?: unknown; props?: { children?: ReactNode } }
    if (element.type === 'tr') {
      const cells: ReactNode[] = []
      const take = (node: ReactNode) => {
        if (Array.isArray(node)) {
          node.forEach(take)
          return
        }
        if (!node || typeof node !== 'object' || !('props' in node)) return
        const cell = node as { type?: unknown; props?: { children?: ReactNode } }
        if (cell.type === 'th' || cell.type === 'td') cells.push(cell.props?.children)
      }
      take(element.props?.children)
      if (cells.length) rows.push(cells)
      return
    }
    visit(element.props?.children)
  }
  visit(elementChildren(node))
  return rows
}

/** 表头第一列是这些词，就按"商品陈列"处理。 */
const PRODUCT_HEADER = ['机型', '商品', '型号', '产品']

function formatPrice(value: number): string {
  return Number(value || 0).toLocaleString('zh-CN')
}

export function ChatMessage({
  message, execution = [], streaming = false, products = [],
  selectedRefs = [], onDetail, onToggle,
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false)
  const [copyError, setCopyError] = useState(false)
  const images = products
    .filter((product) => product.image)
    .map((product) => ({ name: normalize(product.name), image: product.image as string, product }))

  function cellImage(children: ReactNode): string | undefined {
    const text = normalize(nodeText(children))
    if (!text) return undefined
    // 双向匹配：正文写全名、表格里常用简称（「雪脊 S5」↔「苍岭 雪脊 S5」）。
    return images.find((item) => item.name && (
      text.includes(item.name) || (text.length >= 4 && item.name.includes(text))
    ))?.image
  }

  /** 单元格文本对应的商品对象，用来复用大卡。 */
  function cellProduct(children: ReactNode): RecommendedProduct | undefined {
    const text = normalize(nodeText(children))
    if (!text) return undefined
    return images.find((item) => item.name && (
      text.includes(item.name) || (text.length >= 4 && item.name.includes(text))
    ))?.product
  }

  /**
   * 把卡片标记插到商品名首次出现的段落之后。
   *
   * 为什么在渲染前算：react-markdown 的渲染器没法定"插在哪个块后面"，
   * 而渲染期用 Set 记录"已插过"在 React 严格模式下会因双渲染丢卡。
   * 这里先在文本层插入 `::CARD::{ref}::` 占位段落，渲染器只负责替换。
   *
   * 顺序约定：**推荐卡在对比表之上**。模型常写成「结论 → 表格 → 结论」，
   * 商品名最后一次出现在结尾，按"最近提及"落点会把卡压到表格下面；
   * 因此搜索范围只到第一张表格为止，表格前找不到落点就直接插在表格前。
   * 只对"本段提到 ≤3 款"生效——多栏清单（十几款）不插大卡，否则墙。
   */
  const contentWithCards = useMemo(() => {
    const source = message.content || ''
    if (!source || !products.length) return source
    const blocks = source.split(/\n{2,}/)
    const firstTable = blocks.findIndex((block) => block.trimStart().startsWith('|'))
    const plain = normalize(source)
    // 只算本条回复里真正提到的商品：右侧列表是会话累积，不能拿总数做判断。
    const mentioned = products.filter((product) => {
      const name = normalize(product.name)
      return name ? plain.includes(name) : false
    })
    if (!mentioned.length) return source

    const placed = new Set<string>()
    /** 越靠近块首，越说明这块以它为主语（标题/列表项），优先级越高。 */
    const pickBlock = (name: string, limit: number): number => {
      let mid = -1
      let far = -1
      for (let index = 0; index < Math.min(limit, blocks.length); index += 1) {
        const block = blocks[index]
        if (block.trimStart().startsWith('|')) continue
        const at = normalize(block).indexOf(name)
        if (at < 0) continue
        if (at < 20) return index
        if (at < 80 && mid < 0) mid = index
        if (far < 0) far = index
      }
      return mid >= 0 ? mid : far
    }

    /**
     * 在块内找落点：命中列表项就紧跟这一条，否则挂在整块末尾。
     *
     * 模型常把一个品类写成「标题 + 多条 - 商品：卖点」的列表块，两条商品共处
     * 一个块；整块追加会让卡片全堆在列表下方。按行插入才是「一句介绍 + 一张卡」。
     * 标题行不是列表项，仍走整块末尾——首推卡保持"标题 → 要点 → 卡片"。
     */
    const LIST_ITEM = /^\s*(?:[-*+]|\d+\.)\s+/
    const insertCard = (block: string, name: string, ref: string): string => {
      const lines = block.split('\n')
      const at = lines.findIndex((line) => normalize(line).includes(name))
      if (at >= 0 && LIST_ITEM.test(lines[at])) {
        lines.splice(at + 1, 0, '', `::CARD::${ref}::`, '')
        return lines.join('\n')
      }
      return `${block}\n\n::CARD::${ref}::`
    }

    const lead: string[] = []
    for (const product of mentioned) {
      const name = normalize(product.name)
      const index = pickBlock(name, firstTable >= 0 ? firstTable : blocks.length)
      if (index >= 0) {
        blocks[index] = insertCard(blocks[index], name, product.ref)
        placed.add(product.ref)
        continue
      }
      // 表格前没有落点（「结论 → 表格 → 结论」这种结构）：压到表格前面。
      if (firstTable >= 0 && mentioned.length <= 3) {
        lead.push(`::CARD::${product.ref}::`)
        placed.add(product.ref)
      }
    }
    if (lead.length) blocks.splice(firstTable, 0, ...lead)
    return placed.size ? blocks.join('\n\n') : source
  }, [message.content, products])

  async function copy() {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setCopyError(false)
    } catch {
      setCopyError(true)
    }
  }

  /** 单个商品：图片 + 名称 + 价格 + 简短介绍的大卡。 */
  function productCard(product: RecommendedProduct) {
    const active = selectedRefs.includes(product.ref)
    return (
      <article className={`pd-card ${product.best ? 'is-best' : ''}`} key={product.ref}>
        <div className="pd-media">
          {product.image ? <img src={product.image} alt="" loading="lazy" /> : <span>#{String(product.display_no).padStart(2, '0')}</span>}
        </div>
        <div className="pd-body">
          <h4 className="pd-name">{product.best && <span className="pd-tag">首选</span>}{product.name}</h4>
          {(product.summary || product.reason) && <p className="pd-desc">{product.summary || product.reason}</p>}
          <div className="pd-price"><b><i>¥</i>{formatPrice(product.price)}</b></div>
          {(onDetail || onToggle) && (
            <div className="pd-actions">
              {onDetail && <button className="pd-btn" type="button" onClick={() => onDetail(product)}>查看商品</button>}
              {onToggle && (
                <button className={`pd-btn ghost ${active ? 'is-active' : ''}`} type="button" aria-pressed={active} onClick={() => onToggle(product.ref)}>
                  {active ? '已加入' : '加入对比'}
                </button>
              )}
            </div>
          )}
        </div>
      </article>
    )
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
                  table: ({ children, ...props }) => {
                    const rows = tableRows(children)
                    const header = (rows[0] || []).map(nodeText)
                    const body = rows.slice(1)
                    // 表格里"机型/商品/型号"这种列每出现一次，就代表一组商品字段。
                    const nameCols = header.reduce<number[]>((acc, label, index) => {
                      if (PRODUCT_HEADER.some((word) => label.includes(word))) acc.push(index)
                      return acc
                    }, [])
                    const plain = <div className="table-scroll" tabIndex={0} role="region" aria-label="数据表"><table {...props}>{children}</table></div>
                    if (rows.length < 2 || (!nameCols.length && !header[0]?.includes('维度'))) return plain

                    // 已经是对比表（列 = 商品）：只补表头图片，结构不动。
                    if (!nameCols.length) {
                      return (
                        <div className="pd-compare">
                          <table>
                            <thead>
                              <tr>{header.map((label, index) => (
                                <th key={index}>{index === 0 ? label : (
                                  <span className="pd-ch">
                                    <span className="pd-ch-media">{cellImage(label) && <img src={cellImage(label)} alt="" loading="lazy" />}</span>
                                    <span className="pd-ch-body"><b>{label}</b></span>
                                  </span>
                                )}</th>
                              ))}</tr>
                            </thead>
                            <tbody>
                              {body.map((cells, rowIndex) => (
                                <tr key={rowIndex}>{cells.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}</tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )
                    }

                    // 只有一款的表格等价于"单品推荐"，直接升格成商品大卡。
                    if (nameCols.length === 1 && body.length === 1) {
                      const single = cellProduct(body[0][0])
                      if (single) return productCard(single)
                    }

                    // 多栏清单（左半 / 右半各列一组）：摊平成商品小卡网格。
                    if (nameCols.length > 1) {
                      const blocks = nameCols.map((start, index) => ({
                        start,
                        end: (nameCols[index + 1] ?? header.length) - 1,
                      }))
                      const tiles = body.flatMap((cells, rowIndex) => blocks.flatMap((block, blockIndex) => {
                        const name = cells[block.start]
                        if (!nodeText(name).trim()) return []
                        const priceAt = header.findIndex((label, i) => i > block.start && i <= block.end && label.includes('价'))
                        const descAt = header.findIndex((label, i) => i > block.start && i <= block.end && !label.includes('价'))
                        return [{
                          key: `${rowIndex}-${blockIndex}`,
                          name,
                          price: priceAt > 0 ? cells[priceAt] : null,
                          desc: descAt > 0 ? cells[descAt] : null,
                          image: cellImage(name),
                        }]
                      }))
                      return (
                        <div className="pd-grid" role="list">
                          {tiles.map((tile) => (
                            <article className="pd-grid-card" role="listitem" key={tile.key}>
                              <div className="pd-media">{tile.image && <img src={tile.image} alt="" loading="lazy" />}</div>
                              <b>{tile.name}</b>
                              {tile.price && <span className="pd-grid-price">{tile.price}</span>}
                              {tile.desc && <small>{tile.desc}</small>}
                            </article>
                          ))}
                        </div>
                      )
                    }

                    // 行 = 商品的单表：转置成"列 = 商品"的对比表。
                    const priceIndex = header.findIndex((label) => label.includes('价'))
                    return (
                      <div className="pd-compare">
                        <table>
                          <thead>
                            <tr>
                              <th aria-hidden="true" />
                              {body.map((cells, index) => (
                                <th key={index}>
                                  <span className="pd-ch">
                                    <span className="pd-ch-media">{cellImage(cells[0]) && <img src={cellImage(cells[0])} alt="" loading="lazy" />}</span>
                                    <span className="pd-ch-body">
                                      <b>{cells[0]}</b>
                                      {priceIndex > 0 && <span className="pd-ch-price">{cells[priceIndex]}</span>}
                                    </span>
                                  </span>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {header.slice(1).map((label, rowIndex) => (
                              <tr key={label}>
                                <td>{label}</td>
                                {body.map((cells, index) => <td key={index}>{cells[rowIndex + 1] ?? ''}</td>)}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )
                  },
                  a: ({ children, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer">{children}</a>,
                  // 占位段落 → 商品大卡；其余段落原样渲染。
                  p: ({ children, ...props }) => {
                    const marker = /^::CARD::(.+)::$/.exec(nodeText(children).trim())
                    const product = marker
                      ? products.find((item) => item.ref === marker[1])
                      : undefined
                    return product ? productCard(product) : <p {...props}>{children}</p>
                  },
                }}>{contentWithCards}</ReactMarkdown>
              ) : <div className="answer-loading" role="status"><span /><span /><span /><span className="sr-only">正在处理</span></div>}
            </div>
            {message.content && !streaming && <div className="message-actions"><button className="icon-button" type="button" onClick={() => void copy()} aria-label={copied ? '已复制回答' : '复制回答'} title={copied ? '已复制' : '复制回答'}>{copied ? <Check size={16} /> : <Copy size={16} />}</button>{copyError && <span role="status">无法访问剪贴板</span>}</div>}
          </>
        ) : <p>{message.content}</p>}
      </div>
    </article>
  )
}

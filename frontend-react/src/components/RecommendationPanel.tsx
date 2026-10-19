import { ArrowsClockwise, ArrowSquareOut, CaretRight } from '@phosphor-icons/react'
import type { RecommendedProduct } from '../types/product'

interface RecommendationPanelProps {
  products: RecommendedProduct[]
  selectedRefs: string[]
  onToggle: (ref: string) => void
  onDetail: (product: RecommendedProduct) => void
  onCompare: () => void
}

export function RecommendationPanel({ products, selectedRefs, onToggle, onDetail, onCompare }: RecommendationPanelProps) {
  return (
    <aside className="catalog-panel" aria-label="本轮推荐商品">
      <div className="catalog-heading"><div><h2>本轮推荐</h2><p className="panel-copy">只显示聊天中实际提到的商品</p></div><span className="recommendation-count">{products.length || '0'} 款</span></div>
      <div className="recommendation-list">
        {!products.length && <div className="catalog-empty"><ArrowsClockwise size={28} weight="light" /><p>回复中出现商品后，会在这里按时间保留</p></div>}
        {products.map((product) => {
          const selected = selectedRefs.includes(product.ref)
          return <article className={`recommendation-card ${selected ? 'is-selected' : ''}`} key={product.ref}>
            <div className="recommendation-card-head"><span className="recommendation-number">#{String(product.display_no).padStart(2, '0')}</span><span className="recommendation-category">{product.category}</span></div>
            <button className="recommendation-name" type="button" onClick={() => onDetail(product)}>{product.name}<ArrowSquareOut size={14} aria-hidden="true" /></button>
            <div className="recommendation-meta"><strong>¥{Number(product.price || 0).toLocaleString('zh-CN')}</strong><span>{product.reason}</span></div>
            <div className="recommendation-actions"><button type="button" className={`recommendation-compare ${selected ? 'is-selected' : ''}`} onClick={() => onToggle(product.ref)}>{selected ? '已加入对比' : '加入对比'}</button><button type="button" className="recommendation-detail" onClick={() => onDetail(product)} aria-label={`查看${product.name}详情`} title="查看详情"><CaretRight size={18} /></button></div>
          </article>
        })}
      </div>
      <div className="compare-dock"><div><span>对比中</span><strong>{selectedRefs.length ? selectedRefs.map((ref) => `#${String(products.find((item) => item.ref === ref)?.display_no || '')}`).join('、') : '请选择商品'}</strong></div><button type="button" disabled={selectedRefs.length < 2} onClick={onCompare}>一键对比</button></div>
    </aside>
  )
}

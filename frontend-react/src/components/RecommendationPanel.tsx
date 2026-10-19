import { ArrowUpRight, ArrowsClockwise } from '@phosphor-icons/react'
import type { RecommendedProduct } from '../types/product'

/** 对比上限：与设计稿的 3 个槽位一致。 */
const MAX_COMPARE = 3

interface RecommendationPanelProps {
  products: RecommendedProduct[]
  selectedRefs: string[]
  onToggle: (ref: string) => void
  onDetail: (product: RecommendedProduct) => void
  onCompare: () => void
}

export function RecommendationPanel({ products, selectedRefs, onToggle, onDetail, onCompare }: RecommendationPanelProps) {
  const selected = selectedRefs
    .map((ref) => products.find((product) => product.ref === ref))
    .filter((product): product is RecommendedProduct => Boolean(product))
  const full = selected.length >= MAX_COMPARE

  return (
    <aside className="catalog-panel" aria-label="本轮推荐商品">
      <header className="catalog-heading">
        <div><h2>本轮推荐</h2><p className="panel-copy">只显示聊天中实际提到的商品</p></div>
        <span className="recommendation-count">{products.length} 款</span>
      </header>

      <div className="recommendation-list">
        {!products.length && <div className="catalog-empty"><ArrowsClockwise size={28} weight="light" /><p>回复中出现商品后，会在这里按时间保留</p></div>}
        {products.map((product) => {
          const active = selectedRefs.includes(product.ref)
          return (
            <article className={`recommendation-card ${product.best ? 'is-best' : ''}`} key={product.ref}>
              <div className="recommendation-thumb">
                {product.image
                  ? <img src={product.image} alt="" loading="lazy" />
                  : <span>#{String(product.display_no).padStart(2, '0')}</span>}
              </div>
              <div className="recommendation-body">
                <div className="recommendation-title">
                  <b>{product.name}</b>
                  {product.best && <span className="recommendation-best">首选</span>}
                </div>
                <small className="recommendation-summary">{product.summary || product.reason}</small>
                <div className="recommendation-price-row">
                  <span className="recommendation-price"><i>¥</i>{Number(product.price || 0).toLocaleString('zh-CN')}</span>
                  <div className="recommendation-actions">
                    <button className="recommendation-detail" type="button" onClick={() => onDetail(product)} title="查看详情" aria-label={`查看${product.name}详情`}>
                      <ArrowUpRight size={16} />
                    </button>
                    <button className="recommendation-compare" type="button" aria-pressed={active}
                      disabled={!active && full} onClick={() => onToggle(product.ref)}>
                      {active ? '已加入' : '加入对比'}
                    </button>
                  </div>
                </div>
              </div>
            </article>
          )
        })}
      </div>

      <footer className="compare-dock">
        <div className="compare-status">
          <span>对比中 <b>{selected.length}</b>/{MAX_COMPARE}</span>
          <div className="compare-slots">
            {Array.from({ length: MAX_COMPARE }, (_, index) => {
              const item = selected[index]
              return (
                <span className={item ? 'is-filled' : ''} key={index} title={item?.name}>
                  {item?.image
                    ? <img src={item.image} alt="" />
                    : item ? `#${String(item.display_no).padStart(2, '0')}` : null}
                </span>
              )
            })}
          </div>
        </div>
        <button className="compare-go" type="button" disabled={selected.length < 2} onClick={onCompare}>一键对比</button>
      </footer>
    </aside>
  )
}

import { ArrowUpRight, Heart, HeartStraight, X } from '@phosphor-icons/react'
import type { FavoriteProduct } from '../api/favorites'

interface FavoritesDrawerProps {
  open: boolean
  items: FavoriteProduct[]
  onClose: () => void
  onRemove: (product: FavoriteProduct) => void
  onDetail: (product: FavoriteProduct) => void
}

export function FavoritesDrawer({ open, items, onClose, onRemove, onDetail }: FavoritesDrawerProps) {
  if (!open) return null
  return <div className="detail-drawer" role="dialog" aria-modal="true" aria-label="我的收藏">
    <div className="drawer-backdrop" onClick={onClose} />
    <section className="drawer-content">
      <header><div><span>我的收藏</span><h2>{items.length ? `${items.length} 款商品` : '暂无收藏'}</h2></div><button className="icon-button" type="button" onClick={onClose} title="关闭收藏" aria-label="关闭收藏"><X size={20} /></button></header>
      <div className="favorites-list">
        {!items.length && <div className="favorites-empty"><HeartStraight size={30} weight="light" /><b>这里还空着</b><span>点击商品卡片上的心形，收藏后会集中显示在这里。</span></div>}
        {items.map((product) => (
          <article className="favorite-card" key={product.product_id}>
            <div className="favorite-body">
              <b>{product.name}</b>
              <small>{product.brand || product.category} · {product.category}</small>
              {product.summary && <p>{product.summary}</p>}
              <span className="recommendation-price"><i>¥</i>{Number(product.price || 0).toLocaleString('zh-CN')}</span>
            </div>
            <div className="favorite-actions">
              <button className="recommendation-detail" type="button" title="查看详情" aria-label={`查看${product.name}详情`} onClick={() => onDetail(product)}>
                <ArrowUpRight size={16} />
              </button>
              <button className="recommendation-favorite is-active" type="button" title="取消收藏" aria-label={`取消收藏${product.name}`} onClick={() => onRemove(product)}>
                <Heart size={16} weight="fill" />
              </button>
            </div>
          </article>
        ))}
      </div>
    </section>
  </div>
}

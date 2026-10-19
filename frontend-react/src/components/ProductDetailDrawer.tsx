import { X } from '@phosphor-icons/react'
import type { ProductDetail, RecommendedProduct } from '../types/product'

interface ProductDetailDrawerProps {
  product: RecommendedProduct | null
  detail: ProductDetail | null
  loading: boolean
  onClose: () => void
  onCompare: () => void
}

export function ProductDetailDrawer({ product, detail, loading, onClose, onCompare }: ProductDetailDrawerProps) {
  if (!product) return null
  const current = detail || product
  const specs = detail?.specs ? Object.entries(detail.specs).slice(0, 8) : []
  return <div className="detail-drawer" role="dialog" aria-modal="true" aria-label={`${product.name}商品详情`}>
    <div className="drawer-backdrop" onClick={onClose} />
    <section className="drawer-content">
      <header><div><span>商品详情</span><h2>{current.name || product.name}</h2></div><button className="icon-button" type="button" onClick={onClose} title="关闭详情" aria-label="关闭详情"><X size={20} /></button></header>
      {(current.image || product.image) && <img className="drawer-image" src={current.image || product.image || ''} alt="" />}
      {loading ? <div className="drawer-loading">正在读取当前商品信息…</div> : <>
        <div className="drawer-price">¥{Number(current.price || product.price || 0).toLocaleString('zh-CN')}<span>{current.brand || product.brand} · {current.category || product.category}</span></div>
        <p className="drawer-summary">{detail?.introduction?.[0] || product.summary || product.reason}</p>
        {specs.length > 0 && <dl className="drawer-specs">{specs.map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>)}</dl>}
        <button className="drawer-compare" type="button" onClick={onCompare}>加入对比</button>
      </>}
    </section>
  </div>
}

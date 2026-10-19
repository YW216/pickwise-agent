import type { PublicProduct } from '../types/product'

export function ProductCard({ product }: { product: PublicProduct }) {
  return (
    <article className="product-card">
      <strong>{product.name}</strong>
      <span className="product-price">¥{Number(product.price || 0).toLocaleString('zh-CN')}</span>
      <small>{product.summary || product.category}</small>
      <span className="product-status">{product.category || '商品'} · 当前目录</span>
    </article>
  )
}

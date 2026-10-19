import type { ProductDetail, ProductSearchResult } from '../types/product'
import { requestJson } from './client'

export function searchProducts(query: string, limit = 6): Promise<ProductSearchResult> {
  const params = new URLSearchParams({ mode: 'semantic', query, limit: String(limit) })
  return requestJson(`/api/products/search?${params.toString()}`)
}

export function loadProductDetail(sessionId: string, ref: string): Promise<{ product: ProductDetail }> {
  const params = new URLSearchParams({ session_id: sessionId })
  return requestJson(`/api/products/ref/${encodeURIComponent(ref)}?${params.toString()}`)
}

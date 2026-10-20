import type { ProductDetail } from '../types/product'
import { requestJson } from './client'

export function loadProductDetail(sessionId: string, ref: string): Promise<{ product: ProductDetail }> {
  const params = new URLSearchParams({ session_id: sessionId })
  return requestJson(`/api/products/ref/${encodeURIComponent(ref)}?${params.toString()}`)
}

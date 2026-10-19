import type { ProductSearchResult } from '../types/product'
import { requestJson } from './client'

export function searchProducts(query: string, limit = 6): Promise<ProductSearchResult> {
  const params = new URLSearchParams({ mode: 'semantic', query, limit: String(limit) })
  return requestJson(`/api/products/search?${params.toString()}`)
}

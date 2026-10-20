import { requestJson } from './client'

export interface FavoriteProduct {
  product_id: string
  name: string
  brand: string
  category: string
  price: number
  summary: string
}

export function listFavorites(limit = 50): Promise<{ favorites: FavoriteProduct[]; total: number }> {
  const params = new URLSearchParams({ limit: String(limit) })
  return requestJson(`/api/favorites?${params.toString()}`)
}

export function setFavorite(sessionId: string, ref: string, favorited: boolean): Promise<{ favorited: boolean }> {
  const path = `/api/favorites/ref/${encodeURIComponent(ref)}`
  const body = JSON.stringify({ session_id: sessionId })
  return requestJson(path, { method: favorited ? 'POST' : 'DELETE', body })
}

export function removeFavoriteById(productId: string): Promise<{ favorited: boolean }> {
  return requestJson(`/api/favorites/item/${encodeURIComponent(productId)}`, { method: 'DELETE' })
}

export function loadFavoriteDetail(productId: string): Promise<{ product: import('../types/product').ProductDetail }> {
  return requestJson(`/api/favorites/item/${encodeURIComponent(productId)}/detail`)
}

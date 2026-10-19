export interface PublicProduct {
  candidate_no?: number
  name: string
  brand: string
  category: string
  price: number
  summary: string
}

export interface ProductSearchResult {
  results?: PublicProduct[]
  candidates?: PublicProduct[]
  nearest?: PublicProduct[]
  total?: number
  query?: string
}

export interface RecommendedProduct extends PublicProduct {
  ref: string
  display_no: number
  reason: string
}

export interface ProductDetail {
  name?: string
  brand?: string
  category?: string
  price?: number
  specs?: Record<string, unknown>
  introduction?: string[]
}

export interface RecommendedProduct extends PublicProduct {
  ref: string
  display_no: number
  reason: string
}

export interface ProductDetail {
  name?: string
  brand?: string
  category?: string
  price?: number
  specs?: Record<string, unknown>
  introduction?: string[]
}

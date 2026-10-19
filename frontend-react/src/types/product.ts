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

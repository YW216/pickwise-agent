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
  /** 后端下发的商品示意图 URL；缺省时卡片回退纯文字。 */
  image?: string | null
  /** 本轮主推：卡片上显示「首选」标签，全局只有一个。 */
  best?: boolean
}

export interface ProductDetail {
  name?: string
  brand?: string
  category?: string
  price?: number
  specs?: Record<string, unknown>
  introduction?: string[]
  image?: string | null
}

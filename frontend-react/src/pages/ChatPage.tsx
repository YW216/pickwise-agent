import { useEffect, useReducer, useRef, useState } from 'react'
import { Heart, List, MagnifyingGlass, X } from '@phosphor-icons/react'
import { streamChat } from '../api/chat'
import { loadFavoriteDetail, listFavorites, removeFavoriteById } from '../api/favorites'
import type { FavoriteProduct } from '../api/favorites'
import { loadProductDetail } from '../api/products'
import { createSession, deleteSession, initializeIdentity, listSessions, loadMessages } from '../api/sessions'
import { ChatComposer } from '../components/ChatComposer'
import { ChatMessage } from '../components/ChatMessage'
import { FavoritesDrawer } from '../components/FavoritesDrawer'
import { ProductDetailDrawer } from '../components/ProductDetailDrawer'
import { RecommendationPanel } from '../components/RecommendationPanel'
import { SessionList } from '../components/SessionList'
import { chatReducer, initialChatState } from '../store/chatStore'
import type { ProductDetail, RecommendedProduct } from '../types/product'

const sessionStorageKey = 'pickwise.sessionId'
const palettes = [{ key: 'a', name: '钛金' }, { key: 'b', name: '群青' }, { key: 'c', name: '岩紫' }] as const

export function ChatPage() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState)
  const [detailProduct, setDetailProduct] = useState<RecommendedProduct | null>(null)
  const [detail, setDetail] = useState<ProductDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailKind, setDetailKind] = useState<'recommendation' | 'favorite'>('recommendation')
  const [favoriteItems, setFavoriteItems] = useState<FavoriteProduct[]>([])
  const [favoritesOpen, setFavoritesOpen] = useState(false)
  const [composerValue, setComposerValue] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [catalogOpen, setCatalogOpen] = useState(false)
  const [palette, setPalette] = useState(() => {
    const saved = localStorage.getItem('pickwise.palette')
    return palettes.some((item) => item.key === saved) ? saved : 'a'
  })
  const controllerRef = useRef<AbortController | null>(null)
  const bootedRef = useRef(false)
  const selectedSessionRef = useRef<string | null>(null)
  const feedRef = useRef<HTMLDivElement>(null)

  async function refreshFavorites() {
    try {
      const result = await listFavorites()
      setFavoriteItems(result.favorites)
    } catch {
      setFavoriteItems([])
    }
  }

  async function refreshSessions() {
    const result = await listSessions()
    dispatch({ type: 'SESSIONS_LOADED', sessions: result.sessions })
    return result.sessions
  }

  async function selectSession(sessionId: string) {
    selectedSessionRef.current = sessionId
    dispatch({ type: 'SESSION_SELECTED', sessionId })
    localStorage.setItem(sessionStorageKey, sessionId)
    try {
      const result = await loadMessages(sessionId)
      if (selectedSessionRef.current === sessionId) {
        dispatch({ type: 'MESSAGES_LOADED', messages: result.messages, recommendations: result.recommendations })
      }
    } catch (error) {
      if (selectedSessionRef.current === sessionId) {
        dispatch({ type: 'ERROR', message: error instanceof Error ? error.message : '加载会话失败' })
      }
    }
  }

  useEffect(() => {
    document.documentElement.dataset.p = palette || 'a'
    localStorage.setItem('pickwise.palette', palette || 'a')
  }, [palette])

  useEffect(() => {
    if (bootedRef.current) return
    bootedRef.current = true
    async function boot() {
      try {
        await initializeIdentity()
        dispatch({ type: 'INITIALIZED' })
        void refreshFavorites()
        const sessions = await refreshSessions()
        const preferred = localStorage.getItem(sessionStorageKey)
        const selected = sessions.find((item) => item.session_id === preferred) || sessions[0]
        if (selected) {
          await selectSession(selected.session_id)
        } else {
          const created = await createSession()
          await refreshSessions()
          await selectSession(created.session_id)
        }
      } catch (error) {
        dispatch({ type: 'ERROR', message: error instanceof Error ? error.message : '页面初始化失败' })
      }
    }
    void boot()
  }, [])

  useEffect(() => () => controllerRef.current?.abort(), [])

  useEffect(() => {
    const feed = feedRef.current
    if (feed && state.streaming && feed.scrollHeight - feed.clientHeight - feed.scrollTop < 180) {
      feed.scrollTop = feed.scrollHeight
    }
  }, [state.messages, state.streaming])

  useEffect(() => {
    if (!sidebarOpen && !catalogOpen && !favoritesOpen) return
    function close(event: KeyboardEvent) {
      if (event.key === 'Escape') { setSidebarOpen(false); setCatalogOpen(false); setFavoritesOpen(false) }
    }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [sidebarOpen, catalogOpen, favoritesOpen])

  async function handleCreateSession() {
    if (state.streaming || controllerRef.current) return
    try {
      const created = await createSession()
      await refreshSessions()
      await selectSession(created.session_id)
      setSidebarOpen(false)
      setComposerValue('')
    } catch (error) {
      dispatch({ type: 'ERROR', message: error instanceof Error ? error.message : '创建会话失败' })
    }
  }

  async function handleDeleteSession(sessionId: string) {
    if (state.streaming || !window.confirm('确定删除这个会话吗？')) return
    try {
      await deleteSession(sessionId)
      const sessions = await refreshSessions()
      if (selectedSessionRef.current !== sessionId) return
      if (sessions[0]) await selectSession(sessions[0].session_id)
      else {
        const created = await createSession()
        await refreshSessions()
        await selectSession(created.session_id)
      }
    } catch (error) {
      dispatch({ type: 'ERROR', message: error instanceof Error ? error.message : '删除会话失败' })
    }
  }

  async function handleSend(message: string, productRefs: string[] = []) {
    const sessionId = state.currentSessionId
    if (!sessionId || state.loading || controllerRef.current) return
    const controller = new AbortController()
    controllerRef.current = controller
    dispatch({ type: 'STREAM_STARTED', message })
    try {
      await streamChat({ session_id: sessionId, message, product_refs: productRefs }, crypto.randomUUID(), controller.signal,
        (event) => dispatch({ type: 'STREAM_EVENT', event }))
      await refreshSessions().catch(() => undefined)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        dispatch({ type: 'STREAM_STOPPED', message: '已停止接收；后台仍会继续处理并保存' })
      } else {
        dispatch({ type: 'ERROR', message: error instanceof Error ? error.message : '聊天请求失败' })
      }
    } finally {
      controllerRef.current = null
    }
  }

  async function openDetails(product: RecommendedProduct) {
    if (!state.currentSessionId) return
    setDetailKind('recommendation')
    setDetailProduct(product)
    setDetail(null)
    setDetailLoading(true)
    try {
      const result = await loadProductDetail(state.currentSessionId, product.ref)
      setDetail(result.product)
    } catch (error) {
      dispatch({ type: 'ERROR', message: error instanceof Error ? error.message : '商品详情加载失败' })
    } finally {
      setDetailLoading(false)
    }
  }

  async function openFavoriteDetails(favorite: FavoriteProduct) {
    setFavoritesOpen(false)
    setDetailKind('favorite')
    setDetailProduct({
      ref: favorite.product_id,
      display_no: 0,
      name: favorite.name,
      brand: favorite.brand,
      category: favorite.category,
      price: favorite.price,
      summary: favorite.summary,
      reason: favorite.summary,
    })
    setDetail(null)
    setDetailLoading(true)
    try {
      const result = await loadFavoriteDetail(favorite.product_id)
      setDetail(result.product)
    } catch (error) {
      dispatch({ type: 'ERROR', message: error instanceof Error ? error.message : '收藏商品详情加载失败' })
    } finally {
      setDetailLoading(false)
    }
  }

  async function removeFavorite(favorite: FavoriteProduct) {
    try {
      await removeFavoriteById(favorite.product_id)
      await refreshFavorites()
    } catch (error) {
      dispatch({ type: 'ERROR', message: error instanceof Error ? error.message : '取消收藏失败' })
    }
  }

  function toggleCompare(ref: string) {
    dispatch({ type: 'TOGGLE_COMPARE', ref })
  }

  function compareSelected() {
    const selected = state.recommendations.filter((product) => state.selectedRefs.includes(product.ref))
    if (selected.length < 2) return
    const message = `请对比本轮选中的商品：${selected.map((product) => `#${String(product.display_no).padStart(2, '0')} ${product.name}`).join('、')}。重点比较价格、关键参数和适用场景。`
    setCatalogOpen(false)
    // 引用随请求一起上送：服务端据此还原 product_id，不依赖回复文本猜商品。
    void handleSend(message, state.selectedRefs)
  }

  const lastMessageIndex = state.messages.length - 1
  const currentSession = state.sessions.find((session) => session.session_id === state.currentSessionId)
  const streamingMessage = [...state.execution].reverse().find((phase) => phase.state === 'running')?.message

  return (
    <main className={`app ${sidebarOpen ? 'sidebar-open' : ''} ${catalogOpen ? 'catalog-open' : ''}`} aria-label="PickWise 工作区">
      <SessionList sessions={state.sessions} currentSessionId={state.currentSessionId}
        disabled={!state.initialized || state.streaming}
        onCreate={() => void handleCreateSession()}
        onSelect={(id) => { setSidebarOpen(false); void selectSession(id) }}
        onDelete={(id) => void handleDeleteSession(id)} onClose={() => setSidebarOpen(false)} />

      <section className="main" aria-label="聊天">
        <header className="top">
          <button className="icon-button mobile-menu-button" type="button" onClick={() => setSidebarOpen(true)} aria-label="打开会话列表" aria-expanded={sidebarOpen} title="会话列表"><List size={20} /></button>
          <div className="top-title"><h1>{currentSession?.title || '选购助手'}</h1><p>把复杂需求，整理成可执行的选择。</p></div>
          <div className="topbar-actions">
            <div className="palette-switch" role="group" aria-label="配色方案">
              {palettes.map((item) => <button type="button" key={item.key} className={palette === item.key ? 'is-active' : ''} data-palette={item.key}
                onClick={() => setPalette(item.key)} aria-pressed={palette === item.key} title={item.name} aria-label={item.name} />)}
            </div>
            <span className={`live ${!state.initialized || state.error ? 'is-unavailable' : ''}`} role="status">{state.streaming ? '正在分析' : state.error ? '需要处理' : state.initialized ? '会话就绪' : '正在连接'}</span>
            <button className="icon-button catalog-toggle" type="button" onClick={() => setCatalogOpen(true)} aria-label="打开本轮推荐" aria-expanded={catalogOpen} title="本轮推荐"><MagnifyingGlass size={20} /></button>
            <button className="icon-button favorite-toggle" type="button" onClick={() => setFavoritesOpen(true)} aria-label={`我的收藏，${favoriteItems.length}款商品`} aria-expanded={favoritesOpen} title="我的收藏"><Heart size={20} weight={favoriteItems.length ? 'fill' : 'regular'} /><span>{favoriteItems.length}</span></button>
          </div>
        </header>

        <div className="messages" aria-live="polite" aria-busy={state.loading} ref={feedRef}>
          {state.loading && <div className="loading-state" role="status" aria-label="正在加载会话"><span className="skeleton-line skeleton-line-short" /><span className="skeleton-line" /><span className="skeleton-line skeleton-line-medium" /></div>}
          {!state.loading && state.messages.length === 0 && <div className="empty-chat"><span className="empty-mark" aria-hidden="true">PW</span><h2>这次，想选什么？</h2><p>预算、用途，以及你最在意的那一点。</p><div className="quick-questions">{['6000 元以内的轻薄本', '通勤降噪耳机', '适合打游戏的电脑'].map((text) => <button type="button" key={text} className="query-chip" disabled={!state.currentSessionId} onClick={() => { setComposerValue(text); document.getElementById('message-input')?.focus() }}>{text}</button>)}</div></div>}
          {state.messages.map((message, index) => <ChatMessage key={`${message.message_id || 'local'}-${index}`} message={message}
            products={state.recommendations}
            selectedRefs={state.selectedRefs}
            onDetail={(product) => void openDetails(product)}
            onToggle={toggleCompare}
            execution={index === lastMessageIndex && message.role === 'assistant' ? state.execution : []} streaming={state.streaming && index === lastMessageIndex} />)}
        </div>
        {state.error && <div className="inline-error" role="alert"><span>{state.error}</span><button className="icon-button" type="button" title="关闭提示" aria-label="关闭提示" onClick={() => dispatch({ type: 'CLEAR_ERROR' })}><X size={17} /></button></div>}
        {state.streaming && streamingMessage && <p className="stream-status" role="status">{streamingMessage}</p>}
        <ChatComposer disabled={state.loading || !state.currentSessionId} streaming={state.streaming} value={composerValue} onChange={setComposerValue} onSubmit={(message) => void handleSend(message)} onStop={() => controllerRef.current?.abort()} />
      </section>

      <RecommendationPanel products={state.recommendations} selectedRefs={state.selectedRefs} onToggle={toggleCompare} onDetail={openDetails} onCompare={compareSelected} />
      <FavoritesDrawer open={favoritesOpen} items={favoriteItems} onClose={() => setFavoritesOpen(false)} onRemove={(product) => void removeFavorite(product)} onDetail={(product) => void openFavoriteDetails(product)} />
      {(sidebarOpen || catalogOpen) && <button className="panel-backdrop" type="button" aria-label="关闭侧栏" tabIndex={-1} onClick={() => { setSidebarOpen(false); setCatalogOpen(false) }} />}
      <ProductDetailDrawer product={detailProduct} detail={detail} loading={detailLoading} onClose={() => setDetailProduct(null)} onCompare={detailKind === 'recommendation' ? () => { if (detailProduct) toggleCompare(detailProduct.ref); setDetailProduct(null) } : undefined} />
    </main>
  )
}

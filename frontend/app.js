const state = {
  csrfToken: "",
  sessionId: "",
  sessions: [],
  busy: false,
};

const elements = {
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chat-form"),
  messageInput: document.querySelector("#message-input"),
  sendButton: document.querySelector("#send-button"),
  chatError: document.querySelector("#chat-error"),
  sessionStatus: document.querySelector("#session-status"),
  newSession: document.querySelector("#new-session"),
  sessionList: document.querySelector("#session-list"),
  productForm: document.querySelector("#product-form"),
  productQuery: document.querySelector("#product-query"),
  productResults: document.querySelector("#product-results"),
};

async function requestJson(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  const response = await fetch(path, {
    ...options,
    headers,
    credentials: "same-origin",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.error?.message || payload?.detail || `请求失败（${response.status}）`;
    const error = new Error(message);
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function setStatus(text, tone = "") {
  elements.sessionStatus.textContent = text;
  elements.sessionStatus.className = `status-pill ${tone}`.trim();
}

function showChatError(message = "") {
  elements.chatError.textContent = message;
  elements.chatError.hidden = !message;
}

function appendMessage(role, text, meta = "") {
  const wrapper = document.createElement("article");
  wrapper.className = `message message-${role}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "你" : "PickWise";
  const body = document.createElement("p");
  body.textContent = text;
  wrapper.append(label, body);
  if (meta) {
    const note = document.createElement("span");
    note.className = "message-meta";
    note.textContent = meta;
    wrapper.append(note);
  }
  elements.messages.append(wrapper);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return wrapper;
}

function renderHistory(messages) {
  elements.messages.replaceChildren();
  if (!messages.length) {
    appendMessage("assistant", "你好，我是 PickWise。告诉我预算、使用场景和在意的参数，我会帮你缩小选择范围。", "空会话");
    return;
  }
  for (const message of messages) {
    appendMessage(message.role, message.content);
  }
}

function renderSessionList() {
  elements.sessionList.replaceChildren();
  if (!state.sessions.length) {
    const empty = document.createElement("div");
    empty.className = "session-empty";
    empty.textContent = "还没有会话";
    elements.sessionList.append(empty);
    return;
  }
  for (const session of state.sessions) {
    const item = document.createElement("article");
    item.className = `session-item${session.session_id === state.sessionId ? " is-active" : ""}`;
    item.dataset.sessionId = session.session_id;
    item.innerHTML = `
      <button class="session-select" type="button" data-action="select" data-session-id="${escapeHtml(session.session_id)}">
        <strong>${escapeHtml(session.title || "新会话")}</strong>
        <span>${session.message_count || 0} 条消息</span>
      </button>
      <button class="session-delete" type="button" title="删除会话" aria-label="删除会话" data-action="delete" data-session-id="${escapeHtml(session.session_id)}">×</button>
    `;
    elements.sessionList.append(item);
  }
}

function setBusy(busy) {
  state.busy = busy;
  elements.sendButton.disabled = busy;
  elements.messageInput.disabled = busy;
  elements.newSession.disabled = busy;
  elements.sendButton.innerHTML = busy ? "处理中…" : '发送 <span aria-hidden="true">↗</span>';
}

async function initializeIdentity() {
  const payload = await requestJson("/api/auth/anonymous", { method: "POST" });
  state.csrfToken = payload.data.csrf_token;
}

async function refreshSessions() {
  const payload = await requestJson("/api/sessions");
  state.sessions = payload.data.sessions || [];
  renderSessionList();
  return state.sessions;
}

async function createSession() {
  const payload = await requestJson("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Pickwise-CSRF": state.csrfToken },
    body: JSON.stringify({ title: "新会话" }),
  });
  state.sessionId = payload.data.session_id;
  localStorage.setItem("pickwise.sessionId", state.sessionId);
  await refreshSessions();
  renderHistory([]);
  setStatus("会话已连接", "status-ready");
}

async function loadSession(sessionId) {
  const payload = await requestJson(`/api/sessions/${encodeURIComponent(sessionId)}/messages`);
  state.sessionId = payload.data.session_id;
  localStorage.setItem("pickwise.sessionId", state.sessionId);
  renderHistory(payload.data.messages || []);
  renderSessionList();
  setStatus("会话已连接", "status-ready");
}

async function deleteSession(sessionId) {
  await requestJson(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    headers: { "X-Pickwise-CSRF": state.csrfToken },
  });
  const remaining = await refreshSessions();
  if (state.sessionId === sessionId) {
    state.sessionId = "";
    if (remaining.length) {
      await loadSession(remaining[0].session_id);
    } else {
      await createSession();
    }
  }
}

async function sendMessage(message) {
  const idempotencyKey = crypto.randomUUID();
  const payload = await requestJson("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Pickwise-CSRF": state.csrfToken,
      "Idempotency-Key": idempotencyKey,
    },
    body: JSON.stringify({ session_id: state.sessionId, message }),
  });
  return payload.data.reply;
}

async function handleChatSubmit(event) {
  event.preventDefault();
  if (state.busy || !state.sessionId) return;
  const message = elements.messageInput.value.trim();
  if (!message) return;

  showChatError("");
  appendMessage("user", message);
  elements.messageInput.value = "";
  setBusy(true);
  setStatus("正在处理", "status-busy");
  try {
    const reply = await sendMessage(message);
    appendMessage("assistant", reply);
    await refreshSessions();
    setStatus("会话已连接", "status-ready");
  } catch (error) {
    showChatError(error.message);
    appendMessage("assistant", "这次请求没有完成，请稍后重试。", "请求失败");
    setStatus("连接异常", "status-error");
  } finally {
    setBusy(false);
    elements.messageInput.focus();
  }
}

function renderProducts(products) {
  elements.productResults.replaceChildren();
  if (!products.length) {
    elements.productResults.innerHTML = '<div class="empty-state">没有找到匹配商品。</div>';
    return;
  }
  for (const product of products) {
    const card = document.createElement("article");
    card.className = "product-card";
    card.innerHTML = `
      <div class="product-card-top">
        <span class="product-category">${escapeHtml(product.category || "商品")}</span>
        <strong>${escapeHtml(product.product_id || "")}</strong>
      </div>
      <h3>${escapeHtml(product.name || "未命名商品")}</h3>
      <p>${escapeHtml(product.summary || "暂无简介")}</p>
      <div class="product-price">¥${Number(product.price || 0).toLocaleString("zh-CN")}</div>
    `;
    elements.productResults.append(card);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function handleProductSearch(event) {
  event.preventDefault();
  const query = elements.productQuery.value.trim();
  if (!query) return;
  elements.productResults.innerHTML = '<div class="empty-state">正在检索…</div>';
  try {
    const params = new URLSearchParams({ mode: "semantic", query, limit: "6" });
    const payload = await requestJson(`/api/products/search?${params.toString()}`, {
      headers: { "X-Pickwise-CSRF": state.csrfToken },
    });
    renderProducts(payload.data?.results || []);
  } catch (error) {
    elements.productResults.innerHTML = `<div class="empty-state error-state">${escapeHtml(error.message)}</div>`;
  }
}

elements.chatForm.addEventListener("submit", handleChatSubmit);
elements.productForm.addEventListener("submit", handleProductSearch);
elements.newSession.addEventListener("click", async () => {
  if (state.busy) return;
  try {
    await createSession();
    showChatError("");
  } catch (error) {
    showChatError(error.message);
    setStatus("连接异常", "status-error");
  }
});
elements.sessionList.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button || state.busy) return;
  const sessionId = button.dataset.sessionId;
  try {
    if (button.dataset.action === "delete") {
      if (window.confirm("确定删除这个会话吗？")) await deleteSession(sessionId);
    } else {
      await loadSession(sessionId);
    }
  } catch (error) {
    showChatError(error.message);
    setStatus("连接异常", "status-error");
  }
});
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

(async function boot() {
  try {
    await initializeIdentity();
    const sessions = await refreshSessions();
    const preferred = localStorage.getItem("pickwise.sessionId");
    const selected = sessions.find((item) => item.session_id === preferred) || sessions[0];
    if (selected) {
      await loadSession(selected.session_id);
    } else {
      await createSession();
    }
    elements.messageInput.focus();
  } catch (error) {
    showChatError(error.message);
    setStatus("连接异常", "status-error");
  }
})();

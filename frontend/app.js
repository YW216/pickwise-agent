const state = {
  csrfToken: "",
  sessionId: "",
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
    const message = payload?.error?.message || `请求失败（${response.status}）`;
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

async function createSession({ clearMessages = true } = {}) {
  const payload = await requestJson("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Pickwise-CSRF": state.csrfToken },
    body: JSON.stringify({}),
  });
  state.sessionId = payload.data.session_id;
  if (clearMessages) {
    elements.messages.replaceChildren();
    appendMessage("assistant", "你好，我是 PickWise。告诉我预算、使用场景和在意的参数，我会帮你缩小选择范围。", "会话已准备就绪");
  }
  setStatus("会话已连接", "status-ready");
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
  if (state.busy) return;
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
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

(async function boot() {
  try {
    await initializeIdentity();
    await createSession();
    elements.messageInput.focus();
  } catch (error) {
    showChatError(error.message);
    setStatus("连接异常", "status-error");
  }
})();

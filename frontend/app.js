import { readChatStream } from "/stream.js";

const state = {
  csrfToken: "",
  sessionId: "",
  sessions: [],
  busy: false,
  controller: null,
  pending: null,
};

const elements = {
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chat-form"),
  messageInput: document.querySelector("#message-input"),
  sendButton: document.querySelector("#send-button"),
  stopButton: document.querySelector("#stop-button"),
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
  elements.stopButton.hidden = !busy;
  elements.sendButton.textContent = busy ? "处理中…" : "发送";
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

async function sendMessage(pending, onEvent) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    credentials: "same-origin",
    signal: state.controller.signal,
    headers: {
      "Accept": "text/event-stream",
      "Content-Type": "application/json",
      "X-Pickwise-CSRF": state.csrfToken,
      "Idempotency-Key": pending.key,
    },
    body: JSON.stringify({ session_id: pending.sessionId, message: pending.message }),
  });
  await readChatStream(response, onEvent);
}

function createTimeline() {
  const bubble = appendMessage("assistant", "");
  bubble.classList.add("message-streaming");
  const body = bubble.querySelector("p");
  const details = document.createElement("details");
  details.className = "execution-details";
  details.open = true;
  const heading = document.createElement("summary");
  heading.textContent = "准备处理你的需求";
  const list = document.createElement("ol");
  list.className = "execution-phases";
  details.append(heading, list);
  body.before(details);
  const phaseTitles = {
    requirements: "理解需求",
    products: "筛选商品",
    knowledge: "查询资料",
    synthesis: "生成建议",
    persist: "保存会话",
  };
  const phaseRows = new Map();
  const toolRows = new Map();
  let failure = null;

  function ensurePhase(key, hint = "正在处理") {
    if (phaseRows.has(key)) return phaseRows.get(key);
    const title = phaseTitles[key] || "处理步骤";
    const row = document.createElement("li");
    row.className = "execution-phase";
    row.dataset.phase = key;
    row.dataset.state = "running";
    row.innerHTML = `
      <div class="execution-phase-head">
        <strong>${title}</strong>
        <span class="execution-phase-state">进行中</span>
      </div>
      <div class="execution-phase-message">${hint}</div>
      <ul class="execution-phase-details"></ul>
    `;
    list.append(row);
    phaseRows.set(key, row);
    return row;
  }

  function phaseFor(data) {
    if (data.phase && phaseTitles[data.phase]) return data.phase;
    if (data.stage === "routing") return "requirements";
    if (data.stage === "result") return "synthesis";
    if (data.stage === "save") return "persist";
    if (data.agent === "presale") return "products";
    if (data.agent === "consult") return "knowledge";
    if (["search_products", "search_catalog", "get_detail", "compare_products", "get_user_favorites"].includes(data.tool)) return "products";
    if (data.tool === "retrieve_knowledge") return "knowledge";
    return "synthesis";
  }

  function setPhase(key, state, message, count = null) {
    const row = ensurePhase(key, message);
    const priority = { pending: 0, running: 1, partial: 2, failed: 3, done: 4 };
    if (priority[state] < priority[row.dataset.state] && state !== "running") return;
    row.dataset.state = state;
    row.querySelector(".execution-phase-state").textContent = {
      pending: "等待", running: "进行中", partial: "部分完成",
      failed: "未完成", done: "已完成",
    }[state] || "处理中";
    if (message) row.querySelector(".execution-phase-message").textContent = message;
    if (count !== null && count !== undefined) row.dataset.count = String(count);
  }

  function addToolDetail(data, event) {
    const phase = phaseFor(data);
    const row = phaseRows.get(phase);
    if (!row) return;
    const detailKey = data.call_id || `${phase}:${data.tool || data.message}`;
    let detail = toolRows.get(detailKey);
    if (!detail) {
      detail = document.createElement("li");
      row.querySelector(".execution-phase-details").append(detail);
      toolRows.set(detailKey, detail);
    }
    detail.dataset.state = data.state || (event === "tool.started" ? "running" : event === "tool.failed" ? "failed" : "done");
    const name = data.tool_title || data.tool || "查询";
    detail.textContent = `${name}：${data.summary || data.message || "处理中"}`;
  }

  function completedSummary() {
    const summary = [];
    for (const [key, row] of phaseRows) {
      if (key === "requirements") summary.push("需求分析");
      else if (key === "products" && row.dataset.count) summary.push(`筛选 ${row.dataset.count} 款商品`);
      else if (key === "knowledge" && row.dataset.count) summary.push(`参考 ${row.dataset.count} 条资料`);
      else if (key === "products") summary.push("完成商品筛选");
      else if (key === "knowledge") summary.push("查询相关资料");
      else if (key === "synthesis") summary.push("生成建议");
      else if (key === "persist") summary.push("会话已保存");
      else summary.push(phaseTitles[key] || "完成处理");
    }
    return summary.length ? summary.join(" · ") : "处理完成";
  }

  return {
    event(event, data) {
      if (event === "status" || event.startsWith("tool.")) {
        const phase = phaseFor(data);
        ensurePhase(phase, data.message || data.summary);
        const status = event === "tool.finished" ? "running"
          : data.state || (event === "tool.started" ? "running"
            : event === "tool.failed" ? "failed" : "running");
        setPhase(phase, status, data.message || data.summary, data.count);
        if (event.startsWith("tool.")) addToolDetail(data, event);
        heading.textContent = `${data.title || phaseRows.get(phase)?.querySelector("strong")?.textContent || "正在处理"} · ${data.message || data.summary || "处理中"}`;
      } else if (event === "answer.delta") {
        body.textContent += data.text;
        heading.textContent = "正在显示回答";
      } else if (event === "error") {
        failure = data.message || "本轮处理失败";
        setPhase(data.phase || "synthesis", "failed", failure);
      } else if (event === "done") {
        // 以服务端最终快照为准，避免分段丢失、重复造成正文不一致。
        body.textContent = data.reply || "";
        bubble.classList.remove("message-streaming");
        if (!phaseRows.has("synthesis")) {
          setPhase("synthesis", data.failed || failure ? "failed" : "done",
            data.failed || failure ? "未能生成完整建议" : "已生成最终建议");
        } else if (phaseRows.get("synthesis").dataset.state === "running") {
          setPhase("synthesis", data.failed || failure ? "failed" : "done",
            data.failed || failure ? "未能生成完整建议" : "已生成最终建议");
        }
        const partial = [...phaseRows.values()].some((item) => item.dataset.state === "partial");
        heading.textContent = data.replayed ? "已恢复上次结果"
          : `${completedSummary()}${partial ? " · 部分检索未命中" : ""}`;
        details.open = false;
        if (data.warnings?.length) {
          showChatError(data.warnings.includes("PERSISTENCE_FAILED")
            ? "回答已生成，但会话未保存，请勿关闭页面"
            : "回答已生成，部分附属操作未完成");
        }
        if (data.failed || failure) showChatError(failure || "部分任务未完成");
        state.pending = null;
      }
      elements.messages.scrollTop = elements.messages.scrollHeight;
    },
    interrupted(message) {
      bubble.classList.remove("message-streaming");
      heading.textContent = "接收中断";
      for (const row of phaseRows.values()) {
        if (row.dataset.state === "running") {
          setPhase(row.dataset.phase, "failed", "接收中断，后台仍会继续处理");
        }
      }
      if (!body.textContent) body.textContent = message;
    },
  };
}

async function handleChatSubmit(event) {
  event.preventDefault();
  if (state.busy || !state.sessionId) return;
  const message = elements.messageInput.value.trim();
  if (!message) return;

  showChatError("");
  appendMessage("user", message);
  elements.messageInput.value = "";
  const pending = state.pending?.sessionId === state.sessionId && state.pending.message === message
    ? state.pending : { sessionId: state.sessionId, message, key: crypto.randomUUID() };
  state.pending = pending;
  state.controller = new AbortController();
  const timeline = createTimeline();
  setBusy(true);
  setStatus("正在处理", "status-busy");
  try {
    await sendMessage(pending, timeline.event);
    // 列表刷新失败不能把已经完成的聊天误报为失败。
    await refreshSessions().catch(() => showChatError("回答已完成，会话列表暂未刷新"));
    setStatus("会话已连接", "status-ready");
  } catch (error) {
    const message = error.name === "AbortError"
      ? "已停止接收；后台仍会完成并保存，请稍后重新打开本会话"
      : error.message;
    showChatError(message);
    timeline.interrupted(message);
    setStatus("接收中断", "status-error");
  } finally {
    setBusy(false);
    state.controller = null;
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
elements.stopButton.addEventListener("click", () => state.controller?.abort());
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
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
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

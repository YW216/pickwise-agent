// fetch 的网络分块不是 SSE 消息边界；保留半行，直到空行才分发事件。
export async function readChatStream(response, onEvent) {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error?.message || payload.detail || `请求失败（${response.status}）`);
  }
  if (!response.headers.get("content-type")?.includes("text/event-stream") || !response.body) {
    throw new Error("服务未返回事件流");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "message";
  let data = [];
  let completed = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let end;
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end).replace(/\r$/, "");
        buffer = buffer.slice(end + 1);
        if (!line) {
          if (data.length) {
            const payload = JSON.parse(data.join("\n"));
            onEvent(event, payload);
            if (event === "done") completed = true;
          }
          event = "message";
          data = [];
        } else if (line.startsWith("event:")) {
          event = line.slice(6).trimStart();
        } else if (line.startsWith("data:")) {
          data.push(line.slice(5).replace(/^ /, ""));
        }
      }
      if (done) break;
    }
    if (!completed) throw new Error("连接中断；后台可能仍在处理，请稍后刷新本会话查看结果");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

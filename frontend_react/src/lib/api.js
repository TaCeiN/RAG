const API_BASE = "http://127.0.0.1:8010";

export async function fetchChats() {
  const response = await fetch(`${API_BASE}/api/chats`);
  return response.json();
}

export async function createChat(title) {
  const response = await fetch(`${API_BASE}/api/chats`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return response.json();
}

export async function fetchMessages(chatId) {
  const response = await fetch(`${API_BASE}/api/chats/${chatId}/messages`);
  return response.json();
}

export async function fetchFiles(chatId) {
  const response = await fetch(`${API_BASE}/api/chats/${chatId}/files`);
  return response.json();
}

export async function uploadFiles(chatId, files) {
  const form = new FormData();
  for (const file of files) {
    form.append("file", file);
  }
  const response = await fetch(`${API_BASE}/api/chats/${chatId}/files`, {
    method: "POST",
    body: form,
  });
  return response.json();
}

export async function streamMessage(chatId, payload, onEvent) {
  const response = await fetch(`${API_BASE}/api/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    const bodyText = await response.text().catch(() => "");
    const details = bodyText ? `: ${bodyText}` : "";
    throw new Error(`Stream request failed (${response.status})${details}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    let newlineIndex = buffer.indexOf("\n");

    while (newlineIndex !== -1) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) onEvent(JSON.parse(line));
      newlineIndex = buffer.indexOf("\n");
    }
  }

  const tail = buffer.trim();
  if (tail) onEvent(JSON.parse(tail));
}

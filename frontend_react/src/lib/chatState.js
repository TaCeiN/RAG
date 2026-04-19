export function extractExtension(name) {
  if (!name) return "FILE";
  const part = String(name).split(".").pop();
  if (!part || part === name) return "FILE";
  return part.toUpperCase();
}

export function resolveFileName(file) {
  return file?.name || file?.filename || file?.file_name || "Unnamed file";
}

export function resolveFileId(file, index = 0) {
  return file?.id || file?.file_id || `${resolveFileName(file)}-${file?.created_at || index}`;
}

export function toFileAttachments(files = []) {
  return files.map((file) => {
    const name = resolveFileName(file);
    return {
      id: resolveFileId(file),
      name,
      extension: extractExtension(name),
    };
  });
}

export function makeMessage(role, content = "", attachments = [], options = {}) {
  return {
    id: options.id || `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content,
    thinking: options.thinking || "",
    attachments,
    renderMode: options.renderMode || "markdown",
  };
}

export function hydrateMessagesWithFiles(rows = [], files = []) {
  const messages = rows.map((item) =>
    makeMessage(item.role, item.content, [], {
      id: item.id,
      renderMode: "markdown",
    })
  );
  const attachments = toFileAttachments(files);
  if (!attachments.length) return messages;

  const alreadyVisible = messages.some((message) => message.attachments?.length > 0);
  if (alreadyVisible) return messages;

  const firstUserIndex = messages.findIndex((message) => message.role === "user");
  if (firstUserIndex >= 0) {
    return messages.map((message, index) =>
      index === firstUserIndex ? { ...message, attachments } : message
    );
  }

  return [
    makeMessage("user", "", attachments, {
      id: `files-${attachments.map((item) => item.id || item.name).join("-")}`,
    }),
    ...messages,
  ];
}

export function updateAssistantMessage(messages, assistantMessageId, updater) {
  return messages.map((message) =>
    message.id === assistantMessageId ? updater(message) : message
  );
}

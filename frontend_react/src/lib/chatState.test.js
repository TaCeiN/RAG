import assert from "node:assert/strict";
import test from "node:test";

import {
  hydrateMessagesWithFiles,
  updateAssistantMessage,
} from "./chatState.js";

test("hydrateMessagesWithFiles restores uploaded file cards on persisted user messages", () => {
  const rows = [{ id: "m1", role: "user", content: "Разбери документ" }];
  const files = [{ id: "f1", name: "report.docx" }];

  const messages = hydrateMessagesWithFiles(rows, files);

  assert.equal(messages.length, 1);
  assert.equal(messages[0].attachments.length, 1);
  assert.equal(messages[0].attachments[0].name, "report.docx");
  assert.equal(messages[0].attachments[0].extension, "DOCX");
});

test("hydrateMessagesWithFiles creates a visible file card for file-only chats", () => {
  const messages = hydrateMessagesWithFiles([], [{ id: "f1", name: "data.pdf" }]);

  assert.equal(messages.length, 1);
  assert.equal(messages[0].role, "user");
  assert.equal(messages[0].content, "");
  assert.equal(messages[0].attachments[0].name, "data.pdf");
});

test("updateAssistantMessage updates the in-flight assistant draft by id", () => {
  const messages = [
    { id: "u1", role: "user", content: "Вопрос", thinking: "", attachments: [] },
    { id: "a1", role: "assistant", content: "", thinking: "", attachments: [] },
  ];

  const updated = updateAssistantMessage(messages, "a1", (message) => ({
    ...message,
    content: `${message.content}Ответ`,
  }));

  assert.equal(updated[1].content, "Ответ");
  assert.equal(messages[1].content, "");
});

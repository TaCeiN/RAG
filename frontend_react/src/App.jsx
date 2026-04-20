import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  cancelFileProcessing,
  createChat,
  fetchChats,
  fetchFiles,
  fetchMessages,
  streamMessage,
  updateChatTitle,
  uploadFiles,
} from "./lib/api";
import {
  extractExtension,
  hydrateMessagesWithFiles,
  makeMessage,
  resolveFileId,
  resolveFileName,
  updateAssistantMessage,
} from "./lib/chatState";

function deriveTitleFromText(text) {
  const words = String(text || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 3);
  return words.join(" ").trim();
}

function deriveTitleFromFile(file) {
  const base = String(file?.name || "")
    .replace(/\.[^.]+$/, "")
    .replace(/[_-]+/g, " ")
    .trim();
  return deriveTitleFromText(base);
}

function buildChatTitle(text, files) {
  const byText = deriveTitleFromText(text);
  if (byText) return byText;
  if (files?.length) {
    const byFile = deriveTitleFromFile(files[0]);
    if (byFile) return byFile;
  }
  return "New chat";
}

function isGenericChatTitle(title) {
  const value = String(title || "").trim().toLowerCase();
  if (!value) return true;
  return value === "new chat" || /^chat\s+\d+$/.test(value);
}

function isLikelyDebugPayload(text) {
  const value = String(text || "").trim();
  if (!value.startsWith("{") || !value.endsWith("}")) return false;
  return (
    value.includes("\"retrieval_mode\"") ||
    value.includes("\"vector_candidates\"") ||
    value.includes("\"route_meta\"")
  );
}

function normalizeSummaryPoints(file) {
  if (Array.isArray(file?.summary_key_points)) return file.summary_key_points.filter(Boolean);
  if (typeof file?.summary_key_points_json === "string") {
    try {
      const parsed = JSON.parse(file.summary_key_points_json);
      return Array.isArray(parsed) ? parsed.filter(Boolean) : [];
    } catch {
      return [];
    }
  }
  return [];
}

function MarkdownMessage({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="mdP">{children}</p>,
        ul: ({ children }) => <ul className="mdList">{children}</ul>,
        ol: ({ children }) => <ol className="mdList">{children}</ol>,
        li: ({ children }) => <li className="mdItem">{children}</li>,
        strong: ({ children }) => <strong className="mdStrong">{children}</strong>,
        code: ({ children }) => <code className="mdCode">{children}</code>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

export default function App() {
  const STARTER_PRESETS = [
    { label: "Code", prompt: "Помоги разобраться с кодом и предложи пошаговый план." },
    { label: "Learn", prompt: "Объясни тему простыми словами и дай мини-практику." },
    { label: "Strategize", prompt: "Помоги продумать стратегию и сравнить варианты решения." },
    { label: "Write", prompt: "Помоги написать структурированный текст по моей задаче." },
    { label: "Life stuff", prompt: "Помоги разобрать жизненную ситуацию практично и спокойно." },
  ];

  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("Ollama ready");
  const [debug, setDebug] = useState(false);
  const [think, setThink] = useState(false);
  const [responseMode, setResponseMode] = useState(null);
  const [isAwaitingAnswer, setIsAwaitingAnswer] = useState(false);
  const [queuedPrompt, setQueuedPrompt] = useState(null);
  const [statusPhraseIndex, setStatusPhraseIndex] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [selectedSource, setSelectedSource] = useState(null);
  const [isPinnedToBottom, setIsPinnedToBottom] = useState(true);
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [copiedSummaryId, setCopiedSummaryId] = useState(null);
  const messagesRef = useRef(null);
  const streamEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const skipHydrationChatIdRef = useRef(null);
  const activeChatIdRef = useRef(null);
  const chatRuntimeRef = useRef(new Map());
  const activeStreamRef = useRef(null);

  const activeChat = useMemo(() => chats.find((c) => c.id === activeChatId), [chats, activeChatId]);
  const blockingStatuses = useMemo(() => new Set(["uploaded", "indexing", "summarizing"]), []);
  const blockingFile = useMemo(
    () => uploadedFiles.find((file) => blockingStatuses.has(String(file?.status || "").toLowerCase())) || null,
    [uploadedFiles, blockingStatuses]
  );
  const isInputLocked = Boolean(blockingFile);
  const canInterrupt = isAwaitingAnswer || isInputLocked;
  useEffect(() => {
    activeChatIdRef.current = activeChatId;
  }, [activeChatId]);

  const applyRuntimeIfActive = (chatId, runtime) => {
    if (activeChatIdRef.current !== chatId) return;
    setMessages(runtime.messages || []);
    setStatus(runtime.status || "Ollama ready");
    setIsAwaitingAnswer(Boolean(runtime.isAwaitingAnswer));
    setResponseMode(runtime.responseMode || null);
  };

  const updateChatRuntime = (chatId, updater) => {
    const previous = chatRuntimeRef.current.get(chatId) || {
      messages: [],
      status: "Ollama ready",
      isAwaitingAnswer: false,
      responseMode: null,
    };
    const next = updater(previous);
    chatRuntimeRef.current.set(chatId, next);
    applyRuntimeIfActive(chatId, next);
    return next;
  };

  const closeSidebar = () => setSidebarOpen(false);
  const toggleSidebar = () => setSidebarOpen((value) => !value);
  const closeSources = () => setSourcesOpen(false);
  const toggleSources = () => setSourcesOpen((value) => !value);
  const openHome = () => {
    setActiveChatId(null);
    setMessages([]);
    setUploadedFiles([]);
    setPendingFiles([]);
    setResponseMode(null);
    setIsAwaitingAnswer(false);
    setInput("");
    setStatus("Ollama ready");
    setSelectedSource(null);
    closeSidebar();
    closeSources();
  };
  const onSelectChat = (chatId) => {
    setActiveChatId(chatId);
    const runtime = chatRuntimeRef.current.get(chatId);
    if (runtime?.isAwaitingAnswer) {
      setMessages(runtime.messages || []);
      setResponseMode(runtime.responseMode || null);
      setIsAwaitingAnswer(true);
      setStatus(runtime.status || "Streaming...");
    } else {
      setResponseMode(null);
      setIsAwaitingAnswer(false);
      setStatus("Ollama ready");
    }
    setSelectedSource(null);
    closeSidebar();
    closeSources();
  };

  const statusNarrativeVariants = useMemo(() => {
    const low = status.toLowerCase();
    if (!isAwaitingAnswer) return [];
    if (low.includes("upload") || low.includes("index")) {
      return [
        "Готовлю документ к поиску.",
        "Индексация идет, осталось немного.",
        "Собираю контекст из загруженных файлов.",
      ];
    }
    if (low.includes("search") || low.includes("?????") || low.includes("???????????") || low.includes("????????")) {
      return [
        "Ищу самые релевантные фрагменты.",
        "Сверяю вопрос с содержимым документа.",
        "Отбираю лучший контекст для ответа.",
      ];
    }
    if (low.includes("generat") || low.includes("?????????") || low.includes("stream")) {
      return [
        "Формирую итоговый ответ.",
        "Почти готово, дописываю детали.",
        "Проверяю формулировки перед отправкой.",
      ];
    }
    return [
      "Думаю над запросом.",
      "Уточняю детали для более точного ответа.",
      "Секунду, подбираю лучшую формулировку.",
    ];
  }, [isAwaitingAnswer, status]);

  const statusNarrative = statusNarrativeVariants.length
    ? statusNarrativeVariants[statusPhraseIndex % statusNarrativeVariants.length]
    : "";

  const scrollToBottom = () => {
    streamEndRef.current?.scrollIntoView({ block: "end" });
  };

  const onMessagesScroll = () => {
    const node = messagesRef.current;
    if (!node) return;
    const distanceToBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
    setIsPinnedToBottom(distanceToBottom < 64);
  };

  useEffect(() => {
    const bootstrap = async () => {
      let rows = await fetchChats();
      if (!rows.length) {
        const created = await createChat("New chat");
        rows = [{ id: created.id, title: "New chat" }];
      }
      setChats(rows);
      setActiveChatId(null);
    };
    bootstrap().catch((error) => console.error(error));
  }, []);

  useEffect(() => {
    if (!activeChatId) return;
    if (skipHydrationChatIdRef.current === activeChatId) {
      skipHydrationChatIdRef.current = null;
      return;
    }
    Promise.all([fetchMessages(activeChatId), fetchFiles(activeChatId)])
      .then(([rows, files]) => {
        const runtime = chatRuntimeRef.current.get(activeChatId);
        setMessages(runtime?.isAwaitingAnswer ? runtime.messages || [] : hydrateMessagesWithFiles(rows, files));
        setUploadedFiles(files);
        setPendingFiles([]);
        setResponseMode(runtime?.isAwaitingAnswer ? runtime.responseMode || null : null);
        setIsPinnedToBottom(true);
        setIsAwaitingAnswer(Boolean(runtime?.isAwaitingAnswer));
        setStatus(runtime?.isAwaitingAnswer ? runtime.status || "Streaming..." : "Ollama ready");
        setSourcesOpen(false);
        setSelectedSource(null);
      })
      .catch((error) => console.error(error));
  }, [activeChatId]);

  useEffect(() => {
    if (!activeChatId) return undefined;
    const hasPendingUploaded = uploadedFiles.some((file) =>
      blockingStatuses.has(String(file?.status || "").toLowerCase())
    );
    if (!hasPendingUploaded) return undefined;
    const timerId = window.setInterval(() => {
      fetchFiles(activeChatId)
        .then((files) => {
          if (activeChatIdRef.current !== activeChatId) return;
          setUploadedFiles(files);
        })
        .catch((error) => console.error(error));
    }, 1200);
    return () => window.clearInterval(timerId);
  }, [activeChatId, uploadedFiles, blockingStatuses]);

  useEffect(() => {
    if (!isPinnedToBottom) return;
    scrollToBottom();
  }, [isPinnedToBottom, messages, statusNarrative]);

  useEffect(() => {
    setStatusPhraseIndex(0);
  }, [status, isAwaitingAnswer]);

  useEffect(() => {
    if (!isAwaitingAnswer || statusNarrativeVariants.length <= 1) return;
    const timerId = window.setInterval(() => {
      setStatusPhraseIndex((value) => value + 1);
    }, 5000);
    return () => window.clearInterval(timerId);
  }, [isAwaitingAnswer, statusNarrativeVariants]);

  const onCreateChat = async () => {
    const title = `Chat ${chats.length + 1}`;
    const created = await createChat(title);
    setChats((prev) => [{ id: created.id, title }, ...prev]);
    setActiveChatId(created.id);
    setMessages([]);
    setUploadedFiles([]);
    setPendingFiles([]);
    setResponseMode(null);
    setIsAwaitingAnswer(false);
    setStatus("Ollama ready");
    setSourcesOpen(false);
    closeSidebar();
  };

  const maybeRenameChatByFirstPrompt = async (chatId, content, files = []) => {
    const prompt = String(content || "").trim();
    if (!chatId || !prompt) return;
    const currentTitle = chats.find((chat) => chat.id === chatId)?.title || "";
    if (!isGenericChatTitle(currentTitle)) return;

    const nextTitle = buildChatTitle(prompt, files);
    if (!nextTitle || nextTitle === currentTitle) return;

    try {
      await updateChatTitle(chatId, nextTitle);
      setChats((prev) => prev.map((chat) => (chat.id === chatId ? { ...chat, title: nextTitle } : chat)));
    } catch (error) {
      console.error(error);
    }
  };

  const appendUserMessageToChat = (targetChatId, content, attachments = []) => {
    const userMessage = makeMessage("user", content, attachments);
    const baseMessages =
      chatRuntimeRef.current.get(targetChatId)?.messages ||
      (activeChatIdRef.current === targetChatId ? messages : []);
    const nextMessages = [...baseMessages, userMessage];
    updateChatRuntime(targetChatId, (runtime) => ({
      ...runtime,
      messages: nextMessages,
    }));
    setMessages(nextMessages);
    setIsPinnedToBottom(true);
  };

  const sendQuestionNow = async (targetChatId, content, messageAttachments, options = {}) => {
    const appendUserMessage = options.appendUserMessage !== false;
    const assistantMessage = makeMessage("assistant", "", [], { renderMode: debug ? "plain" : "markdown" });
    const assistantMessageId = assistantMessage.id;
    if (appendUserMessage) {
      appendUserMessageToChat(targetChatId, content, messageAttachments);
    }
    const baseMessages =
      chatRuntimeRef.current.get(targetChatId)?.messages ||
      (activeChatIdRef.current === targetChatId ? messages : []);
    const nextMessages = [...baseMessages, assistantMessage];

    updateChatRuntime(targetChatId, (runtime) => ({
      ...runtime,
      messages: nextMessages,
      status: "Streaming...",
      isAwaitingAnswer: true,
      responseMode: null,
    }));
    setMessages(nextMessages);
    setIsPinnedToBottom(true);
    setIsAwaitingAnswer(true);
    setStatus("Streaming...");

    try {
      const controller = new AbortController();
      activeStreamRef.current = { chatId: targetChatId, assistantMessageId, controller };
      await streamMessage(
        targetChatId,
        {
          content,
          think,
          debug_retrieval: debug,
          retrieval_mode: "hybrid_plus",
          force_rag_on_upload: false,
        },
        (event) => {
          const mode = event?.trace?.route_meta?.response_mode;
          if (mode === "rag" || mode === "direct_chat") {
            updateChatRuntime(targetChatId, (runtime) => ({ ...runtime, responseMode: mode }));
          }
          if (event.type === "status") {
            updateChatRuntime(targetChatId, (runtime) => ({
              ...runtime,
              status: event.message || "Model is working...",
              isAwaitingAnswer: true,
            }));
            return;
          }
          if (event.type === "search_started") {
            updateChatRuntime(targetChatId, (runtime) => ({
              ...runtime,
              status: "Searching context...",
              isAwaitingAnswer: true,
            }));
            return;
          }
          if (event.type === "search_ready") {
            updateChatRuntime(targetChatId, (runtime) => ({
              ...runtime,
              status: "Generating answer...",
              isAwaitingAnswer: true,
            }));
            return;
          }
          if (event.type === "retrieval_confidence") {
            const files = Array.isArray(event.files) && event.files.length ? event.files.join(", ") : "без файла";
            const confidence = String(event.level || "unknown");
            const sources = Number(event.sources_count || 0);
            updateChatRuntime(targetChatId, (runtime) => ({
              ...runtime,
              status: `Файл: ${files} · найдено ${sources} фрагм. · уверенность ${confidence}`,
              isAwaitingAnswer: true,
            }));
            return;
          }
          if (event.type === "thinking") {
            updateChatRuntime(targetChatId, (runtime) => ({
              ...runtime,
              messages: updateAssistantMessage(runtime.messages || [], assistantMessage.id, (message) => ({
                ...message,
                thinking: `${message.thinking}${event.delta || ""}`,
              })),
              isAwaitingAnswer: true,
            }));
            return;
          }
          if (event.type === "answer") {
            updateChatRuntime(targetChatId, (runtime) => ({
              ...runtime,
              messages: updateAssistantMessage(runtime.messages || [], assistantMessage.id, (message) => ({
                ...message,
                content: `${message.content}${event.delta || ""}`,
              })),
              isAwaitingAnswer: true,
            }));
            return;
          }
          if (event.type === "error") {
            const errorText = event.message || "Во время ответа произошла ошибка модели.";
            updateChatRuntime(targetChatId, (runtime) => ({
              ...runtime,
              messages: updateAssistantMessage(runtime.messages || [], assistantMessage.id, (message) => ({
                ...message,
                content: errorText,
              })),
              isAwaitingAnswer: false,
              status: "Model error",
            }));
            return;
          }
          if (event.type === "done") {
            updateChatRuntime(targetChatId, (runtime) => ({
              ...runtime,
              isAwaitingAnswer: false,
              status: "Ollama ready",
            }));
            chatRuntimeRef.current.delete(targetChatId);
            if (activeStreamRef.current?.assistantMessageId === assistantMessageId) {
              activeStreamRef.current = null;
            }
          }
        }
        ,
        {
          signal: controller.signal,
        }
      );
    } catch (error) {
      if (activeStreamRef.current?.assistantMessageId === assistantMessageId) {
        activeStreamRef.current = null;
      }
      if (error?.name === "AbortError") {
        const interruptedText = "Генерация была прервана пользователем.";
        updateChatRuntime(targetChatId, (runtime) => ({
          ...runtime,
          messages: updateAssistantMessage(runtime.messages || [], assistantMessageId, (message) => ({
            ...message,
            content: interruptedText,
          })),
          isAwaitingAnswer: false,
          status: "Остановлено",
        }));
        chatRuntimeRef.current.delete(targetChatId);
        if (activeChatIdRef.current === targetChatId) {
          setIsAwaitingAnswer(false);
          setStatus("Остановлено");
        }
        return;
      }
      console.error(error);
      const errorText = String(error?.message || "Request failed");
      updateChatRuntime(targetChatId, (runtime) => ({
        ...runtime,
        messages: updateAssistantMessage(runtime.messages || [], assistantMessageId, (message) => ({
          ...message,
          content: errorText,
        })),
        isAwaitingAnswer: false,
        status: errorText,
      }));
      if (activeChatIdRef.current === targetChatId) {
        setIsAwaitingAnswer(false);
        setStatus(errorText);
      }
    }
  };

  const onSend = async () => {
    const content = input.trim();
    if (isInputLocked) {
      if (content && activeChatId) {
        await maybeRenameChatByFirstPrompt(activeChatId, content, []);
        appendUserMessageToChat(activeChatId, content, []);
        setQueuedPrompt({ chatId: activeChatId, content });
        setInput("");
        setStatus("Вопрос сохранен. Отправлю его автоматически после готовности summary.");
      } else {
        setStatus("Файл загружен. Сначала формируется краткое содержание, после этого можно задавать вопросы.");
      }
      return;
    }
    const hasPendingFiles = pendingFiles.length > 0;
    if (!content && !hasPendingFiles) return;
    const stagedFiles = hasPendingFiles ? [...pendingFiles] : [];
    const messageAttachments = stagedFiles.map((file) => ({
      name: file.name,
      extension: extractExtension(file.name),
    }));
    let targetChatId = activeChatId;

    try {
      if (!targetChatId) {
        const title = buildChatTitle(content, stagedFiles);
        const created = await createChat(title);
        targetChatId = created.id;
        skipHydrationChatIdRef.current = targetChatId;
        setChats((prev) => [{ id: targetChatId, title }, ...prev]);
        setActiveChatId(targetChatId);
        setMessages([]);
        setUploadedFiles([]);
        setResponseMode(null);
        setSourcesOpen(false);
      }

      if (hasPendingFiles) {
        setStatus("Uploading and indexing...");
        await uploadFiles(targetChatId, stagedFiles);
        const files = await fetchFiles(targetChatId);
        setUploadedFiles(files);
        setPendingFiles([]);
        if (fileInputRef.current) fileInputRef.current.value = "";
        setStatus("Files uploaded");
        const hasBlockingAfterUpload = files.some((file) =>
          blockingStatuses.has(String(file?.status || "").toLowerCase())
        );
        if (content && hasBlockingAfterUpload) {
          await maybeRenameChatByFirstPrompt(targetChatId, content, stagedFiles);
          appendUserMessageToChat(targetChatId, content, messageAttachments);
          setInput("");
          setQueuedPrompt({ chatId: targetChatId, content });
          setStatus("Файл обрабатывается. Вопрос сохранен и будет отправлен автоматически.");
          return;
        }
      }

      if (!content) {
        if (messageAttachments.length > 0) {
          setMessages((prev) => [...prev, makeMessage("user", "", messageAttachments)]);
          setIsPinnedToBottom(true);
        }
        return;
      }

      await maybeRenameChatByFirstPrompt(targetChatId, content, stagedFiles);
      setInput("");
      await sendQuestionNow(targetChatId, content, messageAttachments, { appendUserMessage: true });
    } catch (error) {
      console.error(error);
      const errorText = String(error?.message || "Request failed");
      if (activeChatId) {
        setMessages((prev) => [...prev, makeMessage("assistant", errorText)]);
      }
      if (!targetChatId || activeChatIdRef.current === targetChatId) {
        setIsAwaitingAnswer(false);
        setStatus(errorText);
      }
    }
  };

  const onUpload = async (event) => {
    if (isInputLocked) return;
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    setPendingFiles((prev) => {
      const known = new Set(prev.map((item) => `${item.name}:${item.size}:${item.lastModified}`));
      const fresh = files.filter((item) => !known.has(`${item.name}:${item.size}:${item.lastModified}`));
      return [...prev, ...fresh];
    });
    setStatus("Files staged. Press Send to upload.");
    setIsPinnedToBottom(true);
    event.target.value = "";
  };

  const removePendingFile = (targetKey) => {
    setPendingFiles((prev) => prev.filter((file) => `${file.name}:${file.size}:${file.lastModified}` !== targetKey));
  };

  const onPresetClick = (prompt) => {
    setInput(prompt);
    textareaRef.current?.focus();
  };

  const onCopyMessage = async (messageId, content) => {
    try {
      await navigator.clipboard.writeText(String(content || ""));
      setCopiedMessageId(messageId);
      window.setTimeout(() => {
        setCopiedMessageId((current) => (current === messageId ? null : current));
      }, 1500);
    } catch (error) {
      console.error(error);
      setStatus("Не удалось скопировать текст");
    }
  };

  useEffect(() => {
    if (!queuedPrompt) return;
    if (isAwaitingAnswer) return;
    if (activeChatIdRef.current !== queuedPrompt.chatId) return;
    const hasPendingUploaded = uploadedFiles.some((file) =>
      blockingStatuses.has(String(file?.status || "").toLowerCase())
    );
    if (hasPendingUploaded) return;
    const queuedContent = String(queuedPrompt.content || "").trim();
    if (!queuedContent) {
      setQueuedPrompt(null);
      return;
    }
    setQueuedPrompt(null);
    setStatus("Summary готово. Отправляю сохраненный вопрос...");
    sendQuestionNow(queuedPrompt.chatId, queuedContent, [], { appendUserMessage: false }).catch((error) => {
      console.error(error);
      setStatus("Не удалось отправить сохраненный вопрос");
    });
  }, [queuedPrompt, isAwaitingAnswer, uploadedFiles, blockingStatuses]);

  const onCopySummary = async (file) => {
    const fileId = resolveFileId(file);
    const summary = String(file?.summary || "").trim();
    if (!summary) return;
    try {
      await navigator.clipboard.writeText(summary);
      setCopiedSummaryId(fileId);
      window.setTimeout(() => {
        setCopiedSummaryId((current) => (current === fileId ? null : current));
      }, 1500);
    } catch (error) {
      console.error(error);
      setStatus("Не удалось скопировать summary");
    }
  };

  const onComposerKeyDown = (event) => {
    if (event.key !== "Enter") return;
    if (event.shiftKey) return;
    if (event.nativeEvent?.isComposing) return;
    event.preventDefault();
    onSend();
  };

  const onStopStream = () => {
    const active = activeStreamRef.current;
    if (active?.controller) {
      active.controller.abort();
      return;
    }
    if (!activeChatId || !isInputLocked) return;
    const pendingUploads = uploadedFiles.filter((file) =>
      blockingStatuses.has(String(file?.status || "").toLowerCase())
    );
    Promise.all(
      pendingUploads
        .map((file) => resolveFileId(file))
        .filter(Boolean)
        .map((fileId) => cancelFileProcessing(activeChatId, fileId))
    )
      .then(async () => {
        setQueuedPrompt(null);
        setPendingFiles([]);
        if (fileInputRef.current) fileInputRef.current.value = "";
        const files = await fetchFiles(activeChatId);
        setUploadedFiles(files);
        setMessages((prev) => [...prev, makeMessage("assistant", "Обработка файла была прервана пользователем.")]);
        setStatus("Остановлено");
      })
      .catch((error) => {
        console.error(error);
        setStatus("Не удалось прервать обработку файла");
      });
  };

  return (
    <div className="appShell">
      <aside className={sidebarOpen ? "sidebar isOpen" : "sidebar"}>
        <div className="sidebarIntro">
          <button className="brandBtn" onClick={openHome} aria-label="Open home">
            Claudely
          </button>
        </div>
        <div className="headerRow">
          <h1>Chats</h1>
          <button className="newChatBtn" onClick={onCreateChat} aria-label="New chat">
            +
          </button>
        </div>
        <div className="chatList">
          {chats.map((chat) => (
            <button
              key={chat.id}
              className={chat.id === activeChatId ? "chatItem active" : "chatItem"}
              onClick={() => onSelectChat(chat.id)}
            >
              <span className="chatItemTitle">{chat.title}</span>
              {chat.id === activeChatId && <span className="chatItemMore">...</span>}
            </button>
          ))}
        </div>
      </aside>
      {sidebarOpen && <button className="backdrop" onClick={closeSidebar} aria-label="Close sidebar" />}

      <main className="mainPanel">
        <header className="mainHeader">
          <button className="menuBtn" onClick={toggleSidebar} aria-label="Open chats">
            Chats
          </button>
          <div className="headerTitleBlock">
            <h2>{activeChat?.title || "Claudely"}</h2>
          </div>
          {uploadedFiles.length > 0 && (
            <div className="headerActions">
              <span
                className={
                  responseMode === "rag"
                    ? "modeDot isRag"
                    : responseMode === "direct_chat"
                      ? "modeDot isDirect"
                      : "modeDot"
                }
                aria-hidden="true"
                title={
                  responseMode === "rag"
                    ? "RAG mode"
                    : responseMode === "direct_chat"
                      ? "Direct chat mode"
                      : "Mode unknown"
                }
              />
              <button className={sourcesOpen ? "sourcesBtn isActive" : "sourcesBtn"} onClick={toggleSources}>
                Sources
              </button>
            </div>
          )}
        </header>
        <div className={uploadedFiles.length > 0 && sourcesOpen ? "mainBody hasSources" : "mainBody"}>
                    <section className="conversation">
            <div className="messages" ref={messagesRef} onScroll={onMessagesScroll}>
              {!activeChat && (
                <div className="emptyState heroState">
                  <p>Что будем продумывать сегодня?</p>
                </div>
              )}
              {!activeChat && (
                <div className="starterChips" aria-label="Starter prompts">
                  {STARTER_PRESETS.map((preset) => (
                    <button key={preset.label} type="button" className="starterChip" onClick={() => onPresetClick(preset.prompt)}>
                      {preset.label}
                    </button>
                  ))}
                </div>
              )}
              {activeChat && messages.length === 0 && (
                <div className="emptyState">
                  <p>Напишите вопрос или загрузите файл, чтобы начать.</p>
                </div>
              )}
              {activeChat && messages.map((message) => (
                <div key={message.id} className={`msg ${message.role}`}>
                  {message.role === "user" && message.attachments?.length > 0 && (
                    <div className="messageAttachments">
                      {message.attachments.map((attachment, index) => (
                        <article key={`${attachment.name}-${index}`} className="messageFileCard">
                          <p>{attachment.name}</p>
                          <span>{attachment.extension || extractExtension(attachment.name)}</span>
                        </article>
                      ))}
                    </div>
                  )}
                  {message.thinking && <pre className="thinking">{message.thinking}</pre>}
                  {message.content && (
                    <div className="mdContent">
                      {message.renderMode === "plain" || (message.role === "assistant" && isLikelyDebugPayload(message.content)) ? (
                        <pre className="plainContent">{message.content}</pre>
                      ) : (
                        <MarkdownMessage content={message.content} />
                      )}
                    </div>
                  )}
                  {message.role === "assistant" && message.content && (
                    <div className="messageActions">
                      <button
                        type="button"
                        className="messageActionBtn"
                        onClick={() => onCopyMessage(message.id, message.content)}
                      >
                        {copiedMessageId === message.id ? "Скопировано" : "Копировать"}
                      </button>
                    </div>
                  )}
                </div>
              ))}
              {activeChat && isAwaitingAnswer && statusNarrative && (
                <div className="statusNarrative" role="status" aria-live="polite">
                  <span className="statusSpinner" aria-hidden="true" />
                  <span>{statusNarrative}</span>
                </div>
              )}
              {activeChat && !isAwaitingAnswer && isInputLocked && (
                <div className="statusNarrative" role="status" aria-live="polite">
                  <span className="statusSpinner" aria-hidden="true" />
                  <span>Файл загружен. Сначала формируется краткое содержание, после этого можно задавать вопросы.</span>
                </div>
              )}
              <div ref={streamEndRef} />
            </div>

            <div className="composerWrap">
              <div className="composer">
                {pendingFiles.length > 0 && (
                  <div className="pendingFiles">
                    {pendingFiles.map((file) => {
                      const fileKey = `${file.name}:${file.size}:${file.lastModified}`;
                      const extension = file.name.split(".").pop()?.toUpperCase() || "FILE";
                      return (
                        <article className="pendingFileCard" key={fileKey}>
                          <p>{file.name}</p>
                          <div className="pendingFileMeta">
                            <span>{extension}</span>
                            <button type="button" onClick={() => removePendingFile(fileKey)} aria-label="Remove file">
                              x
                            </button>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                )}

                <div className="composerMain">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={onComposerKeyDown}
                    disabled={isInputLocked}
                    placeholder="Ask about uploaded files..."
                  />
                </div>

                <div className="composerActions">
                  <div className="actionIcons">
                    <button
                      type="button"
                      className={think ? "iconToggle isActive" : "iconToggle"}
                      aria-pressed={think}
                      onClick={() => setThink((value) => !value)}
                      title="Thinking"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          fill="currentColor"
                          d="M4 18a2 2 0 1 1 0 4a2 2 0 0 1 0-4m5.5-3a2.5 2.5 0 1 1 0 5a2.5 2.5 0 0 1 0-5M12 2a5.414 5.414 0 0 1 5.33 4.47h.082a3.765 3.765 0 1 1 0 7.53H6.588a3.765 3.765 0 1 1 0-7.53h.082A5.414 5.414 0 0 1 12 2"
                        />
                      </svg>
                    </button>

                    <button
                      type="button"
                      className={debug ? "iconToggle isActive" : "iconToggle"}
                      aria-pressed={debug}
                      onClick={() => setDebug((value) => !value)}
                      title="Debug"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 32 32" aria-hidden="true">
                        <path
                          fill="currentColor"
                          d="m29.83 20l.34-2l-5.17-.85v-4.38l5.06-1.36l-.51-1.93l-4.83 1.29A9 9 0 0 0 20 5V2h-2v2.23a8.8 8.8 0 0 0-4 0V2h-2v3a9 9 0 0 0-4.71 5.82L2.46 9.48L2 11.41l5 1.36v4.38L1.84 18l.32 2L7 19.18a8.9 8.9 0 0 0 .82 3.57l-4.53 4.54l1.42 1.42l4.19-4.2a9 9 0 0 0 14.2 0l4.19 4.2l1.42-1.42l-4.54-4.54a8.9 8.9 0 0 0 .83-3.57ZM15 25.92A7 7 0 0 1 9 19v-6h6ZM9.29 11a7 7 0 0 1 13.42 0ZM23 19a7 7 0 0 1-6 6.92V13h6Z"
                        />
                      </svg>
                    </button>

                    <label className={isInputLocked ? "iconToggle uploadToggle isDisabled" : "iconToggle uploadToggle"} title="Upload files">
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          fill="none"
                          stroke="currentColor"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="1.5"
                          d="m18.375 12.739l-7.693 7.693a4.5 4.5 0 0 1-6.364-6.364l10.94-10.94A3 3 0 1 1 19.5 7.372L8.552 18.32m.009-.01l-.01.01m5.699-9.941l-7.81 7.81a1.5 1.5 0 0 0 2.112 2.13"
                        />
                      </svg>
                      <input ref={fileInputRef} type="file" multiple onChange={onUpload} hidden disabled={isInputLocked} />
                    </label>
                  </div>

                  <button
                    className="sendBtn"
                    onClick={canInterrupt ? onStopStream : onSend}
                    aria-label={canInterrupt ? "Stop generation" : "Send"}
                    disabled={false}
                    title={canInterrupt ? "Остановить ответ" : "Отправить"}
                  >
                    {canInterrupt ? (
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                        role="img"
                        aria-hidden="true"
                      >
                        <rect x="7" y="7" width="10" height="10" fill="currentColor" rx="1.5" />
                      </svg>
                    ) : (
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                        role="img"
                        aria-hidden="true"
                      >
                        <path
                          d="M12 17V7m0 0l-4 4m4-4l4 4"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </section>
          {uploadedFiles.length > 0 && (
            <aside className={sourcesOpen ? "sourcesPanel isOpen" : "sourcesPanel"} aria-label="Uploaded sources">
              <div className="sourcesPanelHeader">
                <h3>Content</h3>
              </div>
              <div className="sourcesList">
                {uploadedFiles.map((file, index) => {
                  const fileName = resolveFileName(file);
                  const extension = extractExtension(fileName);
                  const statusValue = String(file?.status || "").toLowerCase();
                  return (
                    <button
                      type="button"
                      key={resolveFileId(file, index)}
                      className="sourceCard"
                      onClick={() => setSelectedSource(file)}
                    >
                      <p>{fileName}</p>
                      <span>{extension}</span>
                      <small className="statusText">Статус: {statusValue || "unknown"}</small>
                    </button>
                  );
                })}
              </div>
            </aside>
          )}
          {selectedSource && (
            <div className="summaryModalBackdrop" role="presentation" onClick={() => setSelectedSource(null)}>
              <section className="summaryModal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
                <header className="summaryModalHeader">
                  <h3>{resolveFileName(selectedSource)}</h3>
                  <button type="button" onClick={() => setSelectedSource(null)} aria-label="Close summary">
                    x
                  </button>
                </header>
                <p className="statusText">Статус: {String(selectedSource?.status || "unknown")}</p>
                {String(selectedSource?.summary || "").trim() ? (
                  <>
                    <pre className="summaryText">{String(selectedSource.summary).trim()}</pre>
                    {normalizeSummaryPoints(selectedSource).length > 0 && (
                      <ul className="summaryList">
                        {normalizeSummaryPoints(selectedSource).map((point, index) => (
                          <li key={`${resolveFileId(selectedSource)}-${index}`}>{point}</li>
                        ))}
                      </ul>
                    )}
                    <button type="button" className="messageActionBtn" onClick={() => onCopySummary(selectedSource)}>
                      {copiedSummaryId === resolveFileId(selectedSource) ? "Скопировано" : "Копировать summary"}
                    </button>
                  </>
                ) : (
                  <p className="statusText">Summary формируется. До готовности файла вопросы заблокированы.</p>
                )}
              </section>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}





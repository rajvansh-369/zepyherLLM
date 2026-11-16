import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import * as api from "./api";
import type { Status } from "./api";
import { Composer } from "./components/Composer";
import { MemoryPanel } from "./components/MemoryPanel";
import { MessageView } from "./components/MessageView";
import { Sidebar } from "./components/Sidebar";
import { BrainIcon, MenuIcon, MonitorIcon, MoonIcon, SunIcon } from "./icons";
import {
  loadConversations,
  loadPrefs,
  saveConversations,
  savePrefs,
  titleFrom,
  uid,
} from "./store";
import type { Conversation, Health, Message, Prefs } from "./types";

interface Streaming {
  convId: string;
  msgId: string;
  status: string | null;
  controller: AbortController;
}

const SUGGESTIONS = [
  "Explain how Python's asyncio event loop works, with a small example",
  "Write a TypeScript debounce function with tests",
  "What are the latest releases of Node.js?",
  "Review this idea: a CLI that summarises git diffs",
];

function describe(status: Status): string | null {
  switch (status.stage) {
    case "searching":
      return "Searching the web…";
    case "recalled":
      return `Recalled ${status.count} memor${status.count === 1 ? "y" : "ies"}`;
    case "waking":
      return "Loading the model back in…";
    default:
      return null;
  }
}

/** What the API is sent: the conversation without failed, empty replies. */
const toPayload = (messages: Message[]) =>
  messages
    .filter((m) => !(m.role === "assistant" && !m.content))
    .map((m) => ({ role: m.role, content: m.content }));

/** Drop empty replies left at the end by a failure, so turns still alternate. */
const trimFailed = (messages: Message[]) => {
  const out = [...messages];
  while (out.length && out[out.length - 1].role === "assistant" && !out[out.length - 1].content) {
    out.pop();
  }
  return out;
};

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState<Streaming | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [storageFull, setStorageFull] = useState(false);

  // The thread follows new text while the reader is at the bottom, and stops
  // following once they scroll up.
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  const active = conversations.find((c) => c.id === activeId) ?? null;

  // -- persistence ---------------------------------------------------

  useEffect(() => {
    // Streaming changes the list on every token; write at most twice a second.
    const timer = setTimeout(() => setStorageFull(!saveConversations(conversations)), 500);
    return () => clearTimeout(timer);
  }, [conversations]);

  useEffect(() => {
    savePrefs(prefs);
    const root = document.documentElement;
    if (prefs.theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", prefs.theme);
  }, [prefs]);

  // -- health --------------------------------------------------------

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      let next = 3000;
      try {
        const h = await api.getHealth();
        setHealth(h);
        setHealthError(null);
        if (h.status === "ready") next = 30000;
      } catch (e) {
        setHealth(null);
        setHealthError((e as Error).message);
      }
      if (!stopped) timer = setTimeout(poll, next);
    };

    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);

  const ready = health?.status === "ready";
  const notReadyReason = healthError
    ? "zypherLL API is not reachable"
    : health?.status === "error"
      ? "The model failed to load"
      : "The model is loading…";

  // -- state helpers -------------------------------------------------

  const updateConversation = useCallback(
    (id: string, fn: (c: Conversation) => Conversation) =>
      setConversations((list) => list.map((c) => (c.id === id ? fn(c) : c))),
    [],
  );

  const patchMessage = useCallback(
    (convId: string, msgId: string, fn: (m: Message) => Message) =>
      updateConversation(convId, (c) => ({
        ...c,
        messages: c.messages.map((m) => (m.id === msgId ? fn(m) : m)),
      })),
    [updateConversation],
  );

  // -- generation ----------------------------------------------------

  /**
   * Stream an answer into `targetId`. `messages` is what is sent; when its
   * last message is the target itself, the answer is continued in place.
   */
  const generate = useCallback(
    async (convId: string, messages: Message[], targetId: string) => {
      const controller = new AbortController();
      setStreaming({ convId, msgId: targetId, status: null, controller });
      pinned.current = true;

      try {
        const result = await api.streamChat(toPayload(messages), prefs, {
          signal: controller.signal,
          onText: (text) =>
            patchMessage(convId, targetId, (m) => ({ ...m, content: m.content + text })),
          onStatus: (status) =>
            setStreaming((s) => (s && s.msgId === targetId ? { ...s, status: describe(status) } : s)),
        });

        patchMessage(convId, targetId, (m) => ({
          ...m,
          finish: result.finish_reason,
          stopped: false,
          meta: result.meta,
          usage: result.usage,
          error: undefined,
          rating: undefined,
        }));
      } catch (error) {
        if (controller.signal.aborted) {
          // Stopped by the user: keep what arrived, and offer to continue.
          patchMessage(convId, targetId, (m) => ({ ...m, finish: "length", stopped: true }));
        } else {
          patchMessage(convId, targetId, (m) => ({ ...m, error: (error as Error).message }));
        }
      } finally {
        updateConversation(convId, (c) => ({ ...c, updated: Date.now() }));
        setStreaming((s) => (s && s.msgId === targetId ? null : s));
      }
    },
    [prefs, patchMessage, updateConversation],
  );

  const send = (text: string) => {
    if (streaming) return;

    const user: Message = { id: uid(), role: "user", content: text };
    const reply: Message = { id: uid(), role: "assistant", content: "" };
    let convId = activeId;
    let history: Message[];

    if (!active) {
      convId = uid();
      history = [user];
      const now = Date.now();
      setConversations((list) => [
        { id: convId!, title: titleFrom(text), created: now, updated: now, messages: [user, reply] },
        ...list,
      ]);
      setActiveId(convId);
    } else {
      history = [...trimFailed(active.messages), user];
      updateConversation(active.id, (c) => ({ ...c, updated: Date.now(), messages: [...history, reply] }));
      // Most recent first in the sidebar.
      setConversations((list) => [...list].sort((a, b) => b.updated - a.updated));
    }

    generate(convId!, history, reply.id);
  };

  const retry = () => {
    if (!active || streaming) return;

    const history = trimFailed(active.messages);
    const last = history[history.length - 1];
    const base = last?.role === "assistant" ? history.slice(0, -1) : history;

    if (!base.length || base[base.length - 1].role !== "user") return;

    const reply: Message = { id: uid(), role: "assistant", content: "" };
    updateConversation(active.id, (c) => ({ ...c, messages: [...base, reply] }));
    generate(active.id, base, reply.id);
  };

  const resume = () => {
    if (!active || streaming) return;

    const last = active.messages[active.messages.length - 1];
    if (!last || last.role !== "assistant" || !last.content) return;

    patchMessage(active.id, last.id, (m) => ({ ...m, finish: undefined, stopped: false }));
    generate(active.id, active.messages, last.id);
  };

  const edit = (msgId: string, text: string) => {
    if (!active || streaming) return;

    const index = active.messages.findIndex((m) => m.id === msgId);
    if (index < 0) return;

    const history = [...active.messages.slice(0, index), { ...active.messages[index], content: text }];
    const reply: Message = { id: uid(), role: "assistant", content: "" };

    updateConversation(active.id, (c) => ({
      ...c,
      updated: Date.now(),
      title: index === 0 ? titleFrom(text) : c.title,
      messages: [...history, reply],
    }));
    generate(active.id, history, reply.id);
  };

  const rate = async (msg: Message, rating: "good" | "bad") => {
    if (!active || msg.meta?.memory_id == null) return;

    const convId = active.id;
    try {
      await api.rate(msg.meta.memory_id, rating);
      patchMessage(convId, msg.id, (m) => ({ ...m, rating }));
    } catch (e) {
      patchMessage(convId, msg.id, (m) => ({ ...m, error: (e as Error).message }));
    }
  };

  const stop = () => streaming?.controller.abort();

  // -- conversations -------------------------------------------------

  const newChat = () => {
    setActiveId(null);
    setSidebarOpen(false);
  };

  const select = (id: string) => {
    setActiveId(id);
    setSidebarOpen(false);
  };

  const rename = (id: string, title: string) =>
    title.trim() && updateConversation(id, (c) => ({ ...c, title: title.trim() }));

  const remove = (id: string) => {
    setConversations((list) => list.filter((c) => c.id !== id));
    if (id === activeId) setActiveId(null);
  };

  // Ctrl/Cmd+Shift+O: new chat, as in other chat apps. Esc: stop.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        newChat();
      }
      if (e.key === "Escape" && streaming && !memoryOpen) streaming.controller.abort();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [streaming, memoryOpen]);

  // -- scrolling -----------------------------------------------------

  const onScroll = () => {
    const el = scroller.current;
    if (el) pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  // Follow the content's size, not the message list: markdown is rendered a
  // frame after the text changes, so scrolling on a state change lands one
  // render short of the bottom.
  useEffect(() => {
    const el = scroller.current;
    const content = el?.firstElementChild;
    if (!el || !content) return;

    const observer = new ResizeObserver(() => {
      if (pinned.current) el.scrollTop = el.scrollHeight;
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, [activeId, active !== null && active.messages.length > 0]);

  useLayoutEffect(() => {
    pinned.current = true;
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [activeId]);

  // -- render --------------------------------------------------------

  const busy = streaming !== null;
  const themeNext = { system: "light", light: "dark", dark: "system" } as const;
  const ThemeIcon = prefs.theme === "light" ? SunIcon : prefs.theme === "dark" ? MoonIcon : MonitorIcon;

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={sidebarOpen}
        streamingId={streaming?.convId ?? null}
        onNew={newChat}
        onSelect={select}
        onRename={rename}
        onDelete={remove}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="main">
        <header className="topbar">
          <button className="icon-btn menu" title="Chats" onClick={() => setSidebarOpen(true)}>
            <MenuIcon />
          </button>

          <div className="model" title={healthTitle(health, healthError)}>
            <span className={"dot " + (healthError ? "error" : (health?.status ?? "loading"))} />
            <span className="model-name">zephyr-7b</span>
            {health?.gpu && <span className="model-sub">{health.gpu.replace(/^NVIDIA GeForce /, "")}</span>}
          </div>

          <span className="spacer" />

          <button className="icon-btn" title="Memory & settings" onClick={() => setMemoryOpen(true)}>
            <BrainIcon />
          </button>
          <button
            className="icon-btn"
            title={`Theme: ${prefs.theme}`}
            onClick={() => setPrefs({ ...prefs, theme: themeNext[prefs.theme] })}
          >
            <ThemeIcon />
          </button>
        </header>

        {!ready && (
          <div className={"banner " + (healthError || health?.status === "error" ? "bad" : "")}>
            {healthError
              ? healthError
              : health?.status === "error"
                ? `The model failed to load: ${health.error}`
                : "The model is loading. The first load after a restart can take a minute or two."}
          </div>
        )}

        {storageFull && (
          <div className="banner bad">
            Browser storage is full -- new messages are not being saved. Delete some old chats.
          </div>
        )}

        <div className="scroller" ref={scroller} onScroll={onScroll}>
          {active && active.messages.length > 0 ? (
            <div className="thread">
              {active.messages.map((m, i) => (
                <MessageView
                  key={m.id}
                  message={m}
                  streaming={streaming?.msgId === m.id}
                  status={streaming?.msgId === m.id ? streaming.status : null}
                  isLast={i === active.messages.length - 1}
                  busy={busy}
                  onRetry={retry}
                  onContinue={resume}
                  onEdit={(text) => edit(m.id, text)}
                  onRate={(rating) => rate(m, rating)}
                />
              ))}
            </div>
          ) : (
            <div className="welcome">
              <h1>What can I help with?</h1>
              <p>Local Zephyr 7B with web lookup and memory. Nothing leaves your machine except search queries.</p>
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} disabled={!ready || busy} onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <Composer
          busy={busy}
          disabled={!ready}
          disabledReason={notReadyReason}
          prefs={prefs}
          focusKey={activeId}
          onPrefs={setPrefs}
          onSend={send}
          onStop={stop}
        />
      </main>

      {memoryOpen && <MemoryPanel onClose={() => setMemoryOpen(false)} />}
    </div>
  );
}

function healthTitle(health: Health | null, error: string | null): string {
  if (error) return error;
  if (!health) return "Checking…";
  const parts = [`Status: ${health.status}`, `Engine: ${health.backend}`];
  if (health.gpu) parts.push(`GPU: ${health.gpu} (${health.vram_gb} GB)`);
  if (health.context) parts.push(`Context: ${health.context} tokens`);
  if (health.gpu_layers) parts.push(`GPU layers: ${health.gpu_layers}`);
  if (health.kv_cache) parts.push(`KV cache: ${health.kv_cache}`);
  return parts.join("\n");
}

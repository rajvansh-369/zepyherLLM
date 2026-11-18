import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  ArrowDown,
  ChevronDown,
  CircleAlert,
  CodeXml,
  Globe,
  Lightbulb,
  LoaderCircle,
  PanelLeft,
  Sparkles,
  SquarePen,
  WifiOff,
} from "lucide-react";
import * as api from "./api";
import type { Status } from "./api";
import { Logo } from "./brand";
import { Composer } from "./components/Composer";
import { MessageView } from "./components/MessageView";
import { SettingsDialog, type SettingsTab } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import {
  loadConversations,
  loadPrefs,
  saveConversations,
  savePrefs,
  titleFrom,
  uid,
} from "./store";
import { applyTheme, watchSystemTheme, withTransition } from "./theme";
import { Toasts, toast } from "./toast";
import type { Conversation, Health, Message, Prefs, StatusInfo } from "./types";

interface Streaming {
  convId: string;
  msgId: string;
  status: StatusInfo | null;
  controller: AbortController;
}

const NARROW = "(max-width: 820px)";

const SUGGESTIONS = [
  {
    icon: CodeXml,
    title: "Write code",
    text: "A TypeScript debounce function, with tests",
    prompt: "Write a TypeScript debounce function, with tests.",
  },
  {
    icon: Lightbulb,
    title: "Explain a concept",
    text: "How Python's asyncio event loop works",
    prompt: "Explain how Python's asyncio event loop works, with a small example.",
  },
  {
    icon: Globe,
    title: "Look something up",
    text: "The latest releases of Node.js",
    prompt: "What are the latest releases of Node.js?",
  },
  {
    icon: Sparkles,
    title: "Review an idea",
    text: "A CLI that summarises git diffs",
    prompt: "Review this idea: a CLI that summarises git diffs. What would make it useful?",
  },
];

function describe(status: Status): StatusInfo | null {
  switch (status.stage) {
    case "searching":
      return { stage: status.stage, text: "Searching the web…" };
    case "recalled":
      return {
        stage: status.stage,
        text: `Recalled ${status.count} memor${status.count === 1 ? "y" : "ies"}`,
      };
    case "waking":
      return { stage: status.stage, text: "Loading the model back in…" };
    case "switching":
      return { stage: status.stage, text: `Loading ${status.model ?? "your model"}…` };
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

/** A callback that always runs the latest version of `fn` but never changes
 *  identity, so memoized children are not re-rendered for it. */
function useEvent<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R {
  const ref = useRef(fn);
  useLayoutEffect(() => {
    ref.current = fn;
  });
  return useCallback((...args: A) => ref.current(...args), []);
}

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [query]);

  return matches;
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState<string | null>(null);
  const [streaming, setStreaming] = useState<Streaming | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [settings, setSettings] = useState<SettingsTab | null>(null);
  const [storageFull, setStorageFull] = useState(false);
  const [draft, setDraft] = useState("");
  const [atBottom, setAtBottom] = useState(true);
  const [announce, setAnnounce] = useState("");

  const narrow = useMediaQuery(NARROW);
  const searchRef = useRef<HTMLInputElement>(null);

  // The thread follows new text while the reader is at the bottom, and stops
  // following once they scroll up.
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const lastTop = useRef(0);

  const active = conversations.find((c) => c.id === activeId) ?? null;
  const empty = !active || active.messages.length === 0;
  const busy = streaming !== null;

  // -- persistence ---------------------------------------------------

  useEffect(() => {
    // Streaming changes the list on every token; write at most twice a second.
    const timer = setTimeout(() => setStorageFull(!saveConversations(conversations)), 500);
    return () => clearTimeout(timer);
  }, [conversations]);

  useEffect(() => {
    savePrefs(prefs);
  }, [prefs]);

  useEffect(() => {
    applyTheme(prefs.theme);
    if (prefs.theme !== "system") return;
    return watchSystemTheme(() => applyTheme("system"));
  }, [prefs.theme]);

  const setTheme = (theme: Prefs["theme"]) =>
    withTransition(() => {
      // Applied here as well as in the effect, so it is inside the transition.
      applyTheme(theme);
      setPrefs((p) => ({ ...p, theme }));
    });

  // -- health --------------------------------------------------------

  // Checks health at once, e.g. right after the model was switched.
  const pollNow = useRef<() => void>(() => {});

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      clearTimeout(timer);
      let next = 3000;
      try {
        const h = await api.getHealth();
        setHealth(h);
        setOffline(null);
        if (h.status === "ready") next = 30000;
      } catch (e) {
        setHealth(null);
        setOffline((e as Error).message);
      }
      if (!stopped) timer = setTimeout(poll, next);
    };

    pollNow.current = poll;
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);

  // The look follows the model: Zypher (the general model) gets the red,
  // hard-edged theme, any other model the soft one. While a switch loads,
  // it already shows the model being loaded.
  const slot = health?.switching ?? health?.models?.find((m) => m.active)?.slot ?? null;

  useEffect(() => {
    const root = document.documentElement;
    if (slot === "general") root.dataset.model = "zypher";
    else if (slot) root.dataset.model = "soft";
    else delete root.dataset.model;
  }, [slot]);

  const ready = health?.status === "ready";
  const notReadyReason = offline
    ? "The zypherLL API is not reachable"
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
  const generate = async (convId: string, messages: Message[], targetId: string) => {
    const controller = new AbortController();
    setStreaming({ convId, msgId: targetId, status: null, controller });
    setAnnounce("Writing an answer");
    pinned.current = true;
    setAtBottom(true);

    try {
      const result = await api.streamChat(toPayload(messages), prefs, {
        signal: controller.signal,
        onText: (text) =>
          patchMessage(convId, targetId, (m) => ({ ...m, content: m.content + text })),
        onStatus: (status) => {
          const info = describe(status);
          if (!info) return;
          setAnnounce(info.text);
          setStreaming((s) => (s && s.msgId === targetId ? { ...s, status: info } : s));
        },
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
      setAnnounce("Answer complete");
    } catch (error) {
      if (controller.signal.aborted) {
        // Stopped by the reader: keep what arrived, and offer to continue.
        patchMessage(convId, targetId, (m) => ({ ...m, finish: "length", stopped: true }));
        setAnnounce("Stopped");
      } else {
        patchMessage(convId, targetId, (m) => ({ ...m, error: (error as Error).message }));
        setAnnounce("The answer failed");
      }
    } finally {
      updateConversation(convId, (c) => ({ ...c, updated: Date.now() }));
      setStreaming((s) => (s && s.msgId === targetId ? null : s));
    }
  };

  const send = (text: string) => {
    if (busy || !ready) return;

    const user: Message = { id: uid(), role: "user", content: text };
    const reply: Message = { id: uid(), role: "assistant", content: "" };

    if (empty) {
      const id = active?.id ?? uid();
      const now = Date.now();
      const conversation: Conversation = {
        id,
        title: titleFrom(text),
        created: now,
        updated: now,
        messages: [user, reply],
      };

      // The page turns from the welcome screen into a thread; the composer
      // glides from the middle of the page down to its dock.
      withTransition(() => {
        setConversations((list) => [conversation, ...list.filter((c) => c.id !== id)]);
        setActiveId(id);
        generate(id, [user], reply.id);
      });
      return;
    }

    const history = [...trimFailed(active.messages), user];

    setConversations((list) =>
      list
        .map((c) =>
          c.id === active.id ? { ...c, updated: Date.now(), messages: [...history, reply] } : c,
        )
        // Most recent first in the sidebar.
        .sort((a, b) => b.updated - a.updated),
    );
    generate(active.id, history, reply.id);
  };

  const retry = useEvent(() => {
    if (!active || busy) return;

    const history = trimFailed(active.messages);
    const last = history[history.length - 1];
    const base = last?.role === "assistant" ? history.slice(0, -1) : history;

    if (!base.length || base[base.length - 1].role !== "user") return;

    const reply: Message = { id: uid(), role: "assistant", content: "" };
    updateConversation(active.id, (c) => ({ ...c, messages: [...base, reply] }));
    generate(active.id, base, reply.id);
  });

  const resume = useEvent(() => {
    if (!active || busy) return;

    const last = active.messages[active.messages.length - 1];
    if (!last || last.role !== "assistant" || !last.content) return;

    patchMessage(active.id, last.id, (m) => ({ ...m, finish: undefined, stopped: false }));
    generate(active.id, active.messages, last.id);
  });

  const edit = useEvent((msgId: string, text: string) => {
    if (!active || busy) return;

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
  });

  const rate = useEvent(async (msgId: string, rating: "good" | "bad") => {
    const message = active?.messages.find((m) => m.id === msgId);
    if (!active || message?.meta?.memory_id == null) return;

    const convId = active.id;

    try {
      await api.rate(message.meta.memory_id, rating);
      patchMessage(convId, msgId, (m) => ({ ...m, rating }));
      toast(
        rating === "good"
          ? "Marked good: it will be recalled more often."
          : "Marked bad: it will not be recalled again.",
      );
    } catch (e) {
      toast.error((e as Error).message);
    }
  });

  const stop = () => streaming?.controller.abort();

  // -- conversations and sidebar -------------------------------------

  const newChat = () => {
    setDrawerOpen(false);
    if (activeId === null) {
      document.getElementById("composer-input")?.focus();
      return;
    }
    withTransition(() => setActiveId(null));
  };

  const select = (id: string) => {
    setActiveId(id);
    setDrawerOpen(false);
  };

  const rename = (id: string, title: string) =>
    title.trim() && updateConversation(id, (c) => ({ ...c, title: title.trim() }));

  const remove = (id: string) => {
    setConversations((list) => list.filter((c) => c.id !== id));
    if (id === activeId) setActiveId(null);
    toast("Chat deleted.");
  };

  const sidebarHidden = narrow ? !drawerOpen : prefs.sidebarCollapsed;

  const showSidebar = (show: boolean) => {
    if (narrow) setDrawerOpen(show);
    else setPrefs((p) => ({ ...p, sidebarCollapsed: !show }));
  };

  // Shortcuts. Ctrl/Cmd+Shift+O new chat, Ctrl/Cmd+K search, Esc stop,
  // "/" to start typing from anywhere.
  const onKey = useEvent((e: KeyboardEvent) => {
    const mod = e.ctrlKey || e.metaKey;
    const key = e.key.toLowerCase();

    if (mod && e.shiftKey && key === "o") {
      e.preventDefault();
      newChat();
    } else if (mod && key === "k") {
      e.preventDefault();
      // The sidebar has to be on screen (and no longer inert) to take focus.
      flushSync(() => showSidebar(true));
      searchRef.current?.focus();
    } else if (e.key === "Escape" && streaming && !settings) {
      streaming.controller.abort();
    } else if (e.key === "/" && !mod && !isTyping(e.target)) {
      e.preventDefault();
      document.getElementById("composer-input")?.focus();
    }
  });

  useEffect(() => {
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onKey]);

  // -- scrolling -----------------------------------------------------

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;

    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    const movedUp = el.scrollTop < lastTop.current - 2;
    lastTop.current = el.scrollTop;

    // Only the reader scrolling up lets go of the bottom. Being "far from
    // the bottom" is not enough: a burst of new text makes that true for a
    // moment, before the view has had its chance to follow it.
    if (near) pinned.current = true;
    else if (movedUp) pinned.current = false;

    setAtBottom(near || pinned.current);
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
  }, [activeId, empty]);

  useLayoutEffect(() => {
    pinned.current = true;
    setAtBottom(true);
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [activeId, empty]);

  const jumpToBottom = () => {
    const el = scroller.current;
    if (!el) return;
    pinned.current = true;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  // -- render --------------------------------------------------------

  const status = offline ? "error" : (health?.status ?? "loading");
  const gpu = health?.gpu?.replace(/^NVIDIA GeForce /, "").replace(/ Laptop GPU$/, "");
  const modelName = health?.model ?? health?.models?.find((m) => m.active)?.id ?? "zypherLL";
  const modelSub = offline
    ? "Offline"
    : health?.status === "ready"
      ? [gpu, health.context && `${Math.round(health.context / 1024)}k context`]
          .filter(Boolean)
          .join(" · ")
      : health?.status === "error"
        ? "Failed to load"
        : "Loading…";

  const dock = (
    <div className="dock">
      {!empty && !atBottom && (
        <button
          className="jump"
          title="Jump to latest"
          aria-label="Jump to latest message"
          onClick={jumpToBottom}
        >
          <ArrowDown size={16} />
          {busy && <span className="dot live" />}
        </button>
      )}

      <Composer
        text={draft}
        busy={busy}
        disabled={!ready}
        disabledReason={notReadyReason}
        prefs={prefs}
        focusKey={activeId}
        onText={setDraft}
        onPrefs={setPrefs}
        onSend={send}
        onStop={stop}
      />

      <p className="hint">
        <kbd>Enter</kbd> to send · <kbd>Shift</kbd> <kbd>Enter</kbd> for a new line · Runs
        entirely on your machine
      </p>
    </div>
  );

  return (
    <div className={"app" + (prefs.sidebarCollapsed ? " side-collapsed" : "")}>
      <a className="skip-link" href="#composer-input">
        Skip to message box
      </a>

      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={drawerOpen}
        hidden={sidebarHidden}
        streamingId={streaming?.convId ?? null}
        theme={prefs.theme}
        searchRef={searchRef}
        onNew={newChat}
        onSelect={select}
        onRename={rename}
        onDelete={remove}
        onClose={() => showSidebar(false)}
        onTheme={setTheme}
        onSettings={() => setSettings("about")}
      />

      <main className={"main" + (empty ? " is-empty" : "")}>
        <header className="topbar">
          <button
            className="icon-btn side-toggle"
            title="Show sidebar"
            aria-label="Show sidebar"
            onClick={() => showSidebar(true)}
          >
            <PanelLeft size={18} />
          </button>
          <button
            className="icon-btn top-new"
            title="New chat (Ctrl+Shift+O)"
            aria-label="New chat"
            onClick={newChat}
          >
            <SquarePen size={18} />
          </button>

          <button
            className="model-pill"
            title="Engine details"
            aria-label={`${modelName}, ${modelSub}. Open engine details`}
            onClick={() => setSettings("server")}
          >
            <span className={"dot " + status} />
            <span className="model-name">{modelName}</span>
            {modelSub && <span className="model-sub">{modelSub}</span>}
            <ChevronDown size={14} />
          </button>
        </header>

        <div className="notices">
          {offline ? (
            <div className="banner bad" role="alert">
              <WifiOff size={16} />
              {offline.includes("main.py serve") ? (
                <span>
                  The zypherLL API is not running. Start it with <code>python main.py serve</code>
                </span>
              ) : (
                <span>{offline}</span>
              )}
            </div>
          ) : health?.status === "error" ? (
            <div className="banner bad" role="alert">
              <CircleAlert size={16} />
              <span>The model failed to load: {health.error}</span>
            </div>
          ) : !ready ? (
            <div className="banner" role="status">
              <LoaderCircle size={16} className="spin" />
              <span>
                {health?.switching
                  ? `Switching to ${health.models?.find((m) => m.slot === health.switching)?.id ?? "the other model"}. This takes up to a minute.`
                  : "Loading the model. The first load after a restart can take a minute or two."}
              </span>
            </div>
          ) : null}

          {storageFull && (
            <div className="banner bad" role="alert">
              <CircleAlert size={16} />
              <span>
                Browser storage is full, so new messages are not being saved. Delete some old chats.
              </span>
            </div>
          )}
        </div>

        {empty ? (
          <div className="stage">
            <div className="stage-inner">
              <div className="welcome">
                <Logo size={56} />
                <h1>
                  What can I <span className="grad">help with?</span>
                </h1>
                <p>
                  {modelName} running on your own GPU, with live web lookup and memory. Nothing
                  leaves your machine except search queries.
                </p>
              </div>

              {dock}

              <div className="suggestions">
                {SUGGESTIONS.map(({ icon: Icon, title, text, prompt }, i) => (
                  <button
                    key={title}
                    className="suggestion"
                    style={{ "--i": i } as React.CSSProperties}
                    disabled={!ready || busy}
                    onClick={() => send(prompt)}
                  >
                    <span className="ico">
                      <Icon size={17} />
                    </span>
                    <span>
                      <strong>{title}</strong>
                      <small>{text}</small>
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="scroller" ref={scroller} onScroll={onScroll}>
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
                    onEdit={edit}
                    onRate={rate}
                  />
                ))}
              </div>
            </div>
            {dock}
          </>
        )}
      </main>

      {settings && (
        <SettingsDialog
          initialTab={settings}
          health={health}
          offline={offline}
          onModelChange={() => pollNow.current()}
          onClose={() => setSettings(null)}
        />
      )}

      <Toasts />
      <div className="sr-only" aria-live="polite">
        {announce}
      </div>
    </div>
  );
}

function isTyping(target: EventTarget | null) {
  const el = target as HTMLElement | null;
  return !!el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
}

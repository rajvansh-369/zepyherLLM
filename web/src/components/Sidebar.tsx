import { useState } from "react";
import {
  MessageSquare,
  Monitor,
  Moon,
  PanelLeftClose,
  Pencil,
  Search,
  SlidersHorizontal,
  SquarePen,
  Sun,
  Trash2,
} from "lucide-react";
import { Logo } from "../brand";
import type { Conversation, Prefs } from "../types";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  open: boolean;
  hidden: boolean;
  streamingId: string | null;
  theme: Prefs["theme"];
  searchRef: React.RefObject<HTMLInputElement | null>;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
  onTheme: (theme: Prefs["theme"]) => void;
  onSettings: () => void;
}

const DAY = 86_400_000;

const THEMES = [
  { value: "system", label: "Match system", icon: Monitor },
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
] as const;

function group(updated: number): string {
  const today = new Date().setHours(0, 0, 0, 0);
  if (updated >= today) return "Today";
  if (updated >= today - DAY) return "Yesterday";
  if (updated >= today - 7 * DAY) return "Previous 7 days";
  if (updated >= today - 30 * DAY) return "Previous 30 days";
  return "Older";
}

export function Sidebar({
  conversations,
  activeId,
  open,
  hidden,
  streamingId,
  theme,
  searchRef,
  onNew,
  onSelect,
  onRename,
  onDelete,
  onClose,
  onTheme,
  onSettings,
}: Props) {
  const [query, setQuery] = useState("");
  const [renaming, setRenaming] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  const q = query.trim().toLowerCase();
  const shown = q
    ? conversations.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          c.messages.some((m) => m.content.toLowerCase().includes(q)),
      )
    : conversations;

  const groups: [string, Conversation[]][] = [];

  for (const conversation of shown) {
    const name = group(conversation.updated);
    const last = groups[groups.length - 1];
    if (last && last[0] === name) last[1].push(conversation);
    else groups.push([name, [conversation]]);
  }

  return (
    <>
      <div className={"scrim" + (open ? " show" : "")} onClick={onClose} />

      {/* inert while out of sight, so Tab never lands on something invisible */}
      <aside className={"sidebar" + (open ? " open" : "")} aria-label="Chats" inert={hidden}>
        <div className="side-head">
          <Logo size={28} />
          <span className="wordmark">
            zypher<b>LL</b>
          </span>
          <button
            className="icon-btn"
            title="Hide sidebar"
            aria-label="Hide sidebar"
            onClick={onClose}
          >
            <PanelLeftClose size={18} />
          </button>
        </div>

        <button className="new-chat" onClick={onNew}>
          <SquarePen size={16} />
          New chat
          <span className="keys" aria-hidden="true">
            <kbd>Ctrl</kbd>
            <kbd>⇧</kbd>
            <kbd>O</kbd>
          </span>
        </button>

        <label className="search">
          <Search size={15} />
          <input
            ref={searchRef}
            type="search"
            placeholder="Search chats"
            aria-label="Search chats"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape" && query) {
                e.stopPropagation();
                setQuery("");
              }
            }}
          />
          <span className="keys" aria-hidden="true">
            <kbd>Ctrl</kbd>
            <kbd>K</kbd>
          </span>
        </label>

        <nav className="conversations" aria-label="Chat history">
          {groups.length === 0 && (
            <div className="empty">
              <MessageSquare size={22} />
              {q ? "No chats match that search." : "Your chats will appear here."}
            </div>
          )}

          {groups.map(([name, items]) => (
            <div key={name}>
              <div className="group">{name}</div>

              {items.map((c) => {
                const active = c.id === activeId;
                const asking = confirming === c.id;

                return (
                  <div
                    key={c.id}
                    className={
                      "conv" + (active ? " active" : "") + (asking ? " confirming" : "")
                    }
                  >
                    {renaming === c.id ? (
                      <input
                        className="rename"
                        autoFocus
                        aria-label="Chat title"
                        defaultValue={c.title}
                        onFocus={(e) => e.target.select()}
                        onBlur={(e) => {
                          onRename(c.id, e.target.value);
                          setRenaming(null);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                          if (e.key === "Escape") {
                            e.stopPropagation();
                            setRenaming(null);
                          }
                        }}
                      />
                    ) : (
                      <button
                        className="conv-main"
                        title={c.title}
                        aria-current={active ? "true" : undefined}
                        onClick={() => onSelect(c.id)}
                        onDoubleClick={() => setRenaming(c.id)}
                      >
                        {c.id === streamingId && <span className="dot live" />}
                        <span className="conv-title">{c.title}</span>
                      </button>
                    )}

                    {renaming === c.id ? null : asking ? (
                      <span className="conv-actions show">
                        <button className="mini danger" autoFocus onClick={() => onDelete(c.id)}>
                          Delete
                        </button>
                        <button className="mini" onClick={() => setConfirming(null)}>
                          Cancel
                        </button>
                      </span>
                    ) : (
                      <span className="conv-actions">
                        <button
                          className="icon-btn xs"
                          title="Rename"
                          aria-label={`Rename ${c.title}`}
                          onClick={() => setRenaming(c.id)}
                        >
                          <Pencil size={14} />
                        </button>
                        <button
                          className="icon-btn xs"
                          title="Delete"
                          aria-label={`Delete ${c.title}`}
                          disabled={c.id === streamingId}
                          onClick={() => setConfirming(c.id)}
                        >
                          <Trash2 size={14} />
                        </button>
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="side-foot">
          <button className="foot-btn" onClick={onSettings}>
            <SlidersHorizontal size={16} />
            Memory &amp; settings
          </button>

          <div className="theme-switch" role="radiogroup" aria-label="Theme">
            {THEMES.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                role="radio"
                aria-checked={theme === value}
                aria-label={label}
                title={label}
                onClick={() => onTheme(value)}
              >
                <Icon size={14} />
              </button>
            ))}
          </div>
        </div>
      </aside>
    </>
  );
}

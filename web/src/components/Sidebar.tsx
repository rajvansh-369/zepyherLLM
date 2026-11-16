import { useState } from "react";
import type { Conversation } from "../types";
import { EditIcon, PlusIcon, TrashIcon } from "../icons";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  open: boolean;
  streamingId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

const DAY = 86_400_000;

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
  streamingId,
  onNew,
  onSelect,
  onRename,
  onDelete,
  onClose,
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
      <aside className={"sidebar" + (open ? " open" : "")}>
        <div className="brand">
          <span className="logo">Z</span> zypherLL
        </div>

        <button className="new-chat" onClick={onNew}>
          <PlusIcon size={16} /> New chat
        </button>

        <input
          className="search"
          type="search"
          placeholder="Search chats"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <nav className="conversations">
          {groups.length === 0 && (
            <div className="empty-list">{q ? "No chats match." : "No chats yet."}</div>
          )}

          {groups.map(([name, items]) => (
            <div key={name}>
              <div className="group">{name}</div>
              {items.map((c) => (
                <div
                  key={c.id}
                  className={"conv" + (c.id === activeId ? " active" : "")}
                  onClick={() => renaming !== c.id && onSelect(c.id)}
                >
                  {renaming === c.id ? (
                    <input
                      className="rename"
                      autoFocus
                      defaultValue={c.title}
                      onClick={(e) => e.stopPropagation()}
                      onBlur={(e) => {
                        onRename(c.id, e.target.value);
                        setRenaming(null);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                        if (e.key === "Escape") setRenaming(null);
                      }}
                    />
                  ) : (
                    <span className="conv-title">
                      {c.id === streamingId && <span className="dot live" />}
                      {c.title}
                    </span>
                  )}

                  {confirming === c.id ? (
                    <span className="conv-actions show" onClick={(e) => e.stopPropagation()}>
                      <button className="mini danger" onClick={() => onDelete(c.id)}>
                        Delete
                      </button>
                      <button className="mini" onClick={() => setConfirming(null)}>
                        Keep
                      </button>
                    </span>
                  ) : (
                    <span className="conv-actions" onClick={(e) => e.stopPropagation()}>
                      <button className="icon-btn" title="Rename" onClick={() => setRenaming(c.id)}>
                        <EditIcon size={14} />
                      </button>
                      <button
                        className="icon-btn"
                        title="Delete"
                        disabled={c.id === streamingId}
                        onClick={() => setConfirming(c.id)}
                      >
                        <TrashIcon size={14} />
                      </button>
                    </span>
                  )}
                </div>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-foot">Chats are saved in this browser only.</div>
      </aside>
    </>
  );
}

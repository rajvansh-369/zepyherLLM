import { useState } from "react";
import type { Message } from "../types";
import { ContinueIcon, CopyIcon, DownIcon, EditIcon, RetryIcon, UpIcon } from "../icons";
import { Markdown, copyText } from "./Markdown";

interface Props {
  message: Message;
  streaming: boolean;
  status: string | null;
  isLast: boolean;
  busy: boolean;
  onRetry: () => void;
  onContinue: () => void;
  onEdit: (text: string) => void;
  onRate: (rating: "good" | "bad") => void;
}

export function MessageView(props: Props) {
  return props.message.role === "user" ? <UserMessage {...props} /> : <AssistantMessage {...props} />;
}

function UserMessage({ message, busy, onEdit }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);

  if (editing) {
    const save = () => {
      if (!draft.trim()) return;
      setEditing(false);
      onEdit(draft.trim());
    };

    return (
      <div className="msg user">
        <div className="edit-box">
          <textarea
            value={draft}
            autoFocus
            rows={Math.min(12, draft.split("\n").length + 1)}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                save();
              }
              if (e.key === "Escape") setEditing(false);
            }}
          />
          <div className="edit-actions">
            <button className="btn ghost" onClick={() => setEditing(false)}>
              Cancel
            </button>
            <button className="btn primary" onClick={save} disabled={busy || !draft.trim()}>
              Send
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="msg user">
      <div className="bubble">{message.content}</div>
      <div className="actions hover">
        <CopyButton text={message.content} />
        <button
          className="icon-btn"
          title="Edit and resend"
          disabled={busy}
          onClick={() => {
            setDraft(message.content);
            setEditing(true);
          }}
        >
          <EditIcon size={15} />
        </button>
      </div>
    </div>
  );
}

function AssistantMessage({
  message,
  streaming,
  status,
  isLast,
  busy,
  onRetry,
  onContinue,
  onRate,
}: Props) {
  const { meta, usage } = message;
  const empty = !message.content;

  return (
    <div className="msg assistant">
      {streaming && status && <div className="status">{status}</div>}

      {streaming && empty && !status && (
        <div className="typing" aria-label="Thinking">
          <span />
          <span />
          <span />
        </div>
      )}

      {!empty && <Markdown text={message.content} streaming={streaming} />}

      {message.error && (
        <div className="error-box">
          <strong>Something went wrong.</strong> {message.error}
        </div>
      )}

      {!streaming && meta && meta.sources.length > 0 && (
        <div className="sources">
          <div className="sources-title">{meta.cited ? "Sources" : "Searched"}</div>
          <ol>
            {meta.sources.map((source) => (
              <li key={source.n} value={source.n}>
                <a href={source.url} target="_blank" rel="noopener noreferrer">
                  {source.title || source.url}
                </a>
                <span className="host">{host(source.url)}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {!streaming && meta?.notes_captured.map((note) => (
        <div key={note} className="note-chip">
          Remembered: {note}
        </div>
      ))}

      {!streaming && meta?.throttled && (
        <div className="warn-box">
          The GPU is at {meta.throttled.gpu_temp_c}°C and has throttled itself: this ran at{" "}
          {meta.tokens_per_second.toFixed(0)} tok/s against{" "}
          {meta.throttled.usual_tokens_per_second.toFixed(0)} when cool. Give the laptop a minute
          or more airflow.
        </div>
      )}

      {!streaming && (
        <div className={"actions" + (isLast ? "" : " hover")}>
          {!empty && <CopyButton text={message.content} />}

          {isLast && (
            <button className="icon-btn" title="Regenerate" disabled={busy} onClick={onRetry}>
              <RetryIcon size={15} />
            </button>
          )}

          {meta?.memory_id != null && (
            <>
              <button
                className={"icon-btn" + (message.rating === "good" ? " active" : "")}
                title="Good answer -- recall it more"
                onClick={() => onRate("good")}
              >
                <UpIcon size={15} />
              </button>
              <button
                className={"icon-btn" + (message.rating === "bad" ? " active bad" : "")}
                title="Bad answer -- never recall it"
                onClick={() => onRate("bad")}
              >
                <DownIcon size={15} />
              </button>
            </>
          )}

          {isLast && message.finish === "length" && !message.error && (
            <button className="btn small" disabled={busy} onClick={onContinue}>
              <ContinueIcon size={14} /> Continue
            </button>
          )}

          {usage && usage.completion_tokens > 0 && meta && (
            <span className="stats">
              {usage.completion_tokens} tokens · {meta.tokens_per_second.toFixed(1)} tok/s
              {usage.prompt_tokens_details?.cached_tokens
                ? ` · ${usage.prompt_tokens_details.cached_tokens}/${usage.prompt_tokens} cached`
                : ""}
              {meta.recalled ? ` · ${meta.recalled} recalled` : ""}
              {message.stopped ? " · stopped" : ""}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);

  return (
    <button
      className="icon-btn"
      title={done ? "Copied" : "Copy"}
      onClick={() =>
        copyText(text).then((ok) => {
          if (!ok) return;
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        })
      }
    >
      {done ? <span className="tick">✓</span> : <CopyIcon size={15} />}
    </button>
  );
}

function host(url: string) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

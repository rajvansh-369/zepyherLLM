import { memo, useState } from "react";
import {
  ArrowRight,
  Bookmark,
  Check,
  CircleAlert,
  Copy,
  Globe,
  History,
  Link2,
  LoaderCircle,
  Pencil,
  RefreshCw,
  Thermometer,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import type { Message, StatusInfo } from "../types";
import { Markdown, copyText } from "./Markdown";

interface Props {
  message: Message;
  streaming: boolean;
  status: StatusInfo | null;
  isLast: boolean;
  busy: boolean;
  onRetry: () => void;
  onContinue: () => void;
  onEdit: (id: string, text: string) => void;
  onRate: (id: string, rating: "good" | "bad") => void;
}

/**
 * One turn. Memoized: while an answer streams, only the message being
 * written re-renders, however long the conversation above it is.
 */
export const MessageView = memo(function MessageView(props: Props) {
  return props.message.role === "user" ? <UserMessage {...props} /> : <AssistantMessage {...props} />;
});

function UserMessage({ message, busy, onEdit }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);

  if (editing) {
    const save = () => {
      if (!draft.trim() || busy) return;
      setEditing(false);
      onEdit(message.id, draft.trim());
    };

    return (
      <div className="msg user">
        <div className="edit-box">
          <textarea
            value={draft}
            autoFocus
            aria-label="Edit message"
            rows={Math.min(12, draft.split("\n").length + 1)}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                save();
              }
              if (e.key === "Escape") {
                e.stopPropagation();
                setEditing(false);
              }
            }}
          />
          <div className="edit-actions">
            <button className="btn ghost sm" onClick={() => setEditing(false)}>
              Cancel
            </button>
            <button className="btn primary sm" onClick={save} disabled={busy || !draft.trim()}>
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
      <div className="actions quiet">
        <CopyButton text={message.content} />
        <button
          className="icon-btn"
          title="Edit and resend"
          aria-label="Edit and resend"
          disabled={busy}
          onClick={() => {
            setDraft(message.content);
            setEditing(true);
          }}
        >
          <Pencil size={15} />
        </button>
      </div>
    </div>
  );
}

function StatusIcon({ stage }: { stage: string }) {
  if (stage === "searching") return <Globe size={14} />;
  if (stage === "recalled") return <History size={14} />;
  return <LoaderCircle size={14} className="spin" />;
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
    <article className="msg assistant" aria-busy={streaming}>
      <header className="msg-head">
        <span className={"orb" + (streaming ? " live" : "")} aria-hidden="true" />
        <span className="msg-name">zypherLL</span>
        {streaming && status && (
          <span className="status">
            <StatusIcon stage={status.stage} />
            <span className="shimmer">{status.text}</span>
          </span>
        )}
      </header>

      {streaming && empty && (
        <div className="typing" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      )}

      {!empty && <Markdown text={message.content} streaming={streaming} />}

      {message.error && (
        <div className="note bad" role="alert">
          <CircleAlert size={16} />
          <div>
            <strong>Something went wrong.</strong> {message.error}
          </div>
          {isLast && (
            <button className="btn sm" disabled={busy} onClick={onRetry}>
              <RefreshCw size={14} /> Try again
            </button>
          )}
        </div>
      )}

      {!streaming && meta && meta.sources.length > 0 && (
        <section className="sources" aria-label={meta.cited ? "Sources" : "Pages searched"}>
          <div className="sources-title">
            <Link2 size={13} />
            {meta.cited ? "Sources" : "Searched"}
          </div>
          <ol>
            {meta.sources.map((source) => (
              <li key={source.n}>
                <a className="source" href={source.url} target="_blank" rel="noopener noreferrer">
                  <span className="source-n">{source.n}</span>
                  <span className="source-body">
                    <span className="source-title">{source.title || source.url}</span>
                    <span className="source-host">{host(source.url)}</span>
                  </span>
                </a>
              </li>
            ))}
          </ol>
        </section>
      )}

      {!streaming &&
        meta?.notes_captured.map((note) => (
          <div key={note} className="pill-note">
            <Bookmark size={13} />
            Remembered: {note}
          </div>
        ))}

      {!streaming && meta?.throttled && (
        <div className="note warn">
          <Thermometer size={16} />
          <div>
            The GPU is at {meta.throttled.gpu_temp_c}°C and has throttled itself: this ran at{" "}
            {meta.tokens_per_second.toFixed(0)} tok/s against{" "}
            {meta.throttled.usual_tokens_per_second.toFixed(0)} when cool. Give the laptop a minute
            or more airflow.
          </div>
        </div>
      )}

      {!streaming && (
        <div className={"actions" + (isLast ? "" : " quiet")}>
          {!empty && <CopyButton text={message.content} />}

          {isLast && (
            <button
              className="icon-btn"
              title="Regenerate"
              aria-label="Regenerate answer"
              disabled={busy}
              onClick={onRetry}
            >
              <RefreshCw size={15} />
            </button>
          )}

          {meta?.memory_id != null && (
            <>
              <button
                className={"icon-btn" + (message.rating === "good" ? " is-on" : "")}
                title="Good answer: recall it more"
                aria-label="Good answer"
                aria-pressed={message.rating === "good"}
                onClick={() => onRate(message.id, "good")}
              >
                <ThumbsUp size={15} />
              </button>
              <button
                className={"icon-btn" + (message.rating === "bad" ? " is-on bad" : "")}
                title="Bad answer: never recall it"
                aria-label="Bad answer"
                aria-pressed={message.rating === "bad"}
                onClick={() => onRate(message.id, "bad")}
              >
                <ThumbsDown size={15} />
              </button>
            </>
          )}

          {isLast && message.finish === "length" && !message.error && !empty && (
            <button className="btn soft sm" disabled={busy} onClick={onContinue}>
              Continue <ArrowRight size={14} />
            </button>
          )}

          {usage && usage.completion_tokens > 0 && meta && (
            <span className="stats" title={details(message)}>
              {usage.completion_tokens} tokens · {meta.tokens_per_second.toFixed(1)} tok/s ·{" "}
              {meta.seconds.toFixed(1)}s{message.stopped ? " · stopped" : ""}
            </span>
          )}
        </div>
      )}
    </article>
  );
}

function details({ meta, usage }: Message): string {
  if (!meta || !usage) return "";

  const cached = usage.prompt_tokens_details?.cached_tokens ?? 0;

  return [
    `Prompt: ${usage.prompt_tokens} tokens (${cached} already cached)`,
    `Style: ${meta.sampling}`,
    `Memories recalled: ${meta.recalled}`,
  ].join("\n");
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);

  return (
    <button
      className={"icon-btn" + (done ? " is-on" : "")}
      title={done ? "Copied" : "Copy"}
      aria-label={done ? "Copied" : "Copy message"}
      onClick={() =>
        copyText(text).then((ok) => {
          if (!ok) return;
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        })
      }
    >
      {done ? <Check size={15} /> : <Copy size={15} />}
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

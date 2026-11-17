import { useEffect, useRef } from "react";
import { ArrowUp, ChevronDown, Globe, Sparkles, Square } from "lucide-react";
import type { Prefs, Sampling, WebMode } from "../types";

interface Props {
  text: string;
  busy: boolean;
  disabled: boolean;
  disabledReason: string;
  prefs: Prefs;
  focusKey: string | null;
  onText: (text: string) => void;
  onPrefs: (prefs: Prefs) => void;
  onSend: (text: string) => void;
  onStop: () => void;
}

const WEB: Record<WebMode, string> = { auto: "Web: Auto", on: "Web: Always", off: "Web: Off" };
const STYLE: Record<Sampling, string> = {
  auto: "Style: Auto",
  precise: "Style: Precise",
  balanced: "Style: Balanced",
  creative: "Style: Creative",
};

export function Composer({
  text,
  busy,
  disabled,
  disabledReason,
  prefs,
  focusKey,
  onText,
  onPrefs,
  onSend,
  onStop,
}: Props) {
  const area = useRef<HTMLTextAreaElement>(null);

  // Grow with the text, up to a cap.
  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  }, [text]);

  // Ready to type on a new chat -- but not on touch screens, where focusing
  // throws the keyboard up over the page.
  useEffect(() => {
    if (window.matchMedia("(pointer: fine)").matches) area.current?.focus();
  }, [focusKey]);

  const canSend = !busy && !disabled && text.trim().length > 0;

  const send = () => {
    if (!canSend) return;
    onSend(text.trim());
    onText("");
  };

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      <textarea
        id="composer-input"
        ref={area}
        value={text}
        rows={1}
        aria-label="Message zypherLL"
        placeholder={disabled ? disabledReason : "Ask anything…"}
        onChange={(e) => onText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            send();
          }
        }}
      />

      <div className="composer-bar">
        <label className={"chip" + (prefs.web !== "off" ? " is-on" : "")}>
          <Globe size={15} />
          <span aria-hidden="true">{WEB[prefs.web]}</span>
          <ChevronDown size={14} className="chev" />
          <select
            aria-label="Live web lookup"
            value={prefs.web}
            onChange={(e) => onPrefs({ ...prefs, web: e.target.value as WebMode })}
          >
            <option value="auto">Web: Auto (decides per question)</option>
            <option value="on">Web: Always search</option>
            <option value="off">Web: Off</option>
          </select>
        </label>

        <label className={"chip" + (prefs.sampling !== "auto" ? " is-on" : "")}>
          <Sparkles size={15} />
          <span aria-hidden="true">{STYLE[prefs.sampling]}</span>
          <ChevronDown size={14} className="chev" />
          <select
            aria-label="Answer style"
            value={prefs.sampling}
            onChange={(e) => onPrefs({ ...prefs, sampling: e.target.value as Sampling })}
          >
            <option value="auto">Style: Auto (picked per question)</option>
            <option value="precise">Style: Precise (code, facts)</option>
            <option value="balanced">Style: Balanced</option>
            <option value="creative">Style: Creative (stories, ideas)</option>
          </select>
        </label>

        <span className="spacer" />

        {busy ? (
          <button
            type="button"
            className="send stop"
            title="Stop (Esc)"
            aria-label="Stop generating"
            onClick={onStop}
          >
            <Square size={14} fill="currentColor" />
          </button>
        ) : (
          <button
            type="submit"
            className="send"
            title="Send (Enter)"
            aria-label="Send message"
            disabled={!canSend}
          >
            <ArrowUp size={18} />
          </button>
        )}
      </div>
    </form>
  );
}

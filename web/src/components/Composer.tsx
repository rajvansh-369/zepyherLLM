import { useEffect, useRef, useState } from "react";
import type { Prefs, Sampling, WebMode } from "../types";
import { GlobeIcon, SendIcon, StopIcon } from "../icons";

interface Props {
  busy: boolean;
  disabled: boolean;
  disabledReason: string;
  prefs: Prefs;
  focusKey: string | null;
  onPrefs: (prefs: Prefs) => void;
  onSend: (text: string) => void;
  onStop: () => void;
}

const WEB_LABEL: Record<WebMode, string> = { auto: "Web: auto", on: "Web: always", off: "Web: off" };
const NEXT_WEB: Record<WebMode, WebMode> = { auto: "on", on: "off", off: "auto" };

export function Composer({
  busy,
  disabled,
  disabledReason,
  prefs,
  focusKey,
  onPrefs,
  onSend,
  onStop,
}: Props) {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);

  // Grow with the text, up to a cap.
  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  }, [text]);

  useEffect(() => area.current?.focus(), [focusKey]);

  const canSend = !busy && !disabled && text.trim().length > 0;

  const send = () => {
    if (!canSend) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={area}
          value={text}
          rows={1}
          placeholder={disabled ? disabledReason : "Message zypherLL…"}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
        />
        <div className="composer-bar">
          <button
            className={"chip" + (prefs.web !== "off" ? " on" : "")}
            title="Live web lookup: auto decides per question"
            onClick={() => onPrefs({ ...prefs, web: NEXT_WEB[prefs.web] })}
          >
            <GlobeIcon size={14} /> {WEB_LABEL[prefs.web]}
          </button>

          <label className="chip select" title="Sampling profile">
            <select
              value={prefs.sampling}
              onChange={(e) => onPrefs({ ...prefs, sampling: e.target.value as Sampling })}
            >
              <option value="auto">Style: auto</option>
              <option value="precise">Style: precise</option>
              <option value="balanced">Style: balanced</option>
              <option value="creative">Style: creative</option>
            </select>
          </label>

          <span className="spacer" />

          {busy ? (
            <button className="send stop" title="Stop" onClick={onStop}>
              <StopIcon size={16} />
            </button>
          ) : (
            <button className="send" title="Send" disabled={!canSend} onClick={send}>
              <SendIcon size={18} />
            </button>
          )}
        </div>
      </div>
      <div className="hint">Enter to send · Shift+Enter for a new line · runs entirely on your machine</div>
    </div>
  );
}

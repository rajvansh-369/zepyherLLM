import { memo, useEffect, useRef, useState } from "react";
import { renderMarkdown } from "../markdown";

interface Props {
  text: string;
  streaming: boolean;
}

/**
 * Rendered markdown. While streaming, re-rendering happens at most once a
 * frame however fast the tokens arrive.
 */
export const Markdown = memo(function Markdown({ text, streaming }: Props) {
  const [html, setHtml] = useState(() => renderMarkdown(text));
  const frame = useRef<number | null>(null);

  useEffect(() => {
    if (!streaming) {
      setHtml(renderMarkdown(text));
      return;
    }

    if (frame.current === null) {
      frame.current = requestAnimationFrame(() => {
        frame.current = null;
        setHtml(renderMarkdown(text));
      });
    }
  }, [text, streaming]);

  useEffect(
    () => () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
    },
    [],
  );

  return (
    <div
      className={"markdown" + (streaming ? " streaming" : "")}
      onClick={copyCode}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
});

function copyCode(event: React.MouseEvent<HTMLDivElement>) {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-copy]");
  const code = button?.closest(".code")?.querySelector("code");

  if (!button || !code) return;

  copyText(code.textContent ?? "").then((ok) => {
    button.textContent = ok ? "Copied" : "Failed";
    setTimeout(() => (button.textContent = "Copy"), 1500);
  });
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Clipboard API needs a secure context; 127.0.0.1 is one, a LAN IP is not.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

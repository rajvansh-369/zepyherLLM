// Markdown to safe HTML: GitHub-flavoured, code highlighted, sanitized.

import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import { Marked } from "marked";

const escape = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const marked = new Marked({
  gfm: true,
  breaks: false,
  renderer: {
    code({ text, lang }) {
      const language = (lang || "").trim().split(/\s+/)[0].toLowerCase();
      const known = language && hljs.getLanguage(language);
      const body = known
        ? hljs.highlight(text, { language, ignoreIllegals: true }).value
        : escape(text);

      return (
        `<div class="code"><div class="code-head"><span>${escape(language || "text")}</span>` +
        `<button type="button" class="copy" data-copy>Copy</button></div>` +
        `<pre><code class="hljs">${body}</code></pre></div>`
      );
    },
  },
});

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

/**
 * While an answer is streaming, a code fence may be open; marked renders the
 * rest as code anyway, which is what a reader expects to see mid-answer.
 */
export function renderMarkdown(source: string): string {
  const html = marked.parse(source, { async: false }) as string;
  return DOMPurify.sanitize(html, { ADD_ATTR: ["target", "data-copy"] });
}

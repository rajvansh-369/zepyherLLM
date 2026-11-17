// Markdown to safe HTML: GitHub-flavoured, code highlighted, sanitized.

import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import { Marked } from "marked";

const escape = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

// The two faces of the copy button; CSS shows one or the other.
const ICON =
  'width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';
const COPY_ICON =
  `<svg class="i-copy" ${ICON}><rect x="8" y="8" width="14" height="14" rx="2"/>` +
  `<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;
const CHECK_ICON = `<svg class="i-check" ${ICON}><path d="M20 6 9 17l-5-5"/></svg>`;

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
        `<div class="code"><div class="code-head">` +
        `<span class="code-lang">${escape(language || "text")}</span>` +
        `<button type="button" class="copy" data-copy aria-label="Copy code">` +
        `${COPY_ICON}${CHECK_ICON}<span>Copy</span></button></div>` +
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
  const html = (marked.parse(source, { async: false }) as string)
    // A wide table scrolls inside its own frame instead of widening the page.
    .replace(/<table>/g, '<div class="table-wrap"><table>')
    .replace(/<\/table>/g, "</table></div>");

  return DOMPurify.sanitize(html, { ADD_ATTR: ["target", "data-copy"] });
}

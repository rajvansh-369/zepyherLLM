# zypherLL web

A ChatGPT / Claude style chat in the browser for zypherLL. It is a separate
app that only talks to the zypherLL HTTP API; it does not import or change the
model code.

## Run

Full step-by-step guide, with troubleshooting: `../HOW_TO_RUN.md`. In short,
two terminals:

```bash
# 1. the API, from the project root
python main.py serve

# 2. the web app, from web/
cd web
npm install          # once
copy .env.example .env   # once; put the API key in it (cp on macOS / Linux)
npm run dev
```

Open http://127.0.0.1:5173.

`ZYPHER_API_KEY` in `web/.env` must match the one in the zypherLL `.env`.
The web server adds it to every request it forwards, so the key never reaches
the browser. Leave it empty if the API runs without a key.

For a production build: `npm run build`, then `npm run preview`
(http://127.0.0.1:4173), which serves `dist/` with the same forwarding.

## What it does

| | |
|---|---|
| Chats | sidebar grouped by date, search, rename, delete; kept in this browser's localStorage |
| Answers | streamed token by token; Markdown, tables, highlighted code with a copy button |
| Stop / continue | Stop (or Esc) cancels the generation on the server at once; Continue resumes a stopped or budget-capped answer inside the same reply |
| Edit / regenerate | edit any of your messages and resend from there; regenerate the last answer |
| Web lookup | Auto / Always / Off per message; a "Searching the web…" status while it runs; sources listed under the answer |
| Memory | 👍 / 👎 on answers steers recall; the brain button shows what is known about you, remembered answers, and server settings |
| Status | green dot when the model is ready, a banner while it loads or when the API is down; GPU throttling is reported under the answer |
| Layout | light / dark / system theme; works down to phone width |

Shortcuts: Enter send · Shift+Enter new line · Esc stop · Ctrl+Shift+O new chat.

## How it connects

```
browser ──▶ vite dev/preview server (127.0.0.1:5173) ──▶ zypherLL API (127.0.0.1:8000)
            /health, /v1/*  forwarded, Authorization added
```

Everything goes through the documented API (`projectDetail.md` §5), so any
other front end can do the same. A browser app on another origin that calls
the API directly, without a forwarding server, needs its origin listed in
`ZYPHER_CORS_ORIGINS` in the zypherLL `.env` -- and then holds the API key
itself.

## Files

```
vite.config.ts            dev/preview server and the forwarding to the API
src/
  api.ts                  every API call; SSE stream parsing
  store.ts                localStorage for chats and preferences
  markdown.ts             marked + highlight.js + DOMPurify
  types.ts                API and app shapes
  App.tsx                 state: chats, streaming, health
  components/
    Sidebar.tsx           chat list
    MessageView.tsx       one message with its actions, sources, stats
    Markdown.tsx          rendered answer, code copy
    Composer.tsx          input, web / style toggles, send / stop
    MemoryPanel.tsx       notes, remembered answers, server settings
  styles.css
```

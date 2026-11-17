// The zypherLL HTTP API. Every path is relative: the dev/preview server
// forwards it to the API and adds the key (see vite.config.ts).

import type {
  Health,
  MemoryRecord,
  MemoryStats,
  Meta,
  Role,
  Sampling,
  Settings,
  Usage,
  WebMode,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;

  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the web server.");
  }

  if (!response.ok) {
    throw new ApiError(response.status, await errorText(response));
  }

  return response.json() as Promise<T>;
}

async function errorText(response: Response): Promise<string> {
  // The proxy answers 502/504 with no JSON when zypherLL is not running.
  if (response.status === 502 || response.status === 504) {
    return "The zypherLL API is not running. Start it with: python main.py serve";
  }

  try {
    const body = await response.json();
    const detail = body.detail ?? body.error?.message ?? body;

    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((d) => d.msg ?? String(d)).join("; ");

    return JSON.stringify(detail);
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
}

// -- system ----------------------------------------------------------

export const getHealth = () => request<Health>("/health");
export const getSettings = () => request<Settings>("/v1/settings");

export const patchSettings = (body: Partial<Pick<Settings, "web" | "memory" | "max_tokens">>) =>
  request<Settings>("/v1/settings", { method: "PATCH", body: JSON.stringify(body) });

// -- memory ----------------------------------------------------------

export const getMemory = () =>
  request<{ stats: MemoryStats | null; profile: string[] }>("/v1/memory");

export const getRecords = (kind?: "note" | "exchange", limit = 200) =>
  request<{ total: number; records: MemoryRecord[] }>(
    `/v1/memory/records?limit=${limit}${kind ? `&kind=${kind}` : ""}`,
  );

export const addNote = (text: string) =>
  request<MemoryRecord>("/v1/memory/notes", { method: "POST", body: JSON.stringify({ text }) });

export const rate = (id: number, rating: "good" | "bad") =>
  request<MemoryRecord>(`/v1/memory/${id}/rate`, {
    method: "POST",
    body: JSON.stringify({ rating }),
  });

export const forget = (selector: string) =>
  request<{ removed: number }>(`/v1/memory?selector=${encodeURIComponent(selector)}`, {
    method: "DELETE",
  });

// -- chat ------------------------------------------------------------

export interface ChatOptions {
  web: WebMode;
  sampling: Sampling;
}

/** Progress the API reports before and during an answer. */
export interface Status {
  stage: "searching" | "recalled" | "waking" | string;
  count?: number;
}

export interface ChatResult {
  finish_reason: "stop" | "length";
  usage: Usage;
  meta: Meta;
}

interface Handlers {
  signal: AbortSignal;
  onText: (text: string) => void;
  onStatus: (status: Status) => void;
}

/**
 * Stream one answer. Ending `messages` with an assistant message continues
 * that reply. Aborting the signal closes the connection, which stops the
 * generation on the server at its next token.
 */
export async function streamChat(
  messages: { role: Role; content: string }[],
  options: ChatOptions,
  { signal, onText, onStatus }: Handlers,
): Promise<ChatResult> {
  const body = {
    messages,
    stream: true,
    web: options.web === "auto" ? "auto" : options.web === "on",
    ...(options.sampling !== "auto" && { sampling: options.sampling }),
  };

  let response: Response;

  try {
    response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new ApiError(0, "Cannot reach the web server.");
  }

  if (!response.ok || !response.body) {
    throw new ApiError(response.status, await errorText(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: ChatResult | null = null;

  for (;;) {
    const { done, value } = await reader.read();

    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Server-sent events: blocks separated by a blank line.
    let end: number;

    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);

      for (const line of block.split("\n")) {
        if (!line.startsWith("data:")) continue;

        const data = line.slice(5).trim();

        if (data === "[DONE]") return finish(result);

        const event = JSON.parse(data);

        if (event.error) throw new ApiError(500, event.error.message);

        const choice = event.choices?.[0];
        const text = choice?.delta?.content;

        if (text) onText(text);

        if (choice?.finish_reason) {
          result = { finish_reason: choice.finish_reason, usage: event.usage, meta: event.zypher };
        } else if (event.zypher) {
          onStatus(event.zypher);
        }
      }
    }
  }

  return finish(result);
}

function finish(result: ChatResult | null): ChatResult {
  if (!result) throw new ApiError(500, "The answer ended before it finished.");
  return result;
}

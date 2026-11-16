// Conversations and preferences, kept in this browser.
//
// localStorage can be missing (private windows, blocked site data) or full;
// every access is guarded so the app still runs, just without persistence.

import type { Conversation, Prefs } from "./types";

const CONVERSATIONS = "zypher.conversations.v1";
const PREFS = "zypher.prefs.v1";

export const DEFAULT_PREFS: Prefs = { web: "auto", sampling: "auto", theme: "system" };

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

/** Returns false when the write failed -- usually the storage is full. */
function write(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function loadConversations(): Conversation[] {
  const list = read<Conversation[]>(CONVERSATIONS, []);
  return Array.isArray(list) ? list.sort((a, b) => b.updated - a.updated) : [];
}

export const saveConversations = (list: Conversation[]) => write(CONVERSATIONS, list);

export const loadPrefs = (): Prefs => ({ ...DEFAULT_PREFS, ...read<Partial<Prefs>>(PREFS, {}) });
export const savePrefs = (prefs: Prefs) => write(PREFS, prefs);

export const uid = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2) + Date.now().toString(36);

export function titleFrom(text: string): string {
  const line = text.replace(/\s+/g, " ").trim();
  return line.length > 48 ? line.slice(0, 47).trimEnd() + "…" : line || "New chat";
}

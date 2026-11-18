// Shapes of what the zypherLL API sends back, and of what this app keeps.

export type Role = "user" | "assistant";
export type WebMode = "auto" | "on" | "off";
export type Sampling = "auto" | "precise" | "balanced" | "creative";

export interface Source {
  n: number;
  title: string;
  url: string;
}

export interface Throttled {
  gpu_temp_c: number;
  usual_tokens_per_second: number;
}

/** The `zypher` object on a finished answer. */
export interface Meta {
  /** Id of the model that wrote this answer. */
  model?: string;
  sampling: string;
  live: boolean;
  sources: Source[];
  cited: boolean;
  recalled: number;
  notes_captured: string[];
  memory_id: number | null;
  seconds: number;
  tokens_per_second: number;
  throttled: Throttled | null;
  rounds: number;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens: number };
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  /** "length": the answer stopped early (budget, or the user pressed stop). */
  finish?: "stop" | "length";
  stopped?: boolean;
  meta?: Meta;
  usage?: Usage;
  error?: string;
  rating?: "good" | "bad";
}

export interface Conversation {
  id: string;
  title: string;
  created: number;
  updated: number;
  messages: Message[];
}

export interface Health {
  status: "loading" | "ready" | "error";
  error: string | null;
  backend: string;
  device: string | null;
  gpu?: string | null;
  vram_gb: number;
  context?: number | null;
  gpu_layers?: string | null;
  kv_cache?: string | null;
  /** Id of the model on the GPU. */
  model?: string;
  models?: ModelInfo[];
  /** Slot being loaded after a switch in the settings, or null. */
  switching?: Slot | null;
  auth: boolean;
}

export type Slot = "general" | "code";

export interface ModelInfo {
  slot: Slot;
  id: string;
  active: boolean;
}

export interface Settings {
  web: boolean;
  memory: boolean;
  max_tokens: number;
  max_prompt_tokens: number;
  /** The slot loaded now. */
  model: Slot;
  models: ModelInfo[];
  switching: Slot | null;
  switch_error: string | null;
}

export interface MemoryStats {
  ready: boolean;
  enabled: boolean;
  total: number;
  notes: number;
  exchanges: number;
  rated: number;
  encoder: string;
  dir: string;
  error: string | null;
}

/** What the user wrote about themselves. Every field is "" when unset. */
export interface About {
  name: string;
  role: string;
  about: string;
  style: string;
}

export interface MemoryRecord {
  id: number;
  ts: string;
  kind: "note" | "exchange";
  q: string;
  a: string;
  score: number;
  uses: number;
}

export interface Prefs {
  web: WebMode;
  sampling: Sampling;
  theme: "system" | "light" | "dark";
  /** Desktop only: the sidebar is tucked away. */
  sidebarCollapsed: boolean;
}

/** What the assistant is doing before or while it writes. */
export interface StatusInfo {
  stage: string;
  text: string;
}

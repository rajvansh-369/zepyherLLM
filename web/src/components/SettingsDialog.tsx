import { useCallback, useEffect, useRef, useState } from "react";
import {
  CircleAlert,
  CircleCheck,
  Cpu,
  History,
  IdCard,
  LoaderCircle,
  Trash2,
  UserRound,
  X,
} from "lucide-react";
import * as api from "../api";
import type { About, Health, MemoryRecord, MemoryStats, Settings, Slot } from "../types";

export type SettingsTab = "about" | "notes" | "exchanges" | "server";

interface Props {
  initialTab: SettingsTab;
  health: Health | null;
  offline: string | null;
  /** A model switch was started: check health now rather than on the timer. */
  onModelChange: () => void;
  onClose: () => void;
}

// How each slot is presented. The ids come from the API (ZYPHER_MODEL_ID,
// ZYPHER_CODE_MODEL_ID); these are the names and what each is for.
const SLOT_INFO: Record<Slot, { name: string; text: string }> = {
  general: {
    name: "Zypher",
    text: "Unfiltered general model, fully on the GPU. Fast, direct answers. Red theme.",
  },
  code: {
    name: "Coder",
    text: "Coding model for code, debugging and the zy CLI. Soft theme.",
  },
};

const TABS = [
  { id: "about", label: "About me", icon: IdCard },
  { id: "notes", label: "Notes", icon: UserRound },
  { id: "exchanges", label: "Remembered answers", icon: History },
  { id: "server", label: "Engine", icon: Cpu },
] as const;

export function SettingsDialog({ initialTab, health, offline, onModelChange, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [tab, setTab] = useState<SettingsTab>(initialTab);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [profile, setProfile] = useState<string[]>([]);
  const [records, setRecords] = useState<MemoryRecord[] | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [confirmAll, setConfirmAll] = useState(false);

  // A native modal dialog: the browser traps focus inside it, closes it on
  // Esc, and hands focus back to whatever opened it.
  useEffect(() => {
    const el = dialog.current;
    if (el && !el.open) el.showModal();
  }, []);

  const load = useCallback(async () => {
    setError(null);

    try {
      const [overview, list, current] = await Promise.all([
        api.getMemory(),
        tab === "server" || tab === "about"
          ? Promise.resolve(null)
          : api.getRecords(tab === "notes" ? "note" : "exchange"),
        api.getSettings(),
      ]);

      setStats(overview.stats);
      setProfile(overview.profile);
      setRecords(list ? list.records : null);
      setSettings(current);
    } catch (e) {
      setRecords([]);
      setError((e as Error).message);
    }
  }, [tab]);

  useEffect(() => {
    setRecords(null);
    load();
  }, [load]);

  const run = async (action: () => Promise<unknown>, message?: string) => {
    setError(null);
    setInfo(null);

    try {
      await action();
      if (message) setInfo(message);
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const update = (body: Partial<Pick<Settings, "web" | "memory" | "max_tokens" | "model">>) =>
    run(async () => setSettings(await api.patchSettings(body)));

  return (
    <dialog
      ref={dialog}
      className="sheet"
      aria-labelledby="settings-title"
      onClose={onClose}
      onClick={(e) => {
        // A click on the dialog element itself is a click on the backdrop.
        if (e.target === dialog.current) dialog.current.close();
      }}
    >
      <header className="sheet-head">
        <h2 id="settings-title">Memory &amp; settings</h2>
        <button
          className="icon-btn"
          title="Close"
          aria-label="Close"
          onClick={() => dialog.current?.close()}
        >
          <X size={18} />
        </button>
      </header>

      <div className="sheet-tabs">
        <div className="segmented" role="tablist" aria-label="Sections">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              role="tab"
              id={`tab-${id}`}
              aria-selected={tab === id}
              aria-controls="settings-panel"
              onClick={() => {
                setInfo(null);
                setTab(id);
              }}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="sheet-body" id="settings-panel" role="tabpanel" aria-labelledby={`tab-${tab}`}>
        {error && (
          <div className="note bad" role="alert">
            <CircleAlert size={16} />
            <div>{error}</div>
          </div>
        )}
        {info && (
          <div className="note" role="status">
            <CircleCheck size={16} />
            <div>{info}</div>
          </div>
        )}

        {tab === "about" && <AboutForm onSaved={() => setInfo("Saved. Every new answer uses it.")} />}

        {tab === "notes" && (
          <>
            <p className="muted">
              Smaller facts, picked up automatically when you say things like “I prefer tabs”, or
              added here. The most recent ones go into every conversation.
            </p>
            <form
              className="field-row"
              onSubmit={(e) => {
                e.preventDefault();
                const text = draft.trim();
                if (!text) return;
                run(() => api.addNote(text), "Saved.").then(() => setDraft(""));
              }}
            >
              <input
                className="field"
                value={draft}
                maxLength={200}
                aria-label="A fact about you"
                placeholder="e.g. I write Python with type hints"
                onChange={(e) => setDraft(e.target.value)}
              />
              <button className="btn primary" disabled={!draft.trim()}>
                Add
              </button>
            </form>
            <RecordList
              records={records}
              inPrompt={profile}
              empty="Nothing noted about you yet."
              onForget={(r) => run(() => api.forget(r.a), "Forgotten.")}
            />
          </>
        )}

        {tab === "exchanges" && (
          <>
            <p className="muted">
              Past answers the model is shown again when a similar question comes up. Rate answers
              with the thumbs under each one to steer this.
            </p>
            <RecordList
              records={records}
              inPrompt={[]}
              empty="No answers remembered yet."
              onForget={(r) => run(() => api.forget(r.q), "Forgotten.")}
            />
          </>
        )}

        {tab === "server" && (
          <>
            {(health?.models?.length ?? 0) > 1 && (
              <>
                <h3 className="section-title">Model</h3>
                <p className="muted">
                  One model is loaded at a time, fully on the GPU, for the best speed. Switching
                  takes up to a minute; your choice is kept after a restart.
                </p>
                <div className="model-cards" role="radiogroup" aria-label="Model">
                  {health!.models!.map((m) => {
                    const info = SLOT_INFO[m.slot];
                    const loading = health?.switching === m.slot;
                    const busy = !!health?.switching || health?.status !== "ready";

                    return (
                      <button
                        key={m.slot}
                        role="radio"
                        aria-checked={m.active}
                        className={"model-card " + m.slot}
                        disabled={busy || m.active}
                        onClick={() =>
                          run(async () => {
                            await api.patchSettings({ model: m.slot });
                            onModelChange();
                          }, `Loading ${m.id}…`)
                        }
                      >
                        <span className="model-card-head">
                          <strong>{info.name}</strong>
                          {m.active && !health?.switching && <span className="tag">Loaded</span>}
                          {loading && (
                            <span className="tag">
                              <LoaderCircle size={12} className="spin" /> Loading
                            </span>
                          )}
                        </span>
                        <small>{info.text}</small>
                        <code>{m.id}</code>
                      </button>
                    );
                  })}
                </div>
                {settings?.switch_error && (
                  <p className="muted">Last switch failed: {settings.switch_error}</p>
                )}
              </>
            )}

            <h3 className="section-title">Engine</h3>
            <dl className="engine">
              <Fact label="Status">
                <span className={"dot " + (offline ? "error" : (health?.status ?? "loading"))} />
                {offline ? "Offline" : health ? capitalize(health.status) : "Checking…"}
              </Fact>
              {health?.model && <Fact label="Model">{health.model}</Fact>}
              {health?.backend && <Fact label="Runtime">{health.backend}</Fact>}
              {health?.gpu && (
                <Fact label="GPU">
                  {health.gpu.replace(/^NVIDIA GeForce /, "")} · {health.vram_gb} GB
                </Fact>
              )}
              {health?.context && (
                <Fact label="Context window">{health.context.toLocaleString()} tokens</Fact>
              )}
              {health?.gpu_layers && (
                <Fact label="Layers on GPU">
                  {health.gpu_layers === "all" ? "All" : health.gpu_layers}
                </Fact>
              )}
              {health?.kv_cache && <Fact label="KV cache">{health.kv_cache}</Fact>}
            </dl>
            {(offline || health?.error) && (
              <p className="muted">{offline ?? health?.error}</p>
            )}

            <h3 className="section-title">Server defaults</h3>
            {settings ? (
              <>
                <Toggle
                  title="Live web lookup"
                  hint="Search the web for questions about the present"
                  checked={settings.web}
                  onChange={(web) => update({ web })}
                />
                <Toggle
                  title="Memory"
                  hint="Recall past answers and learn from this chat"
                  checked={settings.memory}
                  onChange={(memory) => update({ memory })}
                />
                <div className="setting">
                  <label className="setting-text" htmlFor="max-tokens">
                    <strong>Answer budget</strong>
                    <small>
                      Most tokens in one answer (prompt budget{" "}
                      {settings.max_prompt_tokens.toLocaleString()})
                    </small>
                  </label>
                  <input
                    id="max-tokens"
                    className="field num"
                    type="number"
                    min={64}
                    step={256}
                    defaultValue={settings.max_tokens}
                    key={settings.max_tokens}
                    onBlur={(e) => {
                      const value = parseInt(e.target.value, 10);
                      if (value >= 1 && value !== settings.max_tokens) update({ max_tokens: value });
                    }}
                  />
                </div>
                <p className="muted" style={{ marginTop: 12 }}>
                  These apply to every client of the API and reset when it restarts.
                </p>
              </>
            ) : (
              <Skeleton rows={3} />
            )}
          </>
        )}
      </div>

      <footer className="sheet-foot">
        <span className="muted">
          {stats
            ? `${stats.total} stored · ${stats.notes} notes · ${stats.exchanges} answers`
            : "Memory is loading…"}
        </span>

        {confirmAll ? (
          <span className="row">
            <button
              className="btn danger sm"
              onClick={() =>
                run(() => api.forget("all"), "Memory cleared.").then(() => setConfirmAll(false))
              }
            >
              Erase everything
            </button>
            <button className="btn ghost sm" onClick={() => setConfirmAll(false)}>
              Cancel
            </button>
          </span>
        ) : (
          <button className="btn ghost sm" onClick={() => setConfirmAll(true)}>
            <Trash2 size={14} /> Clear memory…
          </button>
        )}
      </footer>
    </dialog>
  );
}

// Same caps as the API (zypher/model/memory.py ABOUT_FIELDS).
const ABOUT_LIMITS: Record<keyof About, number> = { name: 60, role: 120, about: 600, style: 600 };

// One click adds a common instruction to the response style.
const STYLE_PRESETS = [
  ["Concise", "Keep answers short and to the point."],
  ["Step by step", "Break complex problems into numbered steps and explain the reasoning."],
  ["Code first", "Lead with working code, then explain it."],
  ["Beginner friendly", "Explain terms simply and avoid unexplained jargon."],
] as const;

const EMPTY_ABOUT: About = { name: "", role: "", about: "", style: "" };

function AboutForm({ onSaved }: { onSaved: () => void }) {
  const [saved, setSaved] = useState<About | null>(null);
  const [draft, setDraft] = useState<About>(EMPTY_ABOUT);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getAbout()
      .then((about) => {
        setSaved(about);
        setDraft(about);
      })
      .catch((e) => {
        setSaved(EMPTY_ABOUT);
        setError((e as Error).message);
      });
  }, []);

  if (saved === null) return <Skeleton rows={4} />;

  const dirty = (Object.keys(draft) as (keyof About)[]).some((k) => draft[k] !== saved[k]);
  const set = (field: keyof About) => (value: string) => setDraft((d) => ({ ...d, [field]: value }));

  const addPreset = (sentence: string) => {
    if (draft.style.includes(sentence)) return;
    const next = (draft.style.trim() + " " + sentence).trim();
    if (next.length <= ABOUT_LIMITS.style) set("style")(next);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);

    try {
      const about = await api.saveAbout(draft);
      setSaved(about);
      setDraft(about);
      onSaved();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="about-form" onSubmit={submit}>
      <p className="muted">
        Tell the model who you are and how you like answers. This goes into every conversation, so
        answers match your level and style from the first message.
      </p>
      {error && (
        <div className="note bad" role="alert">
          <CircleAlert size={16} />
          <div>{error}</div>
        </div>
      )}

      <div className="about-grid">
        <Field label="Name" field="name" value={draft.name} onChange={set("name")} placeholder="e.g. Sam" />
        <Field
          label="What you do"
          field="role"
          value={draft.role}
          onChange={set("role")}
          placeholder="e.g. Backend developer, 5 years of PHP"
        />
      </div>
      <Field
        label="About you"
        field="about"
        multiline
        value={draft.about}
        onChange={set("about")}
        placeholder="Your stack, current projects, what you are learning, what you already know well…"
      />
      <Field
        label="How should answers be written?"
        field="style"
        multiline
        value={draft.style}
        onChange={set("style")}
        placeholder="e.g. Explain the why before the how. Use TypeScript in examples."
      />
      <div className="presets" aria-label="Add a common instruction">
        {STYLE_PRESETS.map(([label, sentence]) => (
          <button
            type="button"
            key={label}
            className="preset"
            aria-pressed={draft.style.includes(sentence)}
            onClick={() => addPreset(sentence)}
          >
            + {label}
          </button>
        ))}
      </div>

      <div className="about-actions">
        {dirty && (
          <button type="button" className="btn ghost sm" onClick={() => setDraft(saved)}>
            Discard
          </button>
        )}
        <button className="btn primary" disabled={!dirty || busy}>
          {busy ? "Saving…" : "Save"}
        </button>
      </div>
    </form>
  );
}

function Field({
  label,
  field,
  value,
  onChange,
  placeholder,
  multiline,
}: {
  label: string;
  field: keyof About;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  multiline?: boolean;
}) {
  const limit = ABOUT_LIMITS[field];
  const id = `about-${field}`;

  return (
    <div className="about-field">
      <label htmlFor={id}>
        <span>{label}</span>
        {value.length > limit * 0.8 && (
          <small className="muted">
            {value.length}/{limit}
          </small>
        )}
      </label>
      {multiline ? (
        <textarea
          id={id}
          className="field area"
          rows={3}
          maxLength={limit}
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : (
        <input
          id={id}
          className="field"
          maxLength={limit}
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </div>
  );
}

const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="kv">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function Toggle({
  title,
  hint,
  checked,
  onChange,
}: {
  title: string;
  hint: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <div className="setting">
      <span className="setting-text">
        <strong>{title}</strong>
        <small>{hint}</small>
      </span>
      <button
        className="switch"
        role="switch"
        aria-checked={checked}
        aria-label={title}
        onClick={() => onChange(!checked)}
      />
    </div>
  );
}

function Skeleton({ rows }: { rows: number }) {
  return (
    <div aria-hidden="true" style={{ marginTop: 12 }}>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} style={{ padding: "10px 0" }}>
          <div className="skeleton" style={{ width: `${70 - i * 12}%` }} />
          <div className="skeleton" style={{ width: `${40 + i * 8}%` }} />
        </div>
      ))}
    </div>
  );
}

function RecordList({
  records,
  inPrompt,
  empty,
  onForget,
}: {
  records: MemoryRecord[] | null;
  inPrompt: string[];
  empty: string;
  onForget: (record: MemoryRecord) => void;
}) {
  const [asking, setAsking] = useState<number | null>(null);

  if (records === null) return <Skeleton rows={3} />;
  if (records.length === 0) return <div className="empty">{empty}</div>;

  return (
    <ul className="records">
      {records.map((r) => (
        <li key={r.id} className="record">
          <div className="record-text">
            {r.kind === "exchange" && <div className="record-q">{r.q}</div>}
            <div className={r.kind === "exchange" ? "record-a" : ""}>{r.a}</div>
            <div className="record-meta">
              <span>{new Date(r.ts).toLocaleDateString()}</span>
              {r.score !== 0 && (
                <span>
                  · score {r.score > 0 ? "+" : ""}
                  {r.score}
                </span>
              )}
              {r.uses > 0 && <span>· recalled {r.uses}×</span>}
              {inPrompt.includes(r.a) && <span className="tag">in every prompt</span>}
            </div>
          </div>

          {asking === r.id ? (
            <span className="record-confirm">
              <button
                className="mini danger"
                autoFocus
                onClick={() => {
                  setAsking(null);
                  onForget(r);
                }}
              >
                Forget
              </button>
              <button className="mini" onClick={() => setAsking(null)}>
                Cancel
              </button>
            </span>
          ) : (
            <button
              className="icon-btn"
              title="Forget this (and anything very similar)"
              aria-label="Forget this memory"
              onClick={() => setAsking(r.id)}
            >
              <Trash2 size={15} />
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}

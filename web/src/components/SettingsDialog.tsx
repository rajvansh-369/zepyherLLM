import { useCallback, useEffect, useRef, useState } from "react";
import { CircleAlert, CircleCheck, Cpu, History, Trash2, UserRound, X } from "lucide-react";
import * as api from "../api";
import type { Health, MemoryRecord, MemoryStats, Settings } from "../types";

export type SettingsTab = "notes" | "exchanges" | "server";

interface Props {
  initialTab: SettingsTab;
  health: Health | null;
  offline: string | null;
  onClose: () => void;
}

const TABS = [
  { id: "notes", label: "About you", icon: UserRound },
  { id: "exchanges", label: "Remembered answers", icon: History },
  { id: "server", label: "Engine", icon: Cpu },
] as const;

export function SettingsDialog({ initialTab, health, offline, onClose }: Props) {
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
        tab === "server"
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

  const update = (body: Partial<Pick<Settings, "web" | "memory" | "max_tokens">>) =>
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

        {tab === "notes" && (
          <>
            <p className="muted">
              Facts about you. The most recent ones go into every conversation. They are also
              picked up automatically when you say things like “I prefer tabs”.
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
            <h3 className="section-title">Engine</h3>
            <dl className="engine">
              <Fact label="Status">
                <span className={"dot " + (offline ? "error" : (health?.status ?? "loading"))} />
                {offline ? "Offline" : health ? capitalize(health.status) : "Checking…"}
              </Fact>
              <Fact label="Model">zephyr-7b</Fact>
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

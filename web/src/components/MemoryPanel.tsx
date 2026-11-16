import { useCallback, useEffect, useState } from "react";
import * as api from "../api";
import type { MemoryRecord, MemoryStats, Settings } from "../types";
import { CloseIcon, TrashIcon } from "../icons";

interface Props {
  onClose: () => void;
}

type Tab = "notes" | "exchanges" | "settings";

export function MemoryPanel({ onClose }: Props) {
  const [tab, setTab] = useState<Tab>("notes");
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [profile, setProfile] = useState<string[]>([]);
  const [records, setRecords] = useState<MemoryRecord[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [confirmAll, setConfirmAll] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [overview, list, current] = await Promise.all([
        api.getMemory(),
        tab === "settings" ? Promise.resolve(null) : api.getRecords(tab === "notes" ? "note" : "exchange"),
        api.getSettings(),
      ]);
      setStats(overview.stats);
      setProfile(overview.profile);
      if (list) setRecords(list.records);
      setSettings(current);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [tab]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

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
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-label="Memory and settings">
        <header>
          <h2>Memory &amp; settings</h2>
          <button className="icon-btn" title="Close" onClick={onClose}>
            <CloseIcon />
          </button>
        </header>

        <div className="tabs">
          {(["notes", "exchanges", "settings"] as Tab[]).map((name) => (
            <button key={name} className={tab === name ? "on" : ""} onClick={() => setTab(name)}>
              {name === "notes" ? "About you" : name === "exchanges" ? "Remembered answers" : "Server"}
            </button>
          ))}
        </div>

        <div className="modal-body">
          {error && <div className="error-box">{error}</div>}
          {info && <div className="info-box">{info}</div>}

          {tab === "notes" && (
            <>
              <p className="muted">
                Facts about you. The most recent ones go into every conversation. They are also
                captured automatically when you say things like “I prefer tabs”.
              </p>
              <form
                className="row"
                onSubmit={(e) => {
                  e.preventDefault();
                  const text = draft.trim();
                  if (!text) return;
                  run(() => api.addNote(text), "Saved.").then(() => setDraft(""));
                }}
              >
                <input
                  value={draft}
                  maxLength={200}
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
                Past answers the model is shown again when a similar question comes up. Rate
                answers with 👍 / 👎 in the chat to steer this.
              </p>
              <RecordList
                records={records}
                inPrompt={[]}
                empty="No answers remembered yet."
                onForget={(r) => run(() => api.forget(r.q), "Forgotten.")}
              />
            </>
          )}

          {tab === "settings" && settings && (
            <div className="settings">
              <label className="switch-row">
                <span>
                  <strong>Live web lookup</strong>
                  <small>Search the web for questions about the present</small>
                </span>
                <input type="checkbox" checked={settings.web} onChange={(e) => update({ web: e.target.checked })} />
              </label>
              <label className="switch-row">
                <span>
                  <strong>Memory</strong>
                  <small>Recall past answers and learn from this chat</small>
                </span>
                <input
                  type="checkbox"
                  checked={settings.memory}
                  onChange={(e) => update({ memory: e.target.checked })}
                />
              </label>
              <label className="switch-row">
                <span>
                  <strong>Answer budget</strong>
                  <small>Max tokens per answer (prompt budget {settings.max_prompt_tokens})</small>
                </span>
                <input
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
              </label>
              <p className="muted">These are server-wide and reset when the API restarts.</p>
            </div>
          )}
        </div>

        <footer>
          <span className="muted">
            {stats
              ? `${stats.total} stored · ${stats.notes} notes · ${stats.exchanges} answers · encoder ${stats.encoder.split("/").pop()}`
              : "Memory is loading…"}
          </span>
          {confirmAll ? (
            <span className="row">
              <button
                className="btn danger"
                onClick={() => run(() => api.forget("all"), "Memory cleared.").then(() => setConfirmAll(false))}
              >
                Erase everything
              </button>
              <button className="btn ghost" onClick={() => setConfirmAll(false)}>
                Cancel
              </button>
            </span>
          ) : (
            <button className="btn ghost" onClick={() => setConfirmAll(true)}>
              Clear memory…
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}

function RecordList({
  records,
  inPrompt,
  empty,
  onForget,
}: {
  records: MemoryRecord[];
  inPrompt: string[];
  empty: string;
  onForget: (record: MemoryRecord) => void;
}) {
  if (records.length === 0) return <div className="empty-list">{empty}</div>;

  return (
    <ul className="records">
      {records.map((r) => (
        <li key={r.id}>
          <div className="record-text">
            {r.kind === "exchange" && <div className="record-q">{r.q}</div>}
            <div className={r.kind === "exchange" ? "record-a" : ""}>{r.a}</div>
            <div className="record-meta">
              {new Date(r.ts).toLocaleDateString()}
              {r.score !== 0 && ` · score ${r.score > 0 ? "+" : ""}${r.score}`}
              {r.uses > 0 && ` · recalled ${r.uses}×`}
              {inPrompt.includes(r.a) && " · in every prompt"}
            </div>
          </div>
          <button
            className="icon-btn"
            title="Forget this (and anything very similar)"
            onClick={() => onForget(r)}
          >
            <TrashIcon size={15} />
          </button>
        </li>
      ))}
    </ul>
  );
}

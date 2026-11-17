// Brief confirmations and errors that do not belong to one message.
//
// A tiny store rather than context: anything can call toast() without being
// handed a function, and only <Toasts /> re-renders.

import { useSyncExternalStore } from "react";
import { CircleAlert, CircleCheck } from "lucide-react";

interface Toast {
  id: number;
  text: string;
  kind: "info" | "error";
}

let toasts: Toast[] = [];
let nextId = 1;
const listeners = new Set<() => void>();

const emit = () => listeners.forEach((listener) => listener());

function push(text: string, kind: Toast["kind"]) {
  const id = nextId++;

  // Three at most; the oldest gives way.
  toasts = [...toasts.slice(-2), { id, text, kind }];
  emit();

  setTimeout(
    () => {
      toasts = toasts.filter((item) => item.id !== id);
      emit();
    },
    kind === "error" ? 5000 : 3500,
  );
}

export const toast = Object.assign((text: string) => push(text, "info"), {
  error: (text: string) => push(text, "error"),
});

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function Toasts() {
  const list = useSyncExternalStore(subscribe, () => toasts);

  // role=status announces politely and never takes focus.
  return (
    <div className="toasts" role="status" aria-live="polite">
      {list.map((item) => (
        <div key={item.id} className="toast">
          {item.kind === "error" ? <CircleAlert size={16} /> : <CircleCheck size={16} />}
          <span>{item.text}</span>
        </div>
      ))}
    </div>
  );
}

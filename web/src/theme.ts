// Theme, and the two browser features the look leans on.

import { flushSync } from "react-dom";
import type { Prefs } from "./types";

type Theme = Prefs["theme"];

const darkQuery = () => window.matchMedia("(prefers-color-scheme: dark)");

export const resolveTheme = (theme: Theme): "light" | "dark" =>
  theme === "system" ? (darkQuery().matches ? "dark" : "light") : theme;

/** index.html sets this before first paint; this keeps it in step afterwards. */
export function applyTheme(theme: Theme) {
  document.documentElement.setAttribute("data-theme", resolveTheme(theme));
}

/** Calls back when the OS switches between light and dark. */
export function watchSystemTheme(onChange: () => void) {
  const query = darkQuery();
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

export const prefersReducedMotion = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Apply a state change as a view transition: the browser cross-fades the old
 * page into the new one, and elements that carry a view-transition-name (the
 * composer) glide to where they now are. Where that is unsupported, or the
 * reader asked for less motion, the change simply happens.
 */
export function withTransition(update: () => void) {
  if (!("startViewTransition" in document) || prefersReducedMotion()) {
    update();
    return;
  }

  document.startViewTransition(() => flushSync(update));
}

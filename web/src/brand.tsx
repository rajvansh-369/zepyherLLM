import { useId } from "react";

/** The zypherLL mark: a Z on the brand gradient. */
export function Logo({ size = 28 }: { size?: number }) {
  // A gradient is referenced by id, and there can be several marks on a page.
  const id = "zg" + useId().replace(/[^a-zA-Z0-9]/g, "");

  return (
    <svg
      className="logo-mark"
      width={size}
      height={size}
      viewBox="0 0 32 32"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#6366F1" />
          <stop offset="0.6" stopColor="#8B5CF6" />
          <stop offset="1" stopColor="#22D3EE" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill={`url(#${id})`} />
      <path
        d="M10 10.5h12L10 21.5h12"
        fill="none"
        stroke="#fff"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

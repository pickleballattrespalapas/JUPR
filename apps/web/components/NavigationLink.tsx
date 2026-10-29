"use client";

import Link, { useLinkStatus } from "next/link";
import type { ComponentProps } from "react";

export function NavigationFeedback() {
  return (
    <span
      role="status"
      aria-live="polite"
      style={{ position: "fixed", bottom: "1rem", right: "1rem", zIndex: 1000,
        padding: "0.7rem 1rem", borderRadius: "10px", background: "#172554",
        color: "white", boxShadow: "0 4px 16px #0003", pointerEvents: "none",
        fontSize: "0.95rem", fontWeight: 700 }}
    >
      Opening page…
    </span>
  );
}

function PendingNavigation() {
  const { pending } = useLinkStatus();
  return pending ? <NavigationFeedback /> : null;
}

/** Next owns the transition, including cancellation and modified/new-tab clicks. */
export default function NavigationLink({ children, ...props }: ComponentProps<typeof Link>) {
  return <Link {...props}>{children}<PendingNavigation /></Link>;
}

"use client";

import type { ReactNode } from "react";

export default function PublicMobileNavigation({ children, className }: { children: ReactNode; className: string }) {
  return (
    <details className={className} onClickCapture={event => {
      if (event.target instanceof Element && event.target.closest("a[href]")) {
        event.currentTarget.open = false;
      }
    }}>
      <summary>Menu</summary>
      {children}
    </details>
  );
}

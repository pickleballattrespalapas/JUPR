"use client";

import type { ReactNode } from "react";
import { useEffect, useRef } from "react";

type Props = {
  leagueName: string;
  children: ReactNode;
};

export default function SelectedLeaguePanelScope({ leagueName, children }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const lastDispatchRef = useRef(0);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    function applySelectedLeague() {
      const articles = Array.from(root.querySelectorAll("article"));
      for (const article of articles) {
        const heading = article.querySelector("h2");
        const headingText = heading?.textContent?.trim() || "";
        if (headingText === "Admin session" || headingText === "Admin session and recovery") {
          (article as HTMLElement).style.display = "none";
        } else if (headingText === "Choose a league") {
          if (heading) heading.textContent = leagueName;
        } else if (headingText === "1. Select and recover league") {
          if (heading) heading.textContent = "Saved awards workflow";
        }
      }

      const select = Array.from(root.querySelectorAll("select")).find((candidate) => {
        const labelText = candidate.closest("label")?.textContent || "";
        return /league/i.test(labelText)
          && Array.from(candidate.options).some((option) => option.value === leagueName);
      });

      if (!select) return;
      const label = select.closest("label");
      if (label) (label as HTMLElement).style.display = "none";

      for (const button of Array.from(root.querySelectorAll("button"))) {
        if ((button.textContent || "").trim() === "Refresh leagues") {
          (button as HTMLElement).style.display = "none";
        }
      }

      if (select.value === leagueName || Date.now() - lastDispatchRef.current < 500) return;
      lastDispatchRef.current = Date.now();
      const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
      if (setter) setter.call(select, leagueName);
      else select.value = leagueName;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }

    applySelectedLeague();
    const observer = new MutationObserver(applySelectedLeague);
    observer.observe(root, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [leagueName]);

  return <div ref={rootRef} data-selected-league={leagueName}>{children}</div>;
}

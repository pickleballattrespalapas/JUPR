"use client";

import type { ReactNode } from "react";
import { useEffect, useRef } from "react";

type Props = {
  tournamentId: string;
  tournamentName?: string | null;
  children: ReactNode;
};

const HIDDEN_CONTEXT_HEADINGS = new Set([
  "admin session",
  "tournament admin session",
  "registration reporting session",
  "create or open a tournament",
  "1. create tournament shell",
  "2. select tournament",
  "choose a tournament",
  "select tournament",
  "tournament selection",
  "tournament"
]);

function normalized(value?: string | null): string {
  return String(value || "").trim().toLowerCase();
}

function matchingTournamentOption(select: HTMLSelectElement, tournamentId: string, tournamentName: string): HTMLOptionElement | null {
  const options = Array.from(select.options);
  return options.find((option) => option.value === tournamentId)
    || options.find((option) => tournamentName && normalized(option.textContent).includes(normalized(tournamentName)))
    || null;
}

export default function SelectedTournamentPanelScope({ tournamentId, tournamentName, children }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const lastDispatchRef = useRef(0);

  useEffect(() => {
    function applySelectedTournament() {
      const container = rootRef.current;
      if (!container) return;

      for (const article of Array.from(container.querySelectorAll("article"))) {
        const heading = article.querySelector("h2");
        const headingText = normalized(heading?.textContent);
        if (HIDDEN_CONTEXT_HEADINGS.has(headingText)) {
          (article as HTMLElement).style.display = "none";
        }
      }

      const candidate = Array.from(container.querySelectorAll("select")).find((select) => {
        const aria = normalized(select.getAttribute("aria-label"));
        const label = normalized(select.closest("label")?.textContent);
        const looksLikeTournament = aria.includes("tournament") || label.includes("tournament");
        return looksLikeTournament && Boolean(matchingTournamentOption(select, tournamentId, tournamentName || ""));
      });

      if (!candidate) return;
      const option = matchingTournamentOption(candidate, tournamentId, tournamentName || "");
      if (!option) return;

      const label = candidate.closest("label");
      if (label) (label as HTMLElement).style.display = "none";
      const article = candidate.closest("article");
      const articleHeading = normalized(article?.querySelector("h2")?.textContent);
      if (article && (HIDDEN_CONTEXT_HEADINGS.has(articleHeading) || /choose|select|open/.test(articleHeading))) {
        (article as HTMLElement).style.display = "none";
      }

      for (const button of Array.from(container.querySelectorAll("button"))) {
        const text = normalized(button.textContent);
        if (["refresh tournaments", "refresh list", "load selected", "load tournament"].includes(text)) {
          (button as HTMLElement).style.display = "none";
        }
      }

      if (candidate.value === option.value || Date.now() - lastDispatchRef.current < 500) return;
      lastDispatchRef.current = Date.now();
      const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
      if (setter) setter.call(candidate, option.value);
      else candidate.value = option.value;
      candidate.dispatchEvent(new Event("change", { bubbles: true }));
    }

    const container = rootRef.current;
    if (!container) return;
    applySelectedTournament();
    const observer = new MutationObserver(applySelectedTournament);
    observer.observe(container, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [tournamentId, tournamentName]);

  return <div ref={rootRef} data-selected-tournament={tournamentId}>{children}</div>;
}

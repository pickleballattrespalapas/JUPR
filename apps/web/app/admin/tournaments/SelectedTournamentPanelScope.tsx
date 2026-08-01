"use client";

import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

type Props = {
  tournamentId: string;
  tournamentName?: string | null;
  children: ReactNode;
};

const HIDDEN_CONTEXT_HEADINGS = new Set([
  "admin session",
  "admin session required",
  "tournament admin session",
  "registration reporting session",
  "bulk registration actions",
  "create or open a tournament",
  "1. create tournament shell",
  "2. select tournament",
  "choose a tournament",
  "select tournament",
  "tournament selection",
  "tournament"
]);
const TRANSIENT_TEXT = [
  "checking admin session",
  "checking the saved admin session",
  "restoring admin access",
  "loading tournaments",
  "refreshing",
  "choose tournament",
  "choose a tournament"
];

function normalized(value?: string | null): string {
  return String(value || "").trim().toLowerCase();
}

function matchingTournamentOption(
  select: HTMLSelectElement,
  tournamentId: string,
  tournamentName: string
): HTMLOptionElement | null {
  const options = Array.from(select.options);
  return (
    options.find((option) => option.value === tournamentId) ||
    options.find(
      (option) =>
        tournamentName &&
        normalized(option.textContent).includes(normalized(tournamentName))
    ) ||
    null
  );
}

function preserveTournamentContext(
  anchor: HTMLAnchorElement,
  tournamentId: string,
  tournamentName: string
) {
  const rawHref = anchor.getAttribute("href") || "";
  if (
    !rawHref.startsWith("/admin/tournaments") &&
    !rawHref.startsWith("/admin/tournament-setup") &&
    !rawHref.startsWith("/admin/tournament-live")
  ) {
    return;
  }
  const url = new URL(rawHref, window.location.origin);
  if (!url.searchParams.get("tournament")) {
    url.searchParams.set("tournament", tournamentId);
  }
  if (tournamentName && !url.searchParams.get("name")) {
    url.searchParams.set("name", tournamentName);
  }
  anchor.setAttribute("href", `${url.pathname}${url.search}${url.hash}`);
}

function hideEmbeddedSessionSummary(container: HTMLDivElement) {
  for (const strong of Array.from(
    container.querySelectorAll<HTMLElement>("strong")
  )) {
    if (!normalized(strong.textContent).startsWith("admin session:")) continue;
    const summary = strong.parentElement;
    if (summary) summary.style.display = "none";
  }
}

function hasVisibleTransientText(container: HTMLDivElement): boolean {
  const text = normalized(container.textContent);
  return TRANSIENT_TEXT.some((value) => text.includes(value));
}

export default function SelectedTournamentPanelScope({
  tournamentId,
  tournamentName,
  children
}: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const lastDispatchRef = useRef(0);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setReady(false);
    let fallbackTimer: number | null = null;

    function applySelectedTournament() {
      const container = rootRef.current;
      if (!container) return;

      for (const anchor of Array.from(
        container.querySelectorAll<HTMLAnchorElement>("a[href]")
      )) {
        preserveTournamentContext(anchor, tournamentId, tournamentName || "");
      }
      hideEmbeddedSessionSummary(container);

      for (const article of Array.from(
        container.querySelectorAll<HTMLElement>("article")
      )) {
        const heading = article.querySelector("h2");
        const headingText = normalized(heading?.textContent);
        if (HIDDEN_CONTEXT_HEADINGS.has(headingText)) {
          article.style.display = "none";
        }
      }

      const candidate = Array.from(
        container.querySelectorAll<HTMLSelectElement>("select")
      ).find((select) => {
        const aria = normalized(select.getAttribute("aria-label"));
        const label = normalized(select.closest("label")?.textContent);
        const looksLikeTournament =
          aria.includes("tournament") || label.includes("tournament");
        return (
          looksLikeTournament &&
          Boolean(
            matchingTournamentOption(
              select,
              tournamentId,
              tournamentName || ""
            )
          )
        );
      });

      if (candidate) {
        const option = matchingTournamentOption(
          candidate,
          tournamentId,
          tournamentName || ""
        );
        if (option) {
          const label = candidate.closest("label");
          if (label) (label as HTMLElement).style.display = "none";
          const article = candidate.closest("article");
          const articleHeading = normalized(
            article?.querySelector("h2")?.textContent
          );
          if (
            article &&
            (HIDDEN_CONTEXT_HEADINGS.has(articleHeading) ||
              /choose|select|open/.test(articleHeading))
          ) {
            (article as HTMLElement).style.display = "none";
          }

          if (
            candidate.value !== option.value &&
            Date.now() - lastDispatchRef.current >= 500
          ) {
            lastDispatchRef.current = Date.now();
            const setter = Object.getOwnPropertyDescriptor(
              HTMLSelectElement.prototype,
              "value"
            )?.set;
            if (setter) setter.call(candidate, option.value);
            else candidate.value = option.value;
            candidate.dispatchEvent(new Event("change", { bubbles: true }));
          }
        }
      }

      for (const button of Array.from(
        container.querySelectorAll<HTMLButtonElement>("button")
      )) {
        const text = normalized(button.textContent);
        if (
          [
            "refresh tournaments",
            "refresh list",
            "load selected",
            "load tournament"
          ].includes(text)
        ) {
          button.style.display = "none";
        }
      }

      if (!hasVisibleTransientText(container)) setReady(true);
    }

    const container = rootRef.current;
    if (!container) return;
    applySelectedTournament();
    const observer = new MutationObserver(applySelectedTournament);
    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true
    });
    fallbackTimer = window.setTimeout(() => {
      applySelectedTournament();
      setReady(true);
    }, 2200);
    return () => {
      observer.disconnect();
      if (fallbackTimer != null) window.clearTimeout(fallbackTimer);
    };
  }, [tournamentId, tournamentName]);

  return (
    <div data-selected-tournament={tournamentId}>
      {!ready ? (
        <article
          role="status"
          aria-live="polite"
          style={{
            border: "1px solid #dbeafe",
            borderRadius: "14px",
            padding: "1rem",
            background: "#eff6ff",
            color: "#334155"
          }}
        >
          Loading {tournamentName || "tournament"}…
        </article>
      ) : null}
      <div ref={rootRef} style={{ display: ready ? "block" : "none" }}>
        {children}
      </div>
    </div>
  );
}

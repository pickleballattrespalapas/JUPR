"use client";

import { useEffect, useRef } from "react";

type MatchPayload = {
  league?: string;
  week_tag?: string;
  rating_scope?: string;
  [key: string]: unknown;
};

type BatchPayload = {
  source?: string;
  matches?: MatchPayload[];
  [key: string]: unknown;
};

function labelWithText(root: ParentNode, text: string): HTMLLabelElement | null {
  return Array.from(root.querySelectorAll("label")).find((label) => {
    const strong = label.querySelector("strong");
    return strong?.textContent?.trim() === text;
  }) ?? null;
}

function controlInLabel<T extends HTMLInputElement | HTMLSelectElement>(
  root: ParentNode,
  text: string,
  selector: string,
): T | null {
  return labelWithText(root, text)?.querySelector<T>(selector) ?? null;
}

function articleWithHeading(root: ParentNode, heading: string): HTMLElement | null {
  return Array.from(root.querySelectorAll<HTMLElement>("article")).find(
    (article) => article.querySelector("h2")?.textContent?.trim() === heading,
  ) ?? null;
}

function manualMatchCards(root: ParentNode): HTMLElement[] {
  const article = articleWithHeading(root, "Doubles manual / batch score entry");
  if (!article) return [];
  const seen = new Set<HTMLElement>();
  const cards: HTMLElement[] = [];
  for (const section of article.querySelectorAll<HTMLElement>('section[aria-label$="Team 1"]')) {
    const card = section.parentElement?.parentElement;
    if (card && !seen.has(card)) {
      seen.add(card);
      cards.push(card);
    }
  }
  return cards;
}

function metadataGrid(card: HTMLElement): HTMLElement | null {
  return Array.from(card.children).find((child) => {
    if (!(child instanceof HTMLElement)) return false;
    return Boolean(labelWithText(child, "Date") && labelWithText(child, "Rating scope"));
  }) as HTMLElement | null;
}

function officialLeagueOptions(setup: HTMLElement): string[] {
  const select = controlInLabel<HTMLSelectElement>(setup, "Default league", "select");
  const values = Array.from(select?.options ?? [])
    .map((option) => option.value.trim())
    .filter((value) => value && value.toUpperCase() !== "POPUP");
  return values.length ? values : ["Open"];
}

function dispatchSelectChange(select: HTMLSelectElement, value: string) {
  if (select.value === value) return;
  select.value = value;
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

function ensureLeagueSelector(
  grid: HTMLElement,
  defaultLeague: string,
  options: string[],
): HTMLSelectElement {
  let label = grid.querySelector<HTMLLabelElement>("label[data-mu-row-league]");
  if (!label) {
    label = document.createElement("label");
    label.dataset.muRowLeague = "true";
    const strong = document.createElement("strong");
    strong.textContent = "League";
    const br = document.createElement("br");
    const select = document.createElement("select");
    select.setAttribute("aria-label", "League");
    label.append(strong, br, select);
    const weekLabel = labelWithText(grid, "Week / session");
    grid.insertBefore(label, weekLabel ?? labelWithText(grid, "Rating scope"));
  }
  const select = label.querySelector<HTMLSelectElement>("select")!;
  const priorValue = select.value;
  const signature = options.join("\u0000");
  if (select.dataset.muOptions !== signature) {
    select.replaceChildren(...options.map((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      return option;
    }));
    select.dataset.muOptions = signature;
  }
  select.value = options.includes(priorValue) ? priorValue : defaultLeague;
  return select;
}

function ensurePopupScopeSelector(
  grid: HTMLElement,
  original: HTMLSelectElement,
): HTMLSelectElement {
  let label = grid.querySelector<HTMLLabelElement>("label[data-mu-popup-scope]");
  if (!label) {
    label = document.createElement("label");
    label.dataset.muPopupScope = "true";
    const strong = document.createElement("strong");
    strong.textContent = "Rating scope";
    const br = document.createElement("br");
    const select = document.createElement("select");
    select.setAttribute("aria-label", "Pop-Up rating scope");
    for (const [value, text] of [
      ["overall_only", "Overall only (rated)"],
      ["unrated", "Unrated / record only"],
    ] as const) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      select.append(option);
    }
    select.addEventListener("change", () => dispatchSelectChange(original, select.value));
    label.append(strong, br, select);
    grid.append(label);
  }
  const select = label.querySelector<HTMLSelectElement>("select")!;
  select.value = original.value === "unrated" ? "unrated" : "overall_only";
  dispatchSelectChange(original, select.value);
  return select;
}

function selectedPlayerCount(card: HTMLElement): number {
  return card.querySelectorAll('button[aria-label^="Clear Player"]').length;
}

function scoreTotal(card: HTMLElement): number {
  const scores = Array.from(card.querySelectorAll<HTMLInputElement>('section[aria-label*=" Team "] > label:last-child input[type="number"]'));
  return scores.reduce((total, input) => total + (Number(input.value) || 0), 0);
}

function readyManualCards(root: ParentNode): HTMLElement[] {
  return manualMatchCards(root).filter(
    (card) => selectedPlayerCount(card) === 4 && scoreTotal(card) > 0,
  );
}

export default function MatchUploaderAcceptanceEnhancer() {
  const anchorRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const root = anchorRef.current?.parentElement;
    if (!root) return;

    let scheduled = false;
    const apply = () => {
      scheduled = false;
      const setup = articleWithHeading(root, "Match entry setup");
      if (!setup) return;
      const contextSelect = controlInLabel<HTMLSelectElement>(setup, "Context", "select");
      if (!contextSelect) return;
      const context = contextSelect.value === "popup" ? "popup" : "league";
      root.dataset.muContext = context;

      const defaultLeagueLabel = labelWithText(setup, "Default league");
      const defaultWeekLabel = labelWithText(setup, "Default week/session");
      if (defaultLeagueLabel) defaultLeagueLabel.hidden = context === "popup";
      if (defaultWeekLabel) defaultWeekLabel.hidden = context === "popup";

      const options = officialLeagueOptions(setup);
      const defaultLeagueSelect = controlInLabel<HTMLSelectElement>(setup, "Default league", "select");
      const defaultLeague = defaultLeagueSelect?.value || options[0];
      if (defaultLeagueSelect && !defaultLeagueSelect.dataset.muLeagueSync) {
        defaultLeagueSelect.dataset.muLeagueSync = "true";
        defaultLeagueSelect.dataset.muPriorLeague = defaultLeagueSelect.value;
        defaultLeagueSelect.addEventListener("change", () => {
          const prior = defaultLeagueSelect.dataset.muPriorLeague ?? "";
          for (const select of root.querySelectorAll<HTMLSelectElement>("select[data-mu-row-league-control]")) {
            if (!select.value || select.value === prior) select.value = defaultLeagueSelect.value;
          }
          defaultLeagueSelect.dataset.muPriorLeague = defaultLeagueSelect.value;
        });
      }

      for (const card of manualMatchCards(root)) {
        card.dataset.muMatchCard = "true";
        const previousContext = card.dataset.muContext;
        card.dataset.muContext = context;
        const grid = metadataGrid(card);
        const team1 = card.querySelector<HTMLElement>('section[aria-label$="Team 1"]');
        const team2 = card.querySelector<HTMLElement>('section[aria-label$="Team 2"]');
        const teams = team1?.parentElement;
        if (grid) {
          grid.dataset.muMetaGrid = "true";
          grid.dataset.muContext = context;
          const weekLabel = labelWithText(grid, "Week / session");
          const ratingLabel = labelWithText(grid, "Rating scope");
          const ratingSelect = ratingLabel?.querySelector<HTMLSelectElement>("select");
          if (weekLabel) weekLabel.hidden = context === "popup";
          if (ratingLabel && ratingSelect) {
            if (context === "popup") {
              ratingLabel.hidden = true;
              if (previousContext !== "popup") {
                dispatchSelectChange(ratingSelect, ratingSelect.value === "unrated" ? "unrated" : "overall_only");
              }
              ensurePopupScopeSelector(grid, ratingSelect);
            } else {
              ratingLabel.hidden = false;
              grid.querySelector("label[data-mu-popup-scope]")?.remove();
              if (previousContext === "popup") {
                dispatchSelectChange(ratingSelect, ratingSelect.value === "unrated" ? "unrated" : "");
              }
            }
          }
          const leagueLabel = grid.querySelector<HTMLLabelElement>("label[data-mu-row-league]");
          if (context === "league") {
            const select = ensureLeagueSelector(grid, defaultLeague, options);
            select.dataset.muRowLeagueControl = "true";
            if (leagueLabel) leagueLabel.hidden = false;
          } else if (leagueLabel) {
            leagueLabel.hidden = true;
          }
        }
        if (teams) teams.dataset.muTeamGrid = "true";
        if (team1) {
          team1.dataset.muTeam = "1";
          const strong = team1.querySelector<HTMLLabelElement>(":scope > label:last-child")?.querySelector("strong");
          if (strong) strong.textContent = "Team 1 score";
        }
        if (team2) {
          team2.dataset.muTeam = "2";
          const strong = team2.querySelector<HTMLLabelElement>(":scope > label:last-child")?.querySelector("strong");
          if (strong) strong.textContent = "Team 2 score";
        }
      }
    };

    const scheduleApply = () => {
      if (scheduled) return;
      scheduled = true;
      window.requestAnimationFrame(apply);
    };

    const observer = new MutationObserver(scheduleApply);
    observer.observe(root, { childList: true, subtree: true });
    root.addEventListener("change", scheduleApply);
    scheduleApply();

    const originalFetch = window.fetch.bind(window);
    const enhancedFetch: typeof window.fetch = async (input, init) => {
      const url = typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
      if (url.includes("/match-uploader/batch") && typeof init?.body === "string") {
        try {
          const payload = JSON.parse(init.body) as BatchPayload;
          if (Array.isArray(payload.matches)) {
            const setup = articleWithHeading(root, "Match entry setup");
            const context = setup
              ? controlInLabel<HTMLSelectElement>(setup, "Context", "select")?.value
              : "league";
            if (context === "popup") {
              payload.matches = payload.matches.map((match) => ({
                ...match,
                league: "POPUP",
                week_tag: "",
                rating_scope: match.rating_scope === "unrated" ? "unrated" : "overall_only",
              }));
            } else if (payload.source === "next_match_uploader_manual_batch") {
              const readyCards = readyManualCards(root);
              payload.matches = payload.matches.map((match, index) => ({
                ...match,
                league: readyCards[index]?.querySelector<HTMLSelectElement>("select[data-mu-row-league-control]")?.value
                  || match.league,
              }));
            }
            init = { ...init, body: JSON.stringify(payload) };
          }
        } catch {
          // Preserve the original request if it is not JSON.
        }
      }
      return originalFetch(input, init);
    };
    window.fetch = enhancedFetch;

    return () => {
      observer.disconnect();
      root.removeEventListener("change", scheduleApply);
      if (window.fetch === enhancedFetch) window.fetch = originalFetch;
    };
  }, []);

  return <span ref={anchorRef} hidden aria-hidden="true" />;
}

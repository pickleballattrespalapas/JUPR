from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    next_text, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{label}: expected one regex match, found {count}")
    return next_text


component = r'''"use client";

import { useEffect, useMemo, useState } from "react";

export type GeneratorKind = "round_robin" | "ladder";
export type PlayFormat = "singles" | "doubles";
export type LinkedPlayerIds = Record<string, number>;

type DirectoryPlayer = {
  id: number | string;
  name: string;
  is_active?: boolean | null;
};

type DirectoryResponse = {
  players?: DirectoryPlayer[];
};

type Props = {
  apiBase: string | null;
  clubKey: string;
  generatorKind: GeneratorKind;
  playFormat: PlayFormat;
  targetCount: number;
  participantText: string;
  linkedPlayerIds: LinkedPlayerIds;
  onTargetCountChange: (count: number) => void;
  onParticipantTextChange: (text: string) => void;
  onLinkedPlayerIdsChange: (links: LinkedPlayerIds) => void;
  onInvalidate: () => void;
};

const inputStyle = {
  width: "100%",
  boxSizing: "border-box" as const,
  padding: "0.6rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

const secondaryButton = {
  border: "1px solid #cbd5e1",
  borderRadius: "999px",
  padding: "0.4rem 0.75rem",
  background: "white",
  color: "#0f172a",
  fontWeight: 800,
  cursor: "pointer"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export function normalizeRosterName(value: string): string {
  return String(value || "").trim().replace(/\s+/g, " ").toLowerCase();
}

export function rosterNamesFromText(text: string): string[] {
  const names: string[] = [];
  const seen = new Set<string>();
  for (const rawName of String(text || "").split(/\r?\n|,/)) {
    const name = rawName.trim().replace(/\s+/g, " ");
    if (!name) continue;
    const key = normalizeRosterName(name);
    if (seen.has(key)) continue;
    seen.add(key);
    names.push(name);
  }
  return names;
}

function ladderCourtCount(playerCount: number, playFormat: PlayFormat): number {
  const minimum = playFormat === "singles" ? 2 : 4;
  if (playerCount < minimum) return 0;
  if (playFormat === "doubles") {
    let best = 0;
    for (let fives = 0; fives <= Math.floor(playerCount / 5); fives += 1) {
      const remainder = playerCount - fives * 5;
      if (remainder >= 0 && remainder % 4 === 0) {
        const groups = fives + remainder / 4;
        best = Math.max(best, groups);
      }
    }
    if (best > 0) return best;
  }
  return Math.min(
    Math.max(1, Math.ceil(playerCount / 5)),
    Math.max(1, Math.floor(playerCount / minimum))
  );
}

export function recommendedGeneratorSetup(
  generatorKind: GeneratorKind,
  playFormat: PlayFormat,
  playerCount: number
): { courtCount: number; totalRounds: number } {
  const count = Math.max(0, Math.min(40, Math.floor(Number(playerCount) || 0)));
  const minimum = playFormat === "singles" ? 2 : 4;
  if (count < minimum) {
    return { courtCount: 0, totalRounds: generatorKind === "ladder" ? 4 : 1 };
  }
  if (generatorKind === "ladder") {
    return {
      courtCount: ladderCourtCount(count, playFormat),
      totalRounds: 4
    };
  }
  const courtCount = Math.max(1, Math.floor(count / minimum));
  if (playFormat === "singles") {
    return {
      courtCount,
      totalRounds: count % 2 === 0 ? count - 1 : count
    };
  }
  const uniquePartnerPairs = (count * (count - 1)) / 2;
  const partnerPairsPerRound = courtCount * 2;
  return {
    courtCount,
    totalRounds: Math.min(50, Math.max(1, Math.ceil(uniquePartnerPairs / partnerPairsPerRound)))
  };
}

function explanation(
  generatorKind: GeneratorKind,
  playFormat: PlayFormat,
  targetCount: number,
  courtCount: number,
  totalRounds: number
): string {
  if (generatorKind === "ladder") {
    return `${targetCount} players create ${courtCount} balanced ladder court${courtCount === 1 ? "" : "s"} and ${totalRounds} result-driven rounds.`;
  }
  if (playFormat === "singles") {
    return `${targetCount} players create ${courtCount} court${courtCount === 1 ? "" : "s"} and a complete ${totalRounds}-round singles rotation.`;
  }
  return `${targetCount} players create ${courtCount} court${courtCount === 1 ? "" : "s"} and ${totalRounds} rounds for a complete partner rotation.`;
}

export default function GeneratorRosterSetup({
  apiBase,
  clubKey,
  generatorKind,
  playFormat,
  targetCount,
  participantText,
  linkedPlayerIds,
  onTargetCountChange,
  onParticipantTextChange,
  onLinkedPlayerIdsChange,
  onInvalidate
}: Props) {
  const [playerSearch, setPlayerSearch] = useState("");
  const [directoryPlayers, setDirectoryPlayers] = useState<DirectoryPlayer[]>([]);
  const [directoryError, setDirectoryError] = useState<string | null>(null);
  const [pickerMessage, setPickerMessage] = useState<string | null>(null);

  const participantNames = useMemo(
    () => rosterNamesFromText(participantText),
    [participantText]
  );
  const participantNameSet = useMemo(
    () => new Set(participantNames.map(normalizeRosterName)),
    [participantNames]
  );
  const minimumPlayers = playFormat === "singles" ? 2 : 4;
  const participantCounts = useMemo(
    () => Array.from({ length: 40 - minimumPlayers + 1 }, (_, index) => minimumPlayers + index),
    [minimumPlayers]
  );
  const setup = useMemo(
    () => recommendedGeneratorSetup(generatorKind, playFormat, targetCount),
    [generatorKind, playFormat, targetCount]
  );
  const linkedCount = participantNames.filter(
    (name) => Number(linkedPlayerIds[normalizeRosterName(name)] || 0) > 0
  ).length;
  const exactCount = participantNames.length === targetCount;
  const publicClubSlug = clubKey.replace(/_/g, "-");

  useEffect(() => {
    if (!apiBase) return;
    let cancelled = false;
    setDirectoryError(null);
    void fetch(
      apiUrl(
        apiBase,
        `/clubs/${encodeURIComponent(publicClubSlug)}/players?status=active&sort=name&limit=1000`
      ),
      { cache: "no-store" }
    )
      .then(async (response) => {
        const payload = (await response.json().catch(() => null)) as DirectoryResponse | null;
        if (!response.ok) {
          throw new Error(String((payload as { detail?: string } | null)?.detail || `API error (${response.status})`));
        }
        if (cancelled) return;
        const rows = [...(payload?.players || [])]
          .filter((player) => player && player.is_active !== false && String(player.name || "").trim())
          .sort((left, right) => String(left.name).localeCompare(String(right.name)));
        setDirectoryPlayers(rows);
      })
      .catch((error) => {
        if (cancelled) return;
        setDirectoryPlayers([]);
        setDirectoryError(error instanceof Error ? error.message : "Current-player search is unavailable.");
      });
    return () => {
      cancelled = true;
    };
  }, [apiBase, publicClubSlug]);

  const filteredPlayerOptions = useMemo(() => {
    const query = normalizeRosterName(playerSearch);
    if (query.length < 2) return [];
    return directoryPlayers
      .filter((player) => normalizeRosterName(player.name).includes(query))
      .slice(0, 10);
  }, [directoryPlayers, playerSearch]);

  function changeTargetCount(nextCount: number): void {
    onTargetCountChange(nextCount);
    onInvalidate();
    setPickerMessage(null);
  }

  function changeParticipantText(nextText: string): void {
    const nextNames = rosterNamesFromText(nextText);
    const nextNameSet = new Set(nextNames.map(normalizeRosterName));
    const nextLinks = Object.fromEntries(
      Object.entries(linkedPlayerIds).filter(([key]) => nextNameSet.has(key))
    );
    onParticipantTextChange(nextText);
    onLinkedPlayerIdsChange(nextLinks);
    onInvalidate();
    setPickerMessage(null);
  }

  function addCurrentPlayer(player: DirectoryPlayer): void {
    const name = String(player.name || "").trim().replace(/\s+/g, " ");
    const key = normalizeRosterName(name);
    if (!name || participantNameSet.has(key)) return;
    if (participantNames.length >= targetCount) {
      setPickerMessage(`The roster already has the selected ${targetCount} players. Increase the player count or remove a name first.`);
      return;
    }
    const nextNames = [...participantNames, name];
    const playerId = Number(player.id);
    onParticipantTextChange(nextNames.join("\n"));
    if (Number.isFinite(playerId) && playerId > 0) {
      onLinkedPlayerIdsChange({ ...linkedPlayerIds, [key]: playerId });
    }
    onInvalidate();
    setPlayerSearch("");
    setPickerMessage(`${name} added to the roster.`);
  }

  const countMessage = exactCount
    ? `Ready with ${targetCount} players.`
    : participantNames.length < targetCount
      ? `Add ${targetCount - participantNames.length} more player${targetCount - participantNames.length === 1 ? "" : "s"}.`
      : `Remove ${participantNames.length - targetCount} player${participantNames.length - targetCount === 1 ? "" : "s"} or increase the selected count.`;

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "0.75rem"
        }}
      >
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Number of players
          <select
            value={targetCount}
            onChange={(event) => changeTargetCount(Number(event.target.value))}
            style={inputStyle}
          >
            {participantCounts.map((count) => (
              <option key={count} value={count}>{count}</option>
            ))}
          </select>
        </label>
        <div
          aria-live="polite"
          style={{
            border: "1px solid #bfdbfe",
            borderRadius: "10px",
            padding: "0.75rem",
            background: "#eff6ff"
          }}
        >
          <strong>Automatic setup</strong>
          <p style={{ margin: "0.3rem 0 0", color: "#334155" }}>
            {setup.courtCount} court{setup.courtCount === 1 ? "" : "s"} · {setup.totalRounds} round{setup.totalRounds === 1 ? "" : "s"}
          </p>
          <small style={{ color: "#475569" }}>
            {explanation(generatorKind, playFormat, targetCount, setup.courtCount, setup.totalRounds)}
          </small>
        </div>
      </div>

      <div style={{ display: "grid", gap: "0.5rem" }}>
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Search current players
          <input
            value={playerSearch}
            onChange={(event) => {
              setPlayerSearch(event.target.value);
              setPickerMessage(null);
            }}
            placeholder="Type at least 2 letters, then add a player"
            style={inputStyle}
          />
        </label>
        {playerSearch.trim().length < 2 ? (
          <p style={{ margin: 0, color: "#64748b", fontSize: "0.9rem" }}>
            Search the current player list, or type guest names directly in the roster box below.
          </p>
        ) : filteredPlayerOptions.length ? (
          <div style={{ display: "grid", gap: "0.4rem" }}>
            {filteredPlayerOptions.map((player) => {
              const alreadyAdded = participantNameSet.has(normalizeRosterName(player.name));
              const rosterFull = participantNames.length >= targetCount;
              return (
                <div
                  key={String(player.id)}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: "0.75rem",
                    alignItems: "center",
                    border: "1px solid #cbd5e1",
                    borderRadius: "10px",
                    padding: "0.5rem",
                    background: "white"
                  }}
                >
                  <span>{player.name}</span>
                  <button
                    type="button"
                    onClick={() => addCurrentPlayer(player)}
                    disabled={alreadyAdded || rosterFull}
                    style={{
                      ...secondaryButton,
                      background: alreadyAdded || rosterFull ? "#f1f5f9" : "white",
                      cursor: alreadyAdded || rosterFull ? "default" : "pointer"
                    }}
                  >
                    {alreadyAdded ? "Added" : rosterFull ? "Roster full" : "Add"}
                  </button>
                </div>
              );
            })}
          </div>
        ) : (
          <p style={{ margin: 0, color: "#b45309", fontSize: "0.9rem" }}>
            No current players match “{playerSearch.trim()}”. Type a guest name in the roster box below.
          </p>
        )}
        {directoryError ? (
          <p style={{ margin: 0, color: "#b45309", fontSize: "0.9rem" }}>
            Current-player search is unavailable. Guest roster entry still works. {directoryError}
          </p>
        ) : null}
      </div>

      <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
        Names or roster entry ({participantNames.length} of {targetCount})
        <textarea
          value={participantText}
          onChange={(event) => changeParticipantText(event.target.value)}
          rows={Math.min(14, Math.max(7, targetCount))}
          placeholder="One player per line, in starting order"
          style={inputStyle}
        />
      </label>
      <div>
        <p style={{ margin: 0, color: exactCount ? "#166534" : "#b45309", fontWeight: 700 }}>
          {countMessage}
        </p>
        <p style={{ margin: "0.25rem 0 0", color: "#64748b", fontSize: "0.9rem" }}>
          {linkedCount} selected from the current player list · {participantNames.length - linkedCount} guest or unlinked.
          Roster line order controls starting order and bye priority.
        </p>
        {pickerMessage ? (
          <p role="status" aria-live="polite" style={{ margin: "0.25rem 0 0", color: "#1d4ed8" }}>
            {pickerMessage}
          </p>
        ) : null}
      </div>
    </section>
  );
}
'''
write(WEB / "components" / "GeneratorRosterSetup.tsx", component)


def patch_workspace(path: Path, *, public: bool) -> None:
    text = path.read_text(encoding="utf-8")
    import_anchor = 'import { useRouter } from "next/navigation";\n'
    import_line = (
        'import GeneratorRosterSetup, { normalizeRosterName, recommendedGeneratorSetup, rosterNamesFromText } from "@/components/GeneratorRosterSetup";\n'
    )
    text = replace_once(text, import_anchor, import_anchor + import_line, f"{path.name} roster import")

    text = regex_once(
        text,
        r'''\ntype PlayerDraft = \{\n  key: string;\n  name: string;\n  playerId: string;\n\};\n''',
        "\n",
        f"{path.name} PlayerDraft type",
    )
    text = regex_once(
        text,
        r'''\nfunction initialPlayers\(count = 4\): PlayerDraft\[\] \{.*?\n\}\n\nfunction flattenMatches''',
        "\nfunction flattenMatches",
        f"{path.name} initialPlayers",
    )

    old_state = '''  const [playFormat, setPlayFormat] = useState<PlayFormat>("doubles");
  const [totalRounds, setTotalRounds] = useState(generatorKind === "ladder" ? "4" : "6");
  const [courtCount, setCourtCount] = useState("0");
  const [players, setPlayers] = useState<PlayerDraft[]>(() => initialPlayers(4));
  const [preview, setPreview] = useState<PreviewEvent | null>(null);'''
    new_state = '''  const [playFormat, setPlayFormat] = useState<PlayFormat>("doubles");
  const [targetCount, setTargetCount] = useState(8);
  const [participantText, setParticipantText] = useState("");
  const [linkedPlayerIds, setLinkedPlayerIds] = useState<Record<string, number>>({});
  const [preview, setPreview] = useState<PreviewEvent | null>(null);'''
    text = replace_once(text, old_state, new_state, f"{path.name} setup state")

    old_clean = '''  const cleanRows = useMemo(
    () => players.filter((row) => row.name.trim()),
    [players]
  );'''
    new_clean = '''  const participantNames = useMemo(
    () => rosterNamesFromText(participantText),
    [participantText]
  );
  const automaticSetup = useMemo(
    () => recommendedGeneratorSetup(generatorKind, playFormat, targetCount),
    [generatorKind, playFormat, targetCount]
  );
  const rosterReady = participantNames.length === targetCount;'''
    text = replace_once(text, old_clean, new_clean, f"{path.name} participant memo")

    text = regex_once(
        text,
        r'''\n  function updatePlayer\(key: string, patch: Partial<PlayerDraft>\): void \{.*?\n  function requestBody\(\): Record<string, unknown> \{.*?\n  \}\n\n  async function generatePreview''',
        "\n  function requestBody(): Record<string, unknown> {\n"
        "    if (!rosterReady) {\n"
        "      throw new Error(`Add exactly ${targetCount} unique players before previewing.`);\n"
        "    }\n"
        + (
            "    const participantPlayerIds = Object.fromEntries(\n"
            "      participantNames.flatMap((name) => {\n"
            "        const playerId = Number(linkedPlayerIds[normalizeRosterName(name)] || 0);\n"
            "        return playerId > 0 ? [[name, playerId]] : [];\n"
            "      })\n"
            "    );\n"
            "    return {\n"
            "      generator_kind: generatorKind,\n"
            "      play_format: playFormat,\n"
            "      title: title.trim(),\n"
            "      participant_names: participantNames,\n"
            "      participant_player_ids: participantPlayerIds,\n"
            "      total_rounds: automaticSetup.totalRounds,\n"
            "      court_count: automaticSetup.courtCount\n"
            "    };\n"
            if public
            else
            "    const orderedIds = participantNames.map((name) =>\n"
            "      Number(linkedPlayerIds[normalizeRosterName(name)] || 0)\n"
            "    );\n"
            "    const allLinked = orderedIds.every((playerId) => playerId > 0);\n"
            "    return {\n"
            "      generator_kind: generatorKind,\n"
            "      play_format: playFormat,\n"
            "      title: title.trim(),\n"
            "      participant_names: participantNames,\n"
            "      player_ids: allLinked ? orderedIds : [],\n"
            "      total_rounds: automaticSetup.totalRounds,\n"
            "      court_count: automaticSetup.courtCount\n"
            "    };\n"
        )
        + "  }\n\n  async function generatePreview",
        f"{path.name} roster functions",
    )

    text = replace_once(
        text,
        '''              onChange={(event) => {
                setPlayFormat(event.target.value as PlayFormat);
                invalidatePreview();
              }}''',
        '''              onChange={(event) => {
                const nextFormat = event.target.value as PlayFormat;
                setPlayFormat(nextFormat);
                setTargetCount((current) => Math.max(current, nextFormat === "singles" ? 2 : 4));
                invalidatePreview();
              }}''',
        f"{path.name} format handler",
    )

    text = replace_once(
        text,
        '''  const minimumPlayers = playFormat === "singles" ? 2 : 4;
  const previewParticipants = preview ? participantMap(preview) : new Map<string, Participant>();''',
        '''  const previewParticipants = preview ? participantMap(preview) : new Map<string, Participant>();''',
        f"{path.name} minimumPlayers",
    )

    setup_block = '''        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
            gap: "0.75rem",
            marginBottom: "1rem"
          }}
        >
          <label>
            Session title
            <br />
            <input
              value={title}
              onChange={(event) => {
                setTitle(event.target.value);
                invalidatePreview();
              }}
              style={inputStyle}
            />
          </label>
          <label>
            Play format
            <br />
            <select
              value={playFormat}
              onChange={(event) => {
                const nextFormat = event.target.value as PlayFormat;
                setPlayFormat(nextFormat);
                setTargetCount((current) => Math.max(current, nextFormat === "singles" ? 2 : 4));
                invalidatePreview();
              }}
              style={inputStyle}
            >
              <option value="doubles">Doubles</option>
              <option value="singles">Singles</option>
            </select>
          </label>
        </div>

        <GeneratorRosterSetup
          apiBase={apiBase}
          clubKey={clubId}
          generatorKind={generatorKind}
          playFormat={playFormat}
          targetCount={targetCount}
          participantText={participantText}
          linkedPlayerIds={linkedPlayerIds}
          onTargetCountChange={setTargetCount}
          onParticipantTextChange={setParticipantText}
          onLinkedPlayerIdsChange={setLinkedPlayerIds}
          onInvalidate={invalidatePreview}
        />
        <div style={{ marginTop: "0.9rem", display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={() => void generatePreview()}
            disabled={busy || !rosterReady}
            style={primaryButton}
          >
            {busy ? "Generating…" : "Preview matchups"}
          </button>
        </div>
'''

    text = regex_once(
        text,
        r'''        <div\n          style=\{\{\n            display: "grid",\n            gridTemplateColumns: "repeat\(auto-fit, minmax\(190px, 1fr\)\)",.*?        </div>\n        \{message \? \(''',
        setup_block + "        {message ? (",
        f"{path.name} setup UI",
    )

    forbidden = ["Planned rounds", "Available courts", "Club player ID optional", "Add player"]
    for token in forbidden:
        if token in text:
            raise RuntimeError(f"{path.name}: stale setup token remains: {token}")
    for token in ["GeneratorRosterSetup", "targetCount", "automaticSetup", "rosterReady"]:
        if token not in text:
            raise RuntimeError(f"{path.name}: required token missing: {token}")
    path.write_text(text, encoding="utf-8")


patch_workspace(
    WEB / "app" / "clubs" / "[clubSlug]" / "play-generators" / "PublicGeneratorWorkspace.tsx",
    public=True,
)
patch_workspace(
    WEB / "app" / "admin" / "play-generators" / "GeneratorWorkspace.tsx",
    public=False,
)

write(
    ROOT / "tests" / "test_play_generator_player_count_setup.py",
    r'''from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_generator_setup_uses_player_count_and_previous_picker_pattern() -> None:
    component = read("apps/web/components/GeneratorRosterSetup.tsx")
    public = read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx")
    admin = read("apps/web/app/admin/play-generators/GeneratorWorkspace.tsx")

    for text in (public, admin):
        assert "GeneratorRosterSetup" in text
        assert "targetCount" in text
        assert "automaticSetup.totalRounds" in text
        assert "automaticSetup.courtCount" in text
        assert "Planned rounds" not in text
        assert "Available courts" not in text
        assert "Club player ID optional" not in text

    assert "Number of players" in component
    assert "Automatic setup" in component
    assert "Search current players" in component
    assert "Type at least 2 letters, then add a player" in component
    assert "Names or roster entry" in component
    assert "One player per line, in starting order" in component
    assert "Roster line order controls starting order and bye priority" in component


def test_generator_setup_has_deterministic_auto_shape_rules() -> None:
    component = read("apps/web/components/GeneratorRosterSetup.tsx")
    assert "count % 2 === 0 ? count - 1 : count" in component
    assert "const uniquePartnerPairs = (count * (count - 1)) / 2" in component
    assert "const partnerPairsPerRound = courtCount * 2" in component
    assert "courtCount: ladderCourtCount(count, playFormat)" in component
    assert "totalRounds: 4" in component
    assert "Math.min(50" in component


def test_admin_preserves_official_links_only_for_complete_linked_roster() -> None:
    admin = read("apps/web/app/admin/play-generators/GeneratorWorkspace.tsx")
    assert "const allLinked = orderedIds.every" in admin
    assert "player_ids: allLinked ? orderedIds : []" in admin


def test_public_passes_selected_player_links_by_name() -> None:
    public = read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx")
    assert "participant_player_ids: participantPlayerIds" in public
    assert "linkedPlayerIds[normalizeRosterName(name)]" in public
''',
)

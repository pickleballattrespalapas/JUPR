from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{label}: expected one regex match, found {count}")
    return updated


def section(text: str, start_marker: str, end_marker: str) -> tuple[str, str, str]:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start], text[start:end], text[end:]


def update_uploader() -> None:
    path = ROOT / "apps/web/app/admin/match-uploader/MatchUploaderForm.tsx"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        'import { useEffect, useState } from "react";',
        'import { useEffect, useRef, useState } from "react";\nimport { ConfirmAction } from "@/components/ConfirmAction";',
        "uploader imports",
    )

    text = sub_once(
        text,
        r"function newMatchRow\(\): MatchRow \{.*?\n\}",
        '''function newMatchRow(
  date: string = todayIsoDate(),
  weekTag: string = "Week 1",
  ratingScope: MatchRow["ratingScope"] = "",
): MatchRow {
  return { rowId: randomId("row"), date, weekTag, ratingScope, t1p1: "", t1p2: "", t2p1: "", t2p2: "", s1: "0", s2: "0" };
}''',
        "parameterized match row",
    )

    text = sub_once(
        text,
        r"(function validateRow\(row: MatchRow, index: number\): string \| null \{.*?\n\})\n\nfunction validateSingles",
        r'''\1

function isReadyRow(row: MatchRow, index: number): boolean {
  return isFilled(row) && validateRow(row, index) === null;
}

function validateSingles''',
        "ready row predicate",
    )

    before, searchable, after = section(
        text,
        "function SearchablePlayerInput({",
        "function SearchablePlayerMultiInput({",
    )
    searchable = replace_once(
        searchable,
        "  const numericStartingJupr = Number(startingJupr);",
        '''  const matchingPlayers = cleanedQuery
    ? players.filter((player) =>
        String(player.name).trim().toLocaleLowerCase().includes(cleanedQuery.toLocaleLowerCase()),
      )
    : players;
  const numericStartingJupr = Number(startingJupr);''',
        "matching players",
    )
    searchable = sub_once(
        searchable,
        r'''      <div style=\{\{ display: "flex", gap: "0\.35rem" \}\}>.*?      </div>\n      <datalist''',
        '''      {value ? (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: "0.35rem", alignItems: "stretch" }}>
          <div
            title={selectedName}
            style={{ ...inputStyle, minHeight: "2.4rem", display: "flex", alignItems: "center", whiteSpace: "normal", overflowWrap: "anywhere", background: "#f8fafc" }}
          >
            {selectedName}
          </div>
          <button
            type="button"
            aria-label={`Clear ${label}`}
            disabled={disabled || creating}
            onClick={() => {
              setQuery("");
              onChange("");
            }}
          >
            Clear
          </button>
        </div>
      ) : (
        <input
          id={inputId}
          list={`${inputId}-options`}
          value={query}
          placeholder="Search player…"
          autoComplete="off"
          disabled={disabled || creating}
          onChange={(event) => {
            const next = event.target.value;
            setQuery(next);
            const match = players.find(
              (player) =>
                String(player.name).trim().toLocaleLowerCase()
                === next.replace(/\\s+/g, " ").trim().toLocaleLowerCase(),
            );
            onChange(match ? String(match.id) : "");
          }}
          style={inputStyle}
        />
      )}
      <datalist''',
        "selected player display",
    )
    searchable = replace_once(
        searchable,
        "      {cleanedQuery && !exactPlayer ? (",
        "      {cleanedQuery && !exactPlayer && matchingPlayers.length === 0 ? (",
        "safe player creation",
    )
    text = before + searchable + after

    dialog_component = r'''
function SubmissionResultDialog({
  result,
  onClose,
}: {
  result: AdminMatchUploaderWriteResult;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
    return () => {
      if (dialog?.open) dialog.close();
    };
  }, []);

  const inserted = result.result?.inserted ?? 0;
  const email = result.auto_player_updates;
  const emailSummary = email?.mode === "auto_sent"
    ? `${email.sent ?? 0} sent, ${email.skipped ?? 0} skipped, ${email.errors ?? 0} error(s).`
    : "Not sent in staging.";

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="match-submission-result-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      style={{ width: "min(720px, calc(100vw - 2rem))", maxHeight: "calc(100vh - 2rem)", overflowY: "auto", border: 0, borderRadius: "16px", padding: 0, boxShadow: "0 24px 70px rgba(15, 23, 42, 0.35)" }}
    >
      <div style={{ padding: "1.25rem" }}>
        <h2 id="match-submission-result-title" style={{ marginTop: 0 }}>Match submission complete</h2>
        <p role="status" style={{ color: "#166534" }}>
          Successfully inserted {inserted} rated match{inserted === 1 ? "" : "es"}.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
          <div><strong>Inserted</strong><br />{inserted}</div>
          <div><strong>Match write</strong><br />{result.match_write_committed ? "Committed" : "Review required"}</div>
          <div><strong>Rating type</strong><br />{result.feedback?.rating_type || result.result?.match_format || "doubles/overall"}</div>
          <div><strong>Skipped incomplete</strong><br />{result.result?.skipped_incomplete ?? 0}</div>
          <div><strong>Skipped empty</strong><br />{result.result?.skipped_empty ?? 0}</div>
          <div><strong>Skipped unrated</strong><br />{result.result?.skipped_unrated ?? 0}</div>
        </div>
        {email ? <p style={{ marginTop: "1rem" }}><strong>Player-update email:</strong> {emailSummary}</p> : null}
        {result.feedback?.affected_players?.length ? (
          <div style={{ overflowX: "auto", marginTop: "1rem" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Before</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>After</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Change</th></tr></thead>
              <tbody>
                {result.feedback.affected_players.map((player) => (
                  <tr key={player.id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{player.name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_before)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_after)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(player.rating_delta)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        {result.warnings?.length ? <ul style={{ color: "#92400e" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
        <p style={{ display: "flex", justifyContent: "flex-end", marginBottom: 0 }}>
          <button type="button" onClick={onClose} style={buttonStyle}>OK</button>
        </p>
      </div>
    </dialog>
  );
}
'''
    text = replace_once(
        text,
        "\nexport default function MatchUploaderForm({ apiBase, clubId, players, status }: Props) {",
        dialog_component + "\nexport default function MatchUploaderForm({ apiBase, clubId, players, status }: Props) {",
        "result dialog insertion",
    )

    text = sub_once(
        text,
        r'''  const \[entryMethod, setEntryMethod\] = useState<"singles" \| "manual" \| "round_robin">\(.*?\n  \);''',
        '''  const [entryMethod, setEntryMethod] = useState<"singles" | "manual" | "round_robin">("manual");''',
        "manual entry default",
    )
    text = sub_once(
        text,
        r'''  const \[rows, setRows\] = useState<MatchRow\[]>\(.*?\);''',
        '''  const [rows, setRows] = useState<MatchRow[]>(() => [newMatchRow(todayIsoDate(), status.week_tag_options[0] || "Week 1")]);''',
        "one initial row",
    )
    text = sub_once(
        text,
        r'''  const \[result, setResult\] = useState<AdminMatchUploaderWriteResult \| null>\(null\);\n\n  const validRows = rows\.filter\(isFilled\);''',
        '''  const [result, setResult] = useState<AdminMatchUploaderWriteResult | null>(null);
  const [submissionKind, setSubmissionKind] = useState<"manual" | "round_robin" | "singles" | null>(null);

  const filledRows = rows.filter(isFilled);
  const readyRows = rows.filter((row, index) => isReadyRow(row, index));
  const hasInvalidFilledRows = filledRows.length !== readyRows.length;''',
        "ready row state",
    )

    helpers = '''
  function clearEntryFeedback() {
    setMessage(null);
    setResult(null);
  }

  function resetManualRows() {
    const preservedScope = (readyRows[0] || filledRows[0])?.ratingScope || "";
    setRows([newMatchRow(defaultDate, defaultWeekTag, preservedScope)]);
  }

  function acknowledgeSubmission() {
    if (submissionKind === "manual") resetManualRows();
    if (submissionKind === "singles") {
      setSinglesRow((current) => ({
        ...newSinglesRow(),
        date: current.date,
        league: current.league,
        weekTag: current.weekTag,
        ratingScope: current.ratingScope,
      }));
    }
    if (submissionKind === "round_robin") {
      setRrSchedule([]);
      setRrPendingPayload(null);
    }
    setSubmissionKind(null);
    setMessage(null);
    setResult(null);
  }

  function removeRow(rowId: string) {
    clearEntryFeedback();
    setRows((current) => {
      const remaining = current.filter((row) => row.rowId !== rowId);
      return remaining.length ? remaining : [newMatchRow(defaultDate, defaultWeekTag)];
    });
  }

  function removeAllRows() {
    clearEntryFeedback();
    resetManualRows();
  }
'''
    text = replace_once(text, "\n  function requireReady(): boolean {", helpers + "\n  function requireReady(): boolean {", "uploader helpers")

    text = sub_once(
        text,
        r'''  function patchRow\(rowId: string, patch: Partial<MatchRow>\) \{\n    setRows''',
        '''  function patchRow(rowId: string, patch: Partial<MatchRow>) {
    clearEntryFeedback();
    setRows''',
        "clear row feedback",
    )
    text = sub_once(
        text,
        r'''  function patchSingles\(patch: Partial<SinglesRow>\) \{\n    setSinglesRow''',
        '''  function patchSingles(patch: Partial<SinglesRow>) {
    clearEntryFeedback();
    setSinglesRow''',
        "clear singles feedback",
    )
    text = sub_once(
        text,
        r'''  function patchRrCourt\(rowId: string, patch: Partial<RrCourtInput>\) \{\n    setRrCourts''',
        '''  function patchRrCourt(rowId: string, patch: Partial<RrCourtInput>) {
    clearEntryFeedback();
    setRrCourts''',
        "clear round robin feedback",
    )
    text = sub_once(
        text,
        r'''  function patchRrScore\(rowId: string, patch: Partial<Pick<RrScoreRow, "s1" \| "s2">>\) \{\n    setRrSchedule''',
        '''  function patchRrScore(rowId: string, patch: Partial<Pick<RrScoreRow, "s1" | "s2">>) {
    clearEntryFeedback();
    setRrSchedule''',
        "clear score feedback",
    )

    before, singles_submit, after = section(text, "  async function submitSinglesMatch()", "  async function previewRoundRobin(")
    singles_submit = sub_once(
        singles_submit,
        r'''      setResult\(payload\);(.*?)      setMessage\(`Submitted singles match; inserted \$\{payload\.result\?\.inserted \?\? 0\} rated singles match\.`\);\n      setSinglesRow\(.*?\);''',
        r'''      setResult(payload);
      setSubmissionKind("singles");\1      setMessage(`Submitted singles match; inserted ${payload.result?.inserted ?? 0} rated singles match.`);''',
        "singles success reset",
    )
    text = before + singles_submit + after

    before, submit_section, after = section(text, "  async function submitMatches(", "  async function submitManualBatch()")
    submit_section = replace_once(
        submit_section,
        "  async function submitMatches(matches: Array<Record<string, unknown>>, source: string) {",
        "  async function submitMatches(matches: Array<Record<string, unknown>>, source: string, kind: \"manual\" | \"round_robin\") {",
        "submission kind signature",
    )
    submit_section = replace_once(
        submit_section,
        "      setResult(payload);",
        "      setResult(payload);\n      setSubmissionKind(kind);",
        "submission kind state",
    )
    submit_section = sub_once(
        submit_section,
        r'''      const handoffSummary = handoff\?\.mode === "auto_sent"\n        \? ` Player-update email: \$\{handoff\.sent \?\? 0\} sent, \$\{handoff\.skipped \?\? 0\} skipped, \$\{handoff\.errors \?\? 0\} error\(s\)\.`\n        : handoff\?\.mode\n          \? ` Player-update email: \$\{handoff\.mode\}\$\{handoff\.reason \? ` — \$\{handoff\.reason\}` : ""\}\.`\n          : "";''',
        '''      const handoffSummary = handoff?.mode === "auto_sent"
        ? ` Player-update email: ${handoff.sent ?? 0} sent, ${handoff.skipped ?? 0} skipped, ${handoff.errors ?? 0} error(s).`
        : handoff?.mode
          ? " Player-update email was not sent in staging."
          : "";''',
        "technical free email summary",
    )
    text = before + submit_section + after

    text = replace_once(text, "    const matches = validRows.map((row) => ({", "    const matches = readyRows.map((row) => ({", "ready row submit")
    text = replace_once(text, '    await submitMatches(matches, "next_match_uploader_manual_batch");', '    await submitMatches(matches, "next_match_uploader_manual_batch", "manual");', "manual kind")
    text = replace_once(text, '    await submitMatches(matches, "next_match_uploader_round_robin");', '    await submitMatches(matches, "next_match_uploader_round_robin", "round_robin");', "round robin kind")

    text = replace_once(
        text,
        '<select value={entryMethod} onChange={(event) => setEntryMethod(event.target.value as "singles" | "manual" | "round_robin")} style={inputStyle}>',
        '<select value={entryMethod} onChange={(event) => { clearEntryFeedback(); setEntryMethod(event.target.value as "singles" | "manual" | "round_robin"); }} style={inputStyle}>',
        "entry method feedback",
    )
    text = replace_once(
        text,
        '<input value={defaultDate} onChange={(event) => setDefaultDate(event.target.value)} type="date" style={inputStyle} />',
        '<input value={defaultDate} onChange={(event) => { const value = event.target.value; clearEntryFeedback(); setDefaultDate(value); setRows((current) => current.map((row) => isFilled(row) ? row : { ...row, date: value })); }} type="date" style={inputStyle} />',
        "default date",
    )
    text = replace_once(
        text,
        '<select value={defaultWeekTag} onChange={(event) => setDefaultWeekTag(event.target.value)} style={inputStyle}>',
        '<select value={defaultWeekTag} onChange={(event) => { const value = event.target.value; clearEntryFeedback(); setDefaultWeekTag(value); setRows((current) => current.map((row) => isFilled(row) ? row : { ...row, weekTag: value })); }} style={inputStyle}>',
        "default week",
    )

    before, manual_section, after = section(text, '      {entryMethod === "manual" ? (', '      {entryMethod === "round_robin" ? (')
    manual_section = sub_once(
        manual_section,
        r'''          <div style=\{\{ display: "flex", flexWrap: "wrap", gap: "0\.5rem", marginBottom: "0\.75rem" \}\}>.*?          </div>\n          <div style=\{\{ display: "grid", gap: "0\.75rem" \}\}>''',
        '''          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <button type="button" onClick={() => { clearEntryFeedback(); setRows((current) => [...current, newMatchRow(defaultDate, defaultWeekTag)]); }} disabled={rows.length >= status.max_batch_rows} style={ghostButtonStyle}>Add 1 Match</button>
            <button type="button" onClick={() => { clearEntryFeedback(); setRows((current) => [...current, ...Array.from({ length: 5 }, () => newMatchRow(defaultDate, defaultWeekTag))].slice(0, status.max_batch_rows)); }} disabled={rows.length >= status.max_batch_rows} style={ghostButtonStyle}>Add 5 Matches</button>
            {rows.some(isFilled) ? (
              <ConfirmAction
                triggerLabel="Remove All"
                title="Remove all entered matches?"
                description="This clears every entered player and score and returns the uploader to one blank match."
                confirmLabel="Yes, remove all"
                confirmationText="REMOVE"
                tone="danger"
                disabled={saving}
                busy={saving}
                onConfirm={() => removeAllRows()}
              />
            ) : (
              <button type="button" onClick={removeAllRows} disabled={rows.length <= 1} style={ghostButtonStyle}>Remove All</button>
            )}
          </div>
          <div style={{ display: "grid", gap: "0.75rem" }}>''',
        "manual row controls",
    )
    manual_section = sub_once(
        manual_section,
        r'''                <div style=\{\{ display: "flex", justifyContent: "space-between", gap: "0\.5rem", alignItems: "center", marginBottom: "0\.5rem" \}\}><strong>Match \{index \+ 1\}</strong><button.*?</button></div>''',
        '''                <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem" }}>
                  <strong>Match {index + 1}</strong>
                  {isFilled(row) ? (
                    <ConfirmAction
                      triggerLabel="Remove match"
                      title={`Remove Match ${index + 1}?`}
                      description="This match contains entered data. Removing it cannot be undone from this screen."
                      confirmLabel="Yes, remove match"
                      confirmationText="REMOVE"
                      tone="danger"
                      disabled={saving}
                      busy={saving}
                      onConfirm={() => removeRow(row.rowId)}
                    />
                  ) : (
                    <button type="button" onClick={() => removeRow(row.rowId)} disabled={rows.length <= 1}>Remove match</button>
                  )}
                </div>''',
        "individual row confirmation",
    )
    manual_section = replace_once(
        manual_section,
        'gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))"',
        'gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))"',
        "player field width",
    )
    manual_section = replace_once(
        manual_section,
        '''          <p><strong>Ready rows:</strong> {validRows.length} / {rows.length}</p>
          <button type="button" onClick={submitManualBatch} disabled={saving || !accessToken || !validRows.length} style={buttonStyle}>{saving ? "Submitting…" : "Submit batch"}</button>''',
        '''          <p><strong>Ready rows:</strong> {readyRows.length} / {rows.length}</p>
          <button type="button" onClick={submitManualBatch} disabled={saving || !accessToken || !readyRows.length || hasInvalidFilledRows} style={buttonStyle}>{saving ? "Submitting…" : "Submit batch"}</button>''',
        "ready row display",
    )
    text = before + manual_section + after

    text = replace_once(
        text,
        '{message ? <p aria-live="polite" role={messageIsError ? "alert" : "status" style={{ color: messageIsError ? "#b91c1c" : "#166534" }}>{message}</p> : null}',
        '{message && !result ? <p aria-live="polite" role={messageIsError ? "alert" : "status" style={{ color: messageIsError ? "#b91c1c" : "#166534" }}>{message}</p> : null}',
        "modal success message",
    )
    text = sub_once(
        text,
        r'''\n      \{result \? \(\n        <article style=\{cardStyle\}>.*?\n      \) : null\}\n    </section>''',
        '''
      {result ? <SubmissionResultDialog result={result} onClose={acknowledgeSubmission} /> : null}
    </section>''',
        "result card to dialog",
    )

    path.write_text(text, encoding="utf-8")


def update_match_log_workspace() -> None:
    path = ROOT / "apps/web/app/admin/match-log/MatchLogWorkspace.tsx"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '  const showsMatchContext = mode !== "social" && mode !== "replay";',
        '  const showsMatchContext = mode !== "social" && mode !== "replay";\n  const showsMatchSummary = showsMatchContext && mode !== "edit";\n  const showsMatchTable = showsMatchContext && mode !== "edit" && mode !== "duplicates";',
        "edit mode visibility",
    )
    text = replace_once(
        text,
        '''          {showsMatchContext ? <form data-testid="match-log-filters" style={{ ...cardStyle, marginBottom: "1rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label>Filter<br />''',
        '''          {showsMatchContext ? <form data-testid="match-log-filters" style={{ ...cardStyle, marginBottom: "1rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            {mode === "edit" ? (
              <div style={{ gridColumn: "1 / -1" }}>
                <h2 style={{ margin: 0 }}>Find a match</h2>
                <p style={{ ...muted, marginBottom: 0 }}>Filter the match list, then use the compact selector in the editor below.</p>
              </div>
            ) : null}
            <label>Filter<br />''',
        "edit filters heading",
    )
    text = replace_once(text, '{showsMatchContext ? <div data-testid="match-log-summary"', '{showsMatchSummary ? <div data-testid="match-log-summary"', "hide edit summary")
    text = replace_once(text, '{showsMatchContext && mode !== "duplicates" ? <section data-testid="match-log-results"', '{showsMatchTable ? <section data-testid="match-log-results"', "hide edit match table")
    path.write_text(text, encoding="utf-8")


def update_match_log_panel() -> None:
    path = ROOT / "apps/web/app/admin/match-log/MatchLogApplyPanel.tsx"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '{mode === "guided" ? "Guided match correction" : mode === "bulk" ? "Bulk match correction" : "Duplicate resolution"}',
        '{mode === "guided" ? "Match editor" : mode === "bulk" ? "Bulk match correction" : "Duplicate resolution"}',
        "editor title",
    )
    text = replace_once(
        text,
        '''      {mode === "guided" ? <>
      <h3>Guided match editor</h3>
      <p style={{ color: "#475569" }}>
        Select a match from the current filtered results, change fields with form controls, stage the edit, then apply all staged edits together.
      </p>''',
        '''      {mode === "guided" ? <>
      <p style={{ color: "#475569" }}>
        Use the filters above to narrow the choices, select one match, change its fields, stage the edit, then apply the reviewed change.
      </p>''',
        "editor introduction",
    )
    text = sub_once(
        text,
        r'''      \{mode !== "duplicates" && recentOperations\.length \? \(\n        <div data-testid="match-edit-operation-history".*?\n        </div>\n      \) : null\}''',
        '''      {mode !== "duplicates" && recentOperations.length ? (
        <details data-testid="match-edit-operation-history" style={{ marginTop: "1rem" }}>
          <summary style={{ cursor: "pointer", fontWeight: 700 }}>Recent durable edit operations</summary>
          <div style={{ overflowX: "auto", marginTop: "0.75rem" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "680px" }}>
              <thead><tr>{["Created", "Status", "Replay", "Actor", "Operation"].map((label) => <th key={label} style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>{label}</th>)}</tr></thead>
              <tbody>{recentOperations.map((operation) => (
                <tr key={operation.id} data-operation-status={operation.status}>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{operation.created_at ? new Date(operation.created_at).toISOString().slice(0, 19).replace("T", " ") : "—"}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{operation.status}{operation.error_text ? ` · ${operation.error_text}` : ""}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{operation.replay_target || "Not required"}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{operation.actor_email || "—"}</td>
                  <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem", fontFamily: "monospace" }}>{operation.id}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </details>
      ) : null}''',
        "collapsed durable history",
    )
    path.write_text(text, encoding="utf-8")


def write_tests() -> None:
    path = ROOT / "tests/test_manual_acceptance_ux_regressions.py"
    path.write_text(
        '''from pathlib import Path\n\n\ndef test_match_uploader_manual_acceptance_fixes_are_present() -> None:\n    source = Path("apps/web/app/admin/match-uploader/MatchUploaderForm.tsx").read_text(encoding="utf-8")\n    assert 'useState<"singles" | "manual" | "round_robin">("manual")' in source\n    assert '[newMatchRow(todayIsoDate(), status.week_tag_options[0] || "Week 1")]' in source\n    assert "const readyRows = rows.filter" in source\n    assert "hasInvalidFilledRows" in source\n    assert 'triggerLabel="Remove match"' in source\n    assert 'triggerLabel="Remove All"' in source\n    assert "SubmissionResultDialog" in source\n    assert "Successfully inserted" in source\n    assert "Player-update email was not sent in staging." in source\n    assert "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS is not enabled" not in source\n    assert "matchingPlayers.length === 0" in source\n\n\ndef test_match_log_edit_page_is_compact() -> None:\n    workspace = Path("apps/web/app/admin/match-log/MatchLogWorkspace.tsx").read_text(encoding="utf-8")\n    panel = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx").read_text(encoding="utf-8")\n    assert 'const showsMatchSummary = showsMatchContext && mode !== "edit"' in workspace\n    assert 'const showsMatchTable = showsMatchContext && mode !== "edit"' in workspace\n    assert "Find a match" in workspace\n    assert 'mode === "guided" ? "Match editor"' in panel\n    assert '<details data-testid="match-edit-operation-history"' in panel\n    assert "Use the filters above to narrow the choices" in panel\n''',
        encoding="utf-8",
    )


def main() -> None:
    update_uploader()
    update_match_log_workspace()
    update_match_log_panel()
    write_tests()


if __name__ == "__main__":
    main()

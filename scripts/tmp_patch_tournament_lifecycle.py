from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:140]!r}")
    target.write_text(text.replace(old, new, 1))


def replace_range(path: str, start: str, end: str, replacement: str) -> None:
    target = Path(path)
    text = target.read_text()
    start_at = text.index(start)
    end_at = text.index(end, start_at)
    target.write_text(text[:start_at] + replacement + text[end_at:])


def remove_article_by_heading(path: str, heading: str) -> None:
    target = Path(path)
    text = target.read_text()
    marker = f'<h2 style={{{{ marginTop: 0 }}}}>{heading}</h2>'
    marker_at = text.index(marker)
    start = text.rfind("<article", 0, marker_at)
    end = text.index("</article>", marker_at) + len("</article>")
    target.write_text(text[:start] + text[end:])


def patch_commerce() -> None:
    path = "apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx"
    replace_once(path, "  listAdminTournamentCommerceTournaments,\n", "")
    replace_once(
        path,
        'import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";',
        'import { useAdminSession } from "@/lib/useAdminSession";',
    )
    replace_once(
        path,
        "type Props = {\n  clubId: string;\n};",
        "type Props = {\n  clubId: string;\n  tournamentId: string;\n  tournamentName: string;\n};",
    )
    replace_once(
        path,
        '  width: "100%",\n  padding: "0.55rem",',
        '  width: "100%",\n  minWidth: 0,\n  boxSizing: "border-box" as const,\n  padding: "0.55rem",',
    )
    replace_once(
        path,
        '''export default function TournamentCommercePanel({ clubId }: Props) {
  const {
    session,
    accessToken,
    loading: sessionLoading,
    message: sessionMessage
  } = useAdminSession();
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [tournaments, setTournaments] = useState<
    Array<{ id: string; name: string; status?: string }>
  >([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState("");''',
        '''export default function TournamentCommercePanel({
  clubId,
  tournamentId,
  tournamentName
}: Props) {
  const {
    accessToken,
    loading: sessionLoading,
    message: sessionMessage
  } = useAdminSession();
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [selectedTournamentId, setSelectedTournamentId] = useState(tournamentId);''',
    )

    target = Path(path)
    text = target.read_text()
    start = text.index("  async function loadWorkspace() {")
    end = text.index("  async function loadDetail(", start)
    workspace = '''  async function loadWorkspace() {
    if (!accessToken) return;
    setBusy(true);
    setMessage(null);
    const statusResponse = await getAdminTournamentCommerceStatus(
      clubId,
      accessToken
    );
    if (statusResponse.error) {
      setBusy(false);
      setMessage(statusResponse.error);
      setStatus(null);
      return;
    }
    setStatus(statusResponse.data);
    await loadDetail(tournamentId);
  }

'''
    target.write_text(text[:start] + workspace + text[end:])

    replace_once(
        path,
        "    setMessage(`Loaded ${response.data.tournament.name}.`);",
        "    setMessage(null);",
    )
    replace_once(
        path,
        "  useAuthenticatedAutoLoad(accessToken, loadWorkspace);",
        '  useAuthenticatedAutoLoad(`${accessToken}\\u0000${tournamentId}`, loadWorkspace);',
    )
    remove_article_by_heading(path, "Tournament extras workspace")

    target = Path(path)
    text = target.read_text()
    start = text.index("      {tournaments.length ? (")
    end = text.index("      {detail && draft ? (", start)
    text = text[:start] + text[end:]
    text = text.replace(
        '<section style={{ display: "grid", gap: "1rem" }}>',
        '<section data-commerce-form aria-label={`${tournamentName} payments, extras, and fulfillment`} style={{ display: "grid", gap: "1rem" }}>\n      <style>{`[data-commerce-form] label { min-width: 0; } [data-commerce-form] input, [data-commerce-form] select, [data-commerce-form] textarea, [data-commerce-form] button { box-sizing: border-box; max-width: 100%; } [data-commerce-form] summary { cursor: pointer; }`}</style>',
        1,
    )

    marker = "  const eventLabels = useMemo("
    insert_at = text.index(marker)
    saved_sets = '''  const savedItemIds = useMemo(
    () => new Set((detail?.catalog.items || []).map((item) => item.id)),
    [detail?.catalog.items]
  );
  const savedBundleIds = useMemo(
    () => new Set((detail?.catalog.bundles || []).map((bundle) => bundle.id)),
    [detail?.catalog.bundles]
  );

'''
    text = text[:insert_at] + saved_sets + text[insert_at:]

    old = '''                      <input
                        type="number"
                        min="0"
                        step="0.01"
                        value={(item.base_price_minor / 100).toFixed(2)}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.base_price_minor = moneyMinor(event.target.value);
                          })
                        }
                        style={inputStyle}
                      />'''
    new = '''                      <input
                        key={`${item.id}-${detail?.catalog.catalog_revision || 0}-base-price`}
                        type="text"
                        inputMode="decimal"
                        defaultValue={(item.base_price_minor / 100).toFixed(2)}
                        onBlur={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.base_price_minor = moneyMinor(event.target.value);
                          })
                        }
                        style={inputStyle}
                      />'''
    if old not in text:
        raise SystemExit("commerce base price block not found")
    text = text.replace(old, new, 1)

    old = '''                          <input
                            type="number"
                            step="0.01"
                            value={(variant.price_delta_minor / 100).toFixed(2)}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.variants.find(
                                  (row) => row.id === variant.id
                                )!.price_delta_minor = moneyMinor(
                                  event.target.value
                                );
                              })
                            }
                            style={inputStyle}
                          />'''
    new = '''                          <input
                            key={`${variant.id}-${detail?.catalog.catalog_revision || 0}-price-delta`}
                            type="text"
                            inputMode="decimal"
                            defaultValue={(variant.price_delta_minor / 100).toFixed(2)}
                            onBlur={(event) =>
                              updateDraft((next) => {
                                next.variants.find(
                                  (row) => row.id === variant.id
                                )!.price_delta_minor = moneyMinor(
                                  event.target.value
                                );
                              })
                            }
                            style={inputStyle}
                          />'''
    if old not in text:
        raise SystemExit("commerce option price block not found")
    text = text.replace(old, new, 1)

    old = '''                          <input
                            type="number"
                            min="0"
                            step="0.01"
                            value={(bundle.price_minor / 100).toFixed(2)}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.bundles.find(
                                  (row) => row.id === bundle.id
                                )!.price_minor = moneyMinor(event.target.value);
                              })
                            }
                            style={inputStyle}
                          />'''
    new = '''                          <input
                            key={`${bundle.id}-${detail?.catalog.catalog_revision || 0}-bundle-price`}
                            type="text"
                            inputMode="decimal"
                            defaultValue={(bundle.price_minor / 100).toFixed(2)}
                            onBlur={(event) =>
                              updateDraft((next) => {
                                next.bundles.find(
                                  (row) => row.id === bundle.id
                                )!.price_minor = moneyMinor(event.target.value);
                              })
                            }
                            style={inputStyle}
                          />'''
    if old not in text:
        raise SystemExit("commerce bundle price block not found")
    text = text.replace(old, new, 1)

    item_map_start = text.index("{draft.items.map((item) => (")
    item_map_end = text.index("\n              ))}", item_map_start)
    item_map = text[item_map_start:item_map_end]
    item_open = item_map.index('<article key={item.id} style={cardStyle}>')
    item_close = item_map.rfind("</article>")
    item_map = (
        item_map[:item_open]
        + '''<details key={item.id} open={!savedItemIds.has(item.id)} style={cardStyle}>
                  <summary style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
                    <span><strong>{item.name || "Untitled extra"}</strong><br /><small>{formatCommerceMoney(item.base_price_minor)} · {item.status} · {item.kind.toLowerCase()}</small></span>
                    <span style={{ fontWeight: 800 }}>{savedItemIds.has(item.id) ? "Edit" : "New extra"}</span>
                  </summary>
                  <div style={{ marginTop: "1rem" }}>'''
        + item_map[item_open + len('<article key={item.id} style={cardStyle}>'):item_close]
        + "</div>\n                </details>"
        + item_map[item_close + len("</article>"):]
    )
    text = text[:item_map_start] + item_map + text[item_map_end:]

    bundle_map_start = text.index("{draft.bundles.map((bundle) => (")
    bundle_map_end = text.index("\n                  ))}", bundle_map_start)
    bundle_map = text[bundle_map_start:bundle_map_end]
    bundle_open = bundle_map.index("<section", bundle_map.index("{draft.bundles"))
    bundle_tag_end = bundle_map.index(">", bundle_open) + 1
    bundle_close = bundle_map.rfind("</section>")
    bundle_map = (
        bundle_map[:bundle_open]
        + '''<details key={bundle.id} open={!savedBundleIds.has(bundle.id)} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                      <summary style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
                        <span><strong>{bundle.name || "Untitled bundle"}</strong><br /><small>{formatCommerceMoney(bundle.price_minor)} · {bundle.status} · {(componentsByBundle.get(bundle.id) || []).length} required parts</small></span>
                        <span style={{ fontWeight: 800 }}>{savedBundleIds.has(bundle.id) ? "Edit" : "New bundle"}</span>
                      </summary>
                      <section style={{ marginTop: "1rem" }}>'''
        + bundle_map[bundle_tag_end:bundle_close]
        + "</section>\n                    </details>"
        + bundle_map[bundle_close + len("</section>"):]
    )
    bundle_map = bundle_map.replace(
        '<select\n                          defaultValue=""',
        '<select\n                          key={`${bundle.id}-${draft.variants.length}-${draft.event_options.length}`}\n                          defaultValue=""',
        1,
    )
    text = text[:bundle_map_start] + bundle_map + text[bundle_map_end:]
    target.write_text(text)


def patch_team_competition() -> None:
    path = "apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx"
    replace_once(path, "  listAdminTeamTournaments,\n", "")
    replace_once(
        path,
        'import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";',
        'import { useAdminSession } from "@/lib/useAdminSession";',
    )
    replace_once(
        path,
        "type Props = { clubId: string };",
        'type Props = { clubId: string; initialTournamentId: string };\ntype EventFormat = "STANDARD" | "COMBINED_RATING_CAP" | "FOUR_PLAYER_TEAM";',
    )
    replace_once(
        path,
        '''export default function TeamTournamentAdminPanel({ clubId }: Props) {
  const {
    session,
    accessToken,
    loading: sessionLoading,
    message: sessionMessage
  } = useAdminSession();''',
        '''export default function TeamTournamentAdminPanel({
  clubId,
  initialTournamentId
}: Props) {
  const {
    accessToken,
    loading: sessionLoading,
    message: sessionMessage
  } = useAdminSession();''',
    )
    replace_once(
        path,
        '''  const [tournaments, setTournaments] = useState<
    Array<{ id: string; name: string; status?: string }>
  >([]);
  const [tournamentId, setTournamentId] = useState("");''',
        "  const [tournamentId, setTournamentId] = useState(initialTournamentId);",
    )

    target = Path(path)
    text = target.read_text()
    start = text.index("  async function loadWorkspace(): Promise<void> {")
    end_marker = "  useAuthenticatedAutoLoad(accessToken, loadWorkspace, clubId);"
    end = text.index(end_marker, start) + len(end_marker)
    replacement = '  useAuthenticatedAutoLoad(\n    `${accessToken}\\u0000${initialTournamentId}`,\n    () => loadSnapshot(initialTournamentId),\n    clubId\n  );'
    text = text[:start] + replacement + text[end:]
    text = text.replace(
        "    setSnapshot(null);\n",
        "    if (nextTournamentId !== tournamentId) setSnapshot(null);\n",
        1,
    )
    text = text.replace(
        "    setMessage(`Loaded ${response.data.tournament.name}.`);",
        "    setMessage(null);",
        1,
    )

    marker = "function slotLabel(slot: TeamSlot): string {"
    insert_at = text.index(marker)
    helpers = '''function eventFormat(draft: ConfigDraft | null): EventFormat {
  if (draft?.competitionFormat === "FOUR_PLAYER_TEAM") return "FOUR_PLAYER_TEAM";
  if (draft?.eligibilityMode === "COMBINED_RATING_CAP") return "COMBINED_RATING_CAP";
  return "STANDARD";
}

function friendlyWorkspaceWarning(warning: string): string {
  if (/tournament_team_operations unavailable|apierror/i.test(warning)) {
    return "Team scheduling and scoring data is temporarily unavailable. Event-format setup and rating review remain available.";
  }
  return warning;
}

'''
    text = text[:insert_at] + helpers + text[insert_at:]

    toolbar_start = text.index("      <div className={styles.toolbar}>")
    message_start = text.index("      {message ? (", toolbar_start)
    text = (
        text[:toolbar_start]
        + '      {busy && !snapshot ? <p className={styles.notice}>Loading event setup…</p> : null}\n'
        + text[message_start:]
    )
    old_warning = '''            {snapshot.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}'''
    new_warning = '''            {snapshot.warnings
              .map((warning) => friendlyWorkspaceWarning(warning))
              .map((warning) => (
                <li key={warning}>{warning}</li>
              ))}'''
    if old_warning not in text:
        raise SystemExit("team warning block not found")
    text = text.replace(old_warning, new_warning, 1)

    eligibility_start = text.index(
        "                  <label className={styles.field}>\n                    Registration eligibility"
    )
    team_condition = text.index(
        '                  {configDraft?.competitionFormat === "FOUR_PLAYER_TEAM" ? (',
        eligibility_start,
    )
    unified = '''                  <label className={styles.field}>
                    Event format
                    <select
                      className={styles.select}
                      value={eventFormat(configDraft)}
                      onChange={(event) =>
                        setConfigDraft((current) => {
                          if (!current) return current;
                          const value = event.target.value as EventFormat;
                          if (value === "COMBINED_RATING_CAP") {
                            return {
                              ...current,
                              eligibilityMode: "COMBINED_RATING_CAP",
                              combinedRatingCap: current.combinedRatingCap || "8.0",
                              competitionFormat: "STANDARD"
                            };
                          }
                          if (value === "FOUR_PLAYER_TEAM") {
                            return {
                              ...current,
                              eligibilityMode: "STANDARD",
                              combinedRatingCap: "",
                              competitionFormat: "FOUR_PLAYER_TEAM"
                            };
                          }
                          return {
                            ...current,
                            eligibilityMode: "STANDARD",
                            combinedRatingCap: "",
                            competitionFormat: "STANDARD"
                          };
                        })
                      }
                    >
                      <option value="STANDARD">Standard singles or doubles</option>
                      <option value="COMBINED_RATING_CAP">Combined-rating doubles</option>
                      <option value="FOUR_PLAYER_TEAM">Four-player team · two men and two women</option>
                    </select>
                  </label>
                  {configDraft?.eligibilityMode === "COMBINED_RATING_CAP" ? (
                    <label className={styles.field}>
                      Maximum combined partner rating
                      <input
                        className={styles.input}
                        type="number"
                        min="0.01"
                        max="14"
                        step="0.01"
                        value={configDraft.combinedRatingCap}
                        onChange={(event) =>
                          setConfigDraft((current) =>
                            current
                              ? { ...current, combinedRatingCap: event.target.value }
                              : current
                          )
                        }
                      />
                      <small>Eligibility is strictly below this cap.</small>
                    </label>
                  ) : null}
'''
    text = text[:eligibility_start] + unified + text[team_condition:]

    actions_at = text.index(
        "                <div className={styles.actions}>",
        text.index('id="team-setup-heading"'),
    )
    summary = '''                {configDraft ? (
                  <p className={styles.notice}>
                    <strong>Review:</strong>{" "}
                    {eventFormat(configDraft) === "STANDARD"
                      ? "Standard singles or doubles"
                      : eventFormat(configDraft) === "COMBINED_RATING_CAP"
                        ? `Combined-rating doubles · below ${configDraft.combinedRatingCap || "—"}`
                        : `Four-player team · ${configDraft.allowSubstitutes ? "substitutes allowed" : "no substitutes"} · ${configDraft.tiebreakMode === "SKINNY_RELAY" ? "skinny-singles relay" : "one singles game"} · ${configDraft.playoffFormat.replaceAll("_", " ").toLowerCase()}`}
                  </p>
                ) : null}
'''
    text = text[:actions_at] + summary + text[actions_at:]
    target.write_text(text)


def patch_ops() -> None:
    path = "apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx"
    replace_once(path, "  AdminTournament,\n  AdminTournamentListResponse,\n", "")
    replace_once(
        path,
        'import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";',
        'import { useAdminSession } from "@/lib/useAdminSession";',
    )
    replace_once(
        path,
        'type Props = { apiBase: string | null; clubId: string; status: AdminTournamentStatusResponse; workflow?: OpsWorkflow };',
        'type Props = { apiBase: string | null; clubId: string; status: AdminTournamentStatusResponse; workflow?: OpsWorkflow; initialTournamentId: string };',
    )
    replace_once(
        path,
        '''export default function TournamentOpsPanel({ apiBase, clubId, status, workflow = "all" }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState("");''',
        '''export default function TournamentOpsPanel({
  apiBase,
  clubId,
  status,
  workflow = "all",
  initialTournamentId
}: Props) {
  const { accessToken } = useAdminSession();
  const [selectedTournamentId, setSelectedTournamentId] = useState(initialTournamentId);''',
    )
    replace_once(
        path,
        "  const listRequest = useLatestRequestGuard(accessToken, clearProtectedOpsState);\n",
        "",
    )

    target = Path(path)
    text = target.read_text()
    text = text.replace(
        '    setTournaments([]); setSelectedTournamentId(""); setSelectedDrawId(""); setSnapshot(null);',
        '    setSelectedTournamentId(initialTournamentId); setSelectedDrawId(""); setSnapshot(null);',
        1,
    )
    load_start = text.index("  async function loadTournaments() {")
    load_end = text.index("  async function loadOps(", load_start)
    text = text[:load_start] + text[load_end:]
    select_start = text.index("  function selectTournament(tournamentId: string) {")
    select_end = text.index("  function selectDraw(drawId: string) {", select_start)
    text = text[:select_start] + text[select_end:]
    old_auto = '  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadTournaments, includeArchived ? "archived" : "active");'
    if old_auto not in text:
        raise SystemExit("ops auto-load line not found")
    text = text.replace(
        old_auto,
        '  useAuthenticatedAutoLoad(\n    status.enabled ? `${accessToken}\\u0000${initialTournamentId}` : "",\n    () => loadOps(initialTournamentId, "")\n  );',
        1,
    )
    text = text.replace(
        '      setMessage("Tournament operations snapshot loaded.");',
        "      setMessage(null);",
        1,
    )
    header_start = text.index(
        '      <article style={cardStyle}>\n        <h2 style={{ marginTop: 0 }}>Tournament Ops</h2>'
    )
    banner_start = text.index("      {!operationsWriteReady ? (", header_start)
    text = text[:header_start] + text[banner_start:]
    selector_start = text.index("      {tournaments.length ? (")
    content_start = text.index(
        '      {operationsWriteReady && snapshot && shows("draws") ? (',
        selector_start,
    )
    text = text[:selector_start] + text[content_start:]
    access_guard = '''
  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p><Link href="/admin/login">Open admin login</Link></p>
      </article>
    );
  }
'''
    marker = '  if (!status.enabled) {'
    marker_at = text.index(marker)
    text = text[:marker_at] + access_guard + text[marker_at:]
    for forbidden in (
        "setTournaments(",
        "includeArchived",
        "loadTournaments",
        "tournaments.length",
        "selectTournament(",
        "sessionLoading",
        "sessionMessage",
        "adminSessionLabel",
    ):
        if forbidden in text:
            raise SystemExit(f"ops legacy token remains: {forbidden}")
    target.write_text(text)


def main() -> None:
    patch_commerce()
    patch_team_competition()
    patch_ops()


if __name__ == "__main__":
    main()

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction/types";
import {
  fetchAdminTournamentCheckIn,
  updateAdminTournamentCheckIn,
  updateAdminTournamentCheckInBulk
} from "@/lib/adminTournamentCheckInApi";
import type {
  TournamentCheckInRegistrant,
  TournamentCheckInSnapshot
} from "@/lib/adminTournamentCheckInApi";
import {
  readTournamentRouteContext,
  tournamentRouteHref
} from "@/lib/tournamentRouteContext";
import { useAdminSession } from "@/lib/useAdminSession";
import styles from "./TournamentCheckInPanel.module.css";

type Props = {
  apiBase: string | null;
  clubId: string;
  initialDayId: string;
  tournamentId: string;
};

type AttendanceStatus = "EXPECTED" | "CHECKED_IN" | "ABSENT";
type Filter = "all" | "expected" | "checked_in" | "absent" | "unresolved";
type BulkAttendanceAction = "NO_CHANGE" | AttendanceStatus;
type BulkWaiverAction = "NO_CHANGE" | "VERIFY" | "CLEAR";
const MAX_BULK_UPDATES = 100;
type Draft = {
  attendanceStatus: AttendanceStatus;
  waiverVerified: boolean;
  substitutePlayerId: string;
  notes: string;
  expectedUpdatedAt: string | null;
};

function attendanceStatus(card: TournamentCheckInRegistrant): AttendanceStatus {
  const status = String(card.attendance_status || "").toUpperCase();
  if (status === "CHECKED_IN" || status === "ABSENT") return status;
  return "EXPECTED";
}

function attendanceLabel(status: AttendanceStatus): string {
  if (status === "CHECKED_IN") return "Checked in";
  if (status === "ABSENT") return "Absent";
  return "Not checked in";
}

function initialDraft(card: TournamentCheckInRegistrant): Draft {
  return {
    attendanceStatus: attendanceStatus(card),
    waiverVerified: Boolean(card.waiver.verified),
    substitutePlayerId: card.attendee.is_approved_substitute
      ? String(card.attendee.player_id || "__unavailable__")
      : "",
    notes: String(card.check_in.notes || ""),
    expectedUpdatedAt: card.check_in.updated_at || null
  };
}

function legacySubstitutePending(
  card: TournamentCheckInRegistrant,
  draft: Draft
): boolean {
  return Boolean(card.attendee.is_approved_substitute && draft.substitutePlayerId);
}

function draftChanged(card: TournamentCheckInRegistrant, draft: Draft): boolean {
  return (
    draft.attendanceStatus !== attendanceStatus(card) ||
    draft.waiverVerified !== Boolean(card.waiver.verified) ||
    draft.substitutePlayerId !==
      (card.attendee.is_approved_substitute
        ? String(card.attendee.player_id || "__unavailable__")
        : "") ||
    (draft.notes.trim() || null) !== (String(card.check_in.notes || "").trim() || null)
  );
}

function toneClass(status: string): string {
  const normalized = status.toUpperCase();
  if (normalized === "COMPLETE" || normalized === "CONFIRMED_LINK") {
    return `${styles.badge} ${styles.complete}`;
  }
  if (normalized === "NEEDS_REVIEW") return `${styles.badge} ${styles.review}`;
  return `${styles.badge} ${styles.blocked}`;
}

function statusLabel(status: string): string {
  if (status === "NEEDS_REVIEW") return "Needs review";
  return status
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/^./, (letter) => letter.toUpperCase());
}

export default function TournamentCheckInPanel({
  apiBase,
  clubId,
  initialDayId,
  tournamentId
}: Props) {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [snapshot, setSnapshot] = useState<TournamentCheckInSnapshot | null>(null);
  const [selectedDayId, setSelectedDayId] = useState(initialDayId);
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [bulkAttendance, setBulkAttendance] =
    useState<BulkAttendanceAction>("NO_CHANGE");
  const [bulkWaiver, setBulkWaiver] = useState<BulkWaiverAction>("NO_CHANGE");
  const [loading, setLoading] = useState(false);
  const [savingId, setSavingId] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const operationKeys = useRef<Record<string, { fingerprint: string; key: string }>>({});
  const bulkOperation = useRef<{ fingerprint: string; key: string } | null>(null);

  const preserveDayInUrl = useCallback(
    (dayId: string) => {
      if (!dayId) return;
      const context = readTournamentRouteContext(searchParams);
      if (context.dayId === dayId && searchParams.get("day") === dayId) return;
      router.replace(
        tournamentRouteHref(pathname, {
          ...context,
          tournamentId,
          dayId
        }),
        { scroll: false }
      );
    },
    [pathname, router, searchParams, tournamentId]
  );

  const loadSnapshot = useCallback(
    async (signal?: AbortSignal) => {
      if (!apiBase || !accessToken || !tournamentId) return false;
      setLoading(true);
      setError(null);
      try {
        const payload = await fetchAdminTournamentCheckIn({
          apiBase,
          clubId,
          tournamentId,
          accessToken,
          dayId: selectedDayId || undefined,
          signal
        });
        if (signal?.aborted) return false;
        const authoritativeDayId = payload.day_scope.selected_day_id;
        if (authoritativeDayId && authoritativeDayId !== selectedDayId) {
          setSelectedDayId(authoritativeDayId);
          preserveDayInUrl(authoritativeDayId);
        }
        setSnapshot(payload);
        setDrafts(
          Object.fromEntries(
            payload.registrants.map((card) => [card.registration_id, initialDraft(card)])
          )
        );
        setSelectedIds([]);
        setBulkAttendance("NO_CHANGE");
        setBulkWaiver("NO_CHANGE");
        return true;
      } catch (loadError) {
        if (signal?.aborted) return false;
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Unable to load authoritative tournament check-in state."
        );
        return false;
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [accessToken, apiBase, clubId, preserveDayInUrl, selectedDayId, tournamentId]
  );

  useEffect(() => {
    const controller = new AbortController();
    if (accessToken) void loadSnapshot(controller.signal);
    else {
      setSnapshot(null);
      setDrafts({});
    }
    return () => controller.abort();
  }, [accessToken, loadSnapshot]);

  const unresolvedIds = useMemo(
    () =>
      new Set(
        (snapshot?.unresolved_participants || []).map((row) => row.registration_id)
      ),
    [snapshot]
  );

  const selectedDayLabel = useMemo(() => {
    const day = snapshot?.day_scope.selected_day;
    if (!day) return "selected tournament day";
    return day.event_date ? `${day.label} · ${day.event_date}` : day.label;
  }, [snapshot]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return (snapshot?.registrants || []).filter((card) => {
      const status = attendanceStatus(card);
      if (filter === "expected" && status !== "EXPECTED") return false;
      if (filter === "checked_in" && status !== "CHECKED_IN") return false;
      if (filter === "absent" && status !== "ABSENT") return false;
      if (filter === "unresolved" && !unresolvedIds.has(card.registration_id)) return false;
      if (!needle) return true;
      const haystack = [
        card.original_registrant.name,
        card.attendee.name,
        ...card.events.map((event) => event.event_label),
        ...card.events.map(
          (event) => event.partner_name || event.entered_partner_name || ""
        )
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(needle);
    });
  }, [filter, search, snapshot, unresolvedIds]);

  const selectedIdSet = useMemo(() => new Set(selectedIds), [selectedIds]);

  const selectableFilteredIds = useMemo(
    () =>
      filtered
        .filter((card) => {
          const draft = drafts[card.registration_id] || initialDraft(card);
          return !legacySubstitutePending(card, draft);
        })
        .map((card) => card.registration_id),
    [drafts, filtered]
  );

  const hasSelectedDraftChanges = useMemo(() => {
    if (!snapshot) return false;
    return snapshot.registrants.some((card) => {
      if (!selectedIdSet.has(card.registration_id)) return false;
      return draftChanged(card, drafts[card.registration_id] || initialDraft(card));
    });
  }, [drafts, selectedIdSet, snapshot]);

  const hasBulkChange =
    bulkAttendance !== "NO_CHANGE" ||
    bulkWaiver !== "NO_CHANGE" ||
    hasSelectedDraftChanges;
  const bulkActionIsDestructive =
    bulkAttendance === "ABSENT" ||
    bulkAttendance === "EXPECTED" ||
    bulkWaiver === "CLEAR";
  const bulkActionSummary = [
    bulkAttendance === "CHECKED_IN"
      ? "mark checked in"
      : bulkAttendance === "ABSENT"
        ? "mark absent"
        : bulkAttendance === "EXPECTED"
          ? "reset to not checked in"
          : "",
    bulkWaiver === "VERIFY"
      ? "mark waiver verified"
      : bulkWaiver === "CLEAR"
        ? "clear waiver verification"
        : "",
    hasSelectedDraftChanges ? "save edited operator notes" : ""
  ].filter(Boolean).join("; ");

  function patchDraft(registrationId: string, patch: Partial<Draft>): void {
    setDrafts((current) => ({
      ...current,
      [registrationId]: {
        ...(current[registrationId] || {
          attendanceStatus: "EXPECTED",
          waiverVerified: false,
          substitutePlayerId: "",
          notes: "",
          expectedUpdatedAt: null
        }),
        ...patch
      }
    }));
  }

  async function restoreOriginalRegistrant(
    card: TournamentCheckInRegistrant
  ): Promise<void> {
    if (savingId) return;
    const restoredDraft: Draft = {
      ...(drafts[card.registration_id] || initialDraft(card)),
      substitutePlayerId: "",
      attendanceStatus: "EXPECTED",
      waiverVerified: false
    };
    setSavingId(card.registration_id);
    setMessage(null);
    setError(null);
    try {
      await saveDraft(card, restoredDraft);
      const reloaded = await loadSnapshot();
      if (reloaded) setMessage("Original registrant restored for check-in review.");
    } catch (restoreError) {
      setError(
        restoreError instanceof Error
          ? restoreError.message
          : "Unable to restore the original registrant. Reload before trying again."
      );
    } finally {
      setSavingId("");
    }
  }

  function toggleSelected(registrationId: string): void {
    setSelectedIds((current) => {
      if (current.includes(registrationId)) {
        return current.filter((id) => id !== registrationId);
      }
      if (current.length >= MAX_BULK_UPDATES) {
        setError(`Select no more than ${MAX_BULK_UPDATES} players in one bulk update.`);
        return current;
      }
      return [...current, registrationId];
    });
  }

  function selectAllShown(): void {
    setSelectedIds(() => {
      if (selectableFilteredIds.length > MAX_BULK_UPDATES) {
        setError(
          `Selected the first ${MAX_BULK_UPDATES} players shown. Apply that batch before selecting more.`
        );
      }
      return selectableFilteredIds.slice(0, MAX_BULK_UPDATES);
    });
  }

  function ensureSelected(registrationId: string): void {
    setSelectedIds((current) => {
      if (current.includes(registrationId)) return current;
      if (current.length >= MAX_BULK_UPDATES) {
        setError(`Select no more than ${MAX_BULK_UPDATES} players in one bulk update.`);
        return current;
      }
      return [...current, registrationId];
    });
  }

  function clearSelection(): void {
    setSelectedIds([]);
  }

  async function saveDraft(
    card: TournamentCheckInRegistrant,
    draft: Draft
  ): Promise<void> {
    if (!apiBase || !accessToken || !snapshot) return;
    const substitute = snapshot.player_options.find(
      (player) => String(player.id) === draft.substitutePlayerId
    );
    const update = {
      expected_updated_at: draft.expectedUpdatedAt,
      attendance_status: draft.attendanceStatus,
      waiver_verified: draft.waiverVerified,
      approved_substitute_player_id: substitute?.id || null,
      notes: draft.notes.trim() || null
    } as const;
    const fingerprint = JSON.stringify(update);
    const existingOperation = operationKeys.current[card.registration_id];
    const operationKey =
      existingOperation?.fingerprint === fingerprint
        ? existingOperation.key
        : crypto.randomUUID();
    operationKeys.current[card.registration_id] = {
      fingerprint,
      key: operationKey
    };
    await updateAdminTournamentCheckIn({
      apiBase,
      clubId,
      tournamentId,
      registrationId: card.registration_id,
      dayId: snapshot.day_scope.selected_day_id,
      accessToken,
      input: {
        operation_key: operationKey,
        ...update
      }
    });
    delete operationKeys.current[card.registration_id];
  }

  async function applyBulkChanges(): Promise<string> {
    if (
      !apiBase ||
      !accessToken ||
      !snapshot ||
      !selectedIds.length ||
      !hasBulkChange ||
      savingId
    ) throw new Error("Select players and at least one check-in change before applying.");

    const selectedCards = snapshot.registrants.filter((card) =>
      selectedIdSet.has(card.registration_id)
    );
    const updates = selectedCards
      .map((card) => {
        const currentDraft = drafts[card.registration_id] || initialDraft(card);
        const update: {
          registration_id: string;
          expected_updated_at: string | null;
          attendance_status?: AttendanceStatus;
          waiver_verified?: boolean;
          notes?: string | null;
        } = {
          registration_id: card.registration_id,
          expected_updated_at: currentDraft.expectedUpdatedAt
        };
        if (
          bulkAttendance !== "NO_CHANGE" &&
          bulkAttendance !== attendanceStatus(card)
        ) {
          update.attendance_status = bulkAttendance;
        }
        const requestedWaiver =
          bulkWaiver === "NO_CHANGE" ? null : bulkWaiver === "VERIFY";
        if (
          requestedWaiver !== null &&
          requestedWaiver !== Boolean(card.waiver.verified)
        ) {
          update.waiver_verified = requestedWaiver;
        }
        const note = currentDraft.notes.trim() || null;
        if (note !== (String(card.check_in.notes || "").trim() || null)) {
          update.notes = note;
        }
        return update;
      })
      .filter(
        (update) =>
          update.attendance_status !== undefined ||
          update.waiver_verified !== undefined ||
          update.notes !== undefined
      );

    if (!updates.length) {
      const noChangeMessage = "The selected players already match those check-in values.";
      setMessage(noChangeMessage);
      return noChangeMessage;
    }

    setSavingId("__bulk__");
    setMessage(null);
    setError(null);

    try {
      const fingerprint = JSON.stringify({
        day_id: snapshot.day_scope.selected_day_id,
        updates
      });
      const operationKey =
        bulkOperation.current?.fingerprint === fingerprint
          ? bulkOperation.current.key
          : crypto.randomUUID();
      bulkOperation.current = { fingerprint, key: operationKey };
      const result = await updateAdminTournamentCheckInBulk({
        apiBase,
        clubId,
        tournamentId,
        dayId: snapshot.day_scope.selected_day_id,
        accessToken,
        input: {
          operation_key: operationKey,
          updates
        }
      });
      bulkOperation.current = null;
      const reloaded = await loadSnapshot();
      if (!reloaded) {
        const reloadWarning = `${result.message} The update succeeded, but the refreshed roster could not be loaded. Use Reload check-in before another action.`;
        setError(reloadWarning);
        return reloadWarning;
      }
      setMessage(result.message);
      return result.message;
    } catch (bulkError) {
      const reloaded = await loadSnapshot();
      const baseMessage = bulkError instanceof Error
        ? bulkError.message
        : "The bulk check-in did not complete.";
      const failureMessage = reloaded
        ? `${baseMessage} No partial batch is accepted; authoritative state was reloaded. Review and reselect players before another action.`
        : `${baseMessage} The authoritative reload also failed. Use Reload check-in before trying another action.`;
      setError(failureMessage);
      throw new Error(failureMessage, { cause: bulkError });
    } finally {
      setSavingId("");
    }
  }

  async function confirmBulkChanges() {
    const resultMessage = await applyBulkChanges();
    return actionSuccess("Bulk check-in updated", resultMessage);
  }

  function selectDay(dayId: string): void {
    if (!dayId || dayId === selectedDayId) return;
    setSelectedDayId(dayId);
    setSnapshot(null);
    setDrafts({});
    setSelectedIds([]);
    setBulkAttendance("NO_CHANGE");
    setBulkWaiver("NO_CHANGE");
    operationKeys.current = {};
    bulkOperation.current = null;
    setMessage(null);
    setError(null);
    preserveDayInUrl(dayId);
  }

  if (sessionLoading) return <p className={styles.notice}>Restoring the admin session…</p>;
  if (!accessToken) {
    return (
      <p className={`${styles.notice} ${styles.error}`} role="alert">
        Sign in at /admin/login to open authoritative check-in state.
        {sessionMessage ? ` ${sessionMessage}` : ""}
      </p>
    );
  }
  if (!apiBase) {
    return (
      <p className={`${styles.notice} ${styles.error}`} role="alert">
        The Tournament Admin API base URL is not configured.
      </p>
    );
  }

  return (
    <div
      aria-busy={loading || Boolean(savingId)}
      className={styles.workspace}
    >
      {error ? (
        <p className={`${styles.notice} ${styles.error}`} role="alert">{error}</p>
      ) : null}
      {message ? <p className={styles.notice} role="status">{message}</p> : null}
      {loading && !snapshot ? <p className={styles.notice}>Loading check-in truth…</p> : null}

      {snapshot ? (
        <>
          <section className={`${styles.panel} ${styles.dayPicker}`} aria-labelledby="day-picker-title">
            <div>
              <p className={styles.eyebrow}>Tournament-day scope</p>
              <h2 id="day-picker-title">Choose the day you are operating</h2>
              <p className={styles.muted}>
                Check-in counts, players, unresolved teams, and readiness below apply only
                to the selected day.
              </p>
            </div>
            <label className={styles.field}>
              Tournament day
              <select
                className={styles.select}
                disabled={loading || Boolean(savingId)}
                onChange={(event) => selectDay(event.target.value)}
                value={snapshot.day_scope.selected_day_id}
              >
                {snapshot.day_scope.available_days.map((day) => (
                  <option key={day.id} value={day.id}>
                    {day.event_date ? `${day.label} · ${day.event_date}` : day.label}
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section className={styles.summaryGrid} aria-label="Check-in summary">
            {(
              [
                ["Expected today", snapshot.summary.expected],
                ["Checked in", snapshot.summary.checked_in],
                ["Not checked in", snapshot.summary.not_checked_in],
                ["Absent", snapshot.summary.absent],
                ["Unresolved", snapshot.summary.unresolved]
              ] as const
            ).map(([label, value]) => (
              <article className={styles.summaryCard} key={label}>
                <p className={styles.eyebrow}>{label}</p>
                <strong>{value}</strong>
              </article>
            ))}
          </section>

          <section className={styles.panel} aria-labelledby="readiness-title">
            <div className={styles.sectionHeader}>
              <div>
                <h2 id="readiness-title">Event-day readiness</h2>
                <p className={styles.muted}>
                  Readiness for {selectedDayLabel}. Schedule and draw checks are
                  authoritative; staffing remains a human review.
                </p>
              </div>
              <button
                className={`${styles.button} ${styles.secondaryButton}`}
                disabled={loading || Boolean(savingId)}
                onClick={() => void loadSnapshot()}
                type="button"
              >
                {loading ? "Reloading…" : "Reload check-in"}
              </button>
            </div>
            <div className={styles.readinessGrid}>
              {(
                [
                  ["Dates and courts", snapshot.readiness.schedule],
                  ["Draws", snapshot.readiness.draws],
                  ["Staffing", snapshot.readiness.staffing]
                ] as const
              ).map(([title, readiness]) => (
                <article className={styles.registrantCard} key={title}>
                  <div className={styles.cardHeader}>
                    <h3>{title}</h3>
                    <span className={toneClass(readiness.status)}>
                      {statusLabel(readiness.status)}
                    </span>
                  </div>
                  {"timezone" in readiness && readiness.timezone ? (
                    <p className={styles.muted}>Timezone: {readiness.timezone}</p>
                  ) : null}
                  {readiness.blockers.length ? (
                    <ul className={styles.blockerList}>
                      {readiness.blockers.map((blocker) => (
                        <li className={styles.blockerRow} key={blocker.code}>
                          <strong>{blocker.title}</strong><br />{blocker.detail}
                        </li>
                      ))}
                    </ul>
                  ) : <p className={styles.muted}>Complete.</p>}
                </article>
              ))}
            </div>
          </section>

          {snapshot.unresolved_participants.length ? (
            <section className={styles.panel} aria-labelledby="unresolved-title">
              <h2 id="unresolved-title">Unresolved participants</h2>
              <ul className={styles.blockerList}>
                {snapshot.unresolved_participants.map((row) => (
                  <li
                    className={styles.blockerRow}
                    key={`${row.registration_id}:${row.selection_id}`}
                  >
                    <strong>{row.registration_name} · {row.event_label}</strong><br />
                    {row.detail}
                  </li>
                ))}
              </ul>
            </section>
          ) : null}

          {(snapshot.registration_follow_up || []).length ? (
            <section className={styles.panel} aria-labelledby="registration-follow-up-title">
              <h2 id="registration-follow-up-title">Registered but not rostered</h2>
              <p className={styles.muted}>
                These active registration entries are either off the authoritative draw
                roster for {selectedDayLabel} or do not map uniquely to it. They remain
                visible for follow-up, but do not count toward Expected Today. Any roster
                integrity problem is reported separately as a draw-readiness blocker.
              </p>
              <ul className={styles.blockerList}>
                {snapshot.registration_follow_up.map((row) => (
                  <li
                    className={styles.blockerRow}
                    key={`${row.registration_id}:${row.selection_id}`}
                  >
                    <strong>{row.registration_name} · {row.event_label}</strong><br />
                    {row.detail}
                  </li>
                ))}
              </ul>
            </section>
          ) : null}

          <section className={styles.panel} aria-labelledby="players-title">
            <div className={styles.sectionHeader}>
              <div>
                <h2 id="players-title">Player check-in</h2>
                <p className={styles.muted}>
                  Showing only players scheduled for {selectedDayLabel}. Payment status is
                  read from offline tracking and cannot be edited here.
                </p>
              </div>
              <span className={styles.badge}>{filtered.length} shown</span>
            </div>
            <div className={styles.toolbar}>
              <label className={styles.field}>
                Search players
                <input
                  className={styles.input}
                  onChange={(event) => {
                    setSearch(event.target.value);
                    clearSelection();
                    setBulkAttendance("NO_CHANGE");
                    setBulkWaiver("NO_CHANGE");
                  }}
                  placeholder="Name, division, or partner"
                  type="search"
                  value={search}
                />
              </label>
              <label className={styles.field}>
                View
                <select
                  className={styles.select}
                  onChange={(event) => {
                    setFilter(event.target.value as Filter);
                    clearSelection();
                    setBulkAttendance("NO_CHANGE");
                    setBulkWaiver("NO_CHANGE");
                  }}
                  value={filter}
                >
                  <option value="all">All scheduled</option>
                  <option value="expected">Not checked in</option>
                  <option value="checked_in">Checked in</option>
                  <option value="absent">Absent</option>
                  <option value="unresolved">Unresolved teams</option>
                </select>
              </label>
            </div>

            <div className={styles.bulkActionBar} role="group" aria-label="Bulk check-in actions">
              <div className={styles.selectionControls}>
                <strong>{selectedIds.length} selected</strong>
                <div className={styles.selectionButtons}>
                  <button
                    className={`${styles.button} ${styles.secondaryButton} ${styles.smallButton}`}
                    disabled={!selectableFilteredIds.length || Boolean(savingId)}
                    onClick={selectAllShown}
                    type="button"
                  >
                    Select all shown
                  </button>
                  <button
                    className={`${styles.button} ${styles.secondaryButton} ${styles.smallButton}`}
                    disabled={!selectedIds.length || Boolean(savingId)}
                    onClick={clearSelection}
                    type="button"
                  >
                    Clear
                  </button>
                </div>
              </div>
              <label className={styles.field}>
                Attendance action
                <select
                  className={styles.select}
                  disabled={Boolean(savingId)}
                  onChange={(event) =>
                    setBulkAttendance(event.target.value as BulkAttendanceAction)
                  }
                  value={bulkAttendance}
                >
                  <option value="NO_CHANGE">No attendance change</option>
                  <option value="CHECKED_IN">Mark checked in</option>
                  <option value="ABSENT">Mark absent</option>
                  <option value="EXPECTED">Reset to not checked in</option>
                </select>
              </label>
              <label className={styles.field}>
                Waiver action
                <select
                  className={styles.select}
                  disabled={Boolean(savingId)}
                  onChange={(event) =>
                    setBulkWaiver(event.target.value as BulkWaiverAction)
                  }
                  value={bulkWaiver}
                >
                  <option value="NO_CHANGE">No waiver change</option>
                  <option value="VERIFY">Mark waiver verified</option>
                  <option value="CLEAR">Clear waiver verification</option>
                </select>
              </label>
              {bulkActionIsDestructive ? (
                <ConfirmAction
                  busy={savingId === "__bulk__"}
                  confirmLabel="Yes, apply these changes"
                  confirmationText="APPLY BULK CHECK-IN"
                  description={`This will update ${selectedIds.length} selected player${selectedIds.length === 1 ? "" : "s"} in one atomic operation.`}
                  disabled={!selectedIds.length || !hasBulkChange || Boolean(savingId)}
                  onConfirm={confirmBulkChanges}
                  preview={<p>{bulkActionSummary || "No check-in changes selected."}</p>}
                  title="Apply bulk check-in changes?"
                  tone="danger"
                  triggerLabel={savingId === "__bulk__"
                    ? `Updating ${selectedIds.length} selected…`
                    : `Apply to ${selectedIds.length || 0} selected`}
                  workingLabel={`Updating ${selectedIds.length} selected…`}
                />
              ) : (
                <button
                  className={styles.button}
                  disabled={!selectedIds.length || !hasBulkChange || Boolean(savingId)}
                  onClick={() => void applyBulkChanges().catch(() => undefined)}
                  type="button"
                >
                  {savingId === "__bulk__"
                    ? `Updating ${selectedIds.length} selected…`
                    : `Apply to ${selectedIds.length || 0} selected`}
                </button>
              )}
              <p className={styles.bulkHelp}>
                Check-in is tracking only. Tournament-day play remains available regardless
                of attendance, waiver, or payment status. Roster changes do not happen at check-in.
                Correct the authoritative draw or four-player team roster before play. Apply up to
                {MAX_BULK_UPDATES} players at once.
              </p>
            </div>

            <div className={styles.tableWrap}>
              <table className={styles.playerTable}>
                <caption className={styles.visuallyHidden}>
                  Player check-in roster for {selectedDayLabel}
                </caption>
                <thead>
                  <tr>
                    <th scope="col"><span className={styles.visuallyHidden}>Select</span></th>
                    <th scope="col">Player</th>
                    <th scope="col">Events and partners</th>
                    <th scope="col">Attendance</th>
                    <th scope="col">Waiver</th>
                    <th scope="col">Payment</th>
                    <th scope="col">Operator note</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((card) => {
                    const draft = drafts[card.registration_id] || initialDraft(card);
                    const currentAttendance = draft.attendanceStatus;
                    const substitutePending = legacySubstitutePending(card, draft);
                    const attendingName = substitutePending
                      ? card.attendee.name
                      : card.original_registrant.name;
                    const selected = selectedIdSet.has(card.registration_id);
                    return (
                      <tr
                        className={selected ? styles.selectedRow : undefined}
                        key={card.registration_id}
                      >
                        <td className={styles.selectCell}>
                          <input
                            aria-label={`Select ${attendingName}`}
                            checked={selected}
                            className={styles.selectionCheckbox}
                            disabled={
                              Boolean(savingId) ||
                              substitutePending ||
                              (!selected && selectedIds.length >= MAX_BULK_UPDATES)
                            }
                            onChange={() => toggleSelected(card.registration_id)}
                            type="checkbox"
                          />
                        </td>
                        <th className={styles.playerCell} scope="row">
                          <strong>{attendingName}</strong>
                          {substitutePending ? (
                            <>
                              <span className={styles.rowMeta}>
                                Legacy saved substitute for {card.original_registrant.name}
                              </span>
                              <button
                                className={`${styles.button} ${styles.secondaryButton} ${styles.smallButton}`}
                                disabled={Boolean(savingId)}
                                onClick={() => void restoreOriginalRegistrant(card)}
                                type="button"
                              >
                                Restore original registrant
                              </button>
                            </>
                          ) : null}
                          {card.blockers.length ? (
                            <details className={styles.rowDetails}>
                              <summary>{card.blockers.length} review item{card.blockers.length === 1 ? "" : "s"}</summary>
                              <ul className={styles.compactList}>
                                {card.blockers.map((blocker, index) => (
                                  <li key={`${blocker.code}:${index}`}>
                                    <strong>{blocker.title}</strong>: {blocker.detail}
                                  </li>
                                ))}
                              </ul>
                            </details>
                          ) : null}
                        </th>
                        <td>
                          <ul className={styles.compactList}>
                            {card.events.map((event) => (
                              <li key={event.selection_id}>
                                <strong>{event.event_label}</strong>
                                {event.partner_name ? ` with ${event.partner_name}` : ""}
                                {!event.partner_name && event.entered_partner_name
                                  ? ` · entered partner: ${event.entered_partner_name}`
                                  : ""}
                              </li>
                            ))}
                          </ul>
                        </td>
                        <td>
                          <span
                            className={
                              currentAttendance === "CHECKED_IN"
                                ? `${styles.badge} ${styles.complete}`
                                : currentAttendance === "ABSENT"
                                  ? `${styles.badge} ${styles.blocked}`
                                  : styles.badge
                            }
                          >
                            {attendanceLabel(currentAttendance)}
                          </span>
                        </td>
                        <td>
                          <span
                            className={
                              draft.waiverVerified
                                ? `${styles.badge} ${styles.complete}`
                                : `${styles.badge} ${styles.review}`
                            }
                          >
                            {draft.waiverVerified ? "Verified" : "Not verified"}
                          </span>
                        </td>
                        <td>
                          <span
                            className={
                              card.payment.ready
                                ? `${styles.badge} ${styles.complete}`
                                : `${styles.badge} ${styles.review}`
                            }
                          >
                            {card.payment.status}
                          </span>
                        </td>
                        <td className={styles.noteCell}>
                          <label>
                            <span className={styles.visuallyHidden}>
                              Operator note for {attendingName}
                            </span>
                            <input
                              className={styles.noteInput}
                              disabled={Boolean(savingId) || substitutePending}
                              maxLength={1000}
                              onChange={(event) => {
                                patchDraft(card.registration_id, { notes: event.target.value });
                                ensureSelected(card.registration_id);
                              }}
                              placeholder="Optional"
                              type="text"
                              value={draft.notes}
                            />
                          </label>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {!filtered.length ? (
                <p className={styles.muted}>
                  {snapshot.registrants.length
                    ? "No scheduled players match these filters."
                    : `No players are scheduled for ${selectedDayLabel}.`}
                </p>
              ) : null}
            </div>
          </section>

          {snapshot.inactive_registrants.length ? (
            <details className={`${styles.panel} ${styles.details}`}>
              <summary>
                Cancelled or inactive registrations ({snapshot.inactive_registrants.length})
              </summary>
              <ul className={styles.plainList}>
                {snapshot.inactive_registrants.map((row) => (
                  <li key={row.registration_id}>
                    {row.name} · {statusLabel(row.registration_status)}
                  </li>
                ))}
              </ul>
            </details>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

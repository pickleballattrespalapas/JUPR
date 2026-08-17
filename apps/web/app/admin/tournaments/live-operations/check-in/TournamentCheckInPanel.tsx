"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  fetchAdminTournamentCheckIn,
  updateAdminTournamentCheckIn
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
  const [loading, setLoading] = useState(false);
  const [savingId, setSavingId] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const operationKeys = useRef<Record<string, { fingerprint: string; key: string }>>({});

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
      if (!apiBase || !accessToken || !tournamentId) return;
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
        if (signal?.aborted) return;
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
      } catch (loadError) {
        if (signal?.aborted) return;
        setSnapshot(null);
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Unable to load authoritative tournament check-in state."
        );
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

  function changeAttendee(registrationId: string, substitutePlayerId: string): void {
    patchDraft(registrationId, {
      substitutePlayerId,
      attendanceStatus: "EXPECTED",
      waiverVerified: false
    });
    setMessage(
      "Attending player changed. Reconfirm check-in and the attending player's waiver before saving."
    );
  }

  async function save(card: TournamentCheckInRegistrant): Promise<void> {
    if (!apiBase || !accessToken || !snapshot) return;
    const draft = drafts[card.registration_id] || initialDraft(card);
    const substitute = snapshot.player_options.find(
      (player) => String(player.id) === draft.substitutePlayerId
    );
    setSavingId(card.registration_id);
    setMessage(null);
    setError(null);
    try {
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
      const result = await updateAdminTournamentCheckIn({
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
      setMessage(result.message);
      await loadSnapshot();
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : "Unable to save check-in. Reload before trying again."
      );
    } finally {
      setSavingId("");
    }
  }

  function selectDay(dayId: string): void {
    if (!dayId || dayId === selectedDayId) return;
    setSelectedDayId(dayId);
    setSnapshot(null);
    setDrafts({});
    operationKeys.current = {};
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
                  ["Dates, courts, and times", snapshot.readiness.schedule],
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
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Name, division, or partner"
                  type="search"
                  value={search}
                />
              </label>
              <label className={styles.field}>
                View
                <select
                  className={styles.select}
                  onChange={(event) => setFilter(event.target.value as Filter)}
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

            <div className={styles.cardList}>
              {filtered.map((card) => {
                const draft = drafts[card.registration_id] || initialDraft(card);
                const currentAttendance = draft.attendanceStatus;
                const saving = savingId === card.registration_id;
                const draftSubstitute = snapshot.player_options.find(
                  (player) => String(player.id) === draft.substitutePlayerId
                );
                const savedAttendeeUnavailable = Boolean(
                  draft.substitutePlayerId &&
                  (!draftSubstitute || !card.substitution.allowed)
                );
                const attendingName =
                  draftSubstitute?.name ||
                  (savedAttendeeUnavailable
                    ? card.attendee.name
                    : card.original_registrant.name);
                return (
                  <article className={styles.registrantCard} key={card.registration_id}>
                    <div className={styles.cardHeader}>
                      <div>
                        <h3>{attendingName}</h3>
                        {draftSubstitute ? (
                          <p className={styles.muted}>
                            Approved substitute for {card.original_registrant.name}
                          </p>
                        ) : <p className={styles.muted}>Registered player</p>}
                      </div>
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
                    </div>

                    <ul className={styles.eventList}>
                      {card.events.map((event) => (
                        <li className={styles.eventRow} key={event.selection_id}>
                          <strong>{event.event_label}</strong> · {statusLabel(event.team_state)}
                          {event.partner_name ? ` with ${event.partner_name}` : ""}
                          {!event.partner_name && event.entered_partner_name
                            ? ` · entered partner: ${event.entered_partner_name}`
                            : ""}
                        </li>
                      ))}
                    </ul>

                    <div className={styles.controlGrid}>
                      <div className={styles.checks}>
                        <fieldset className={styles.attendanceControls}>
                          <legend>Attendance</legend>
                          {(
                            [
                              ["EXPECTED", "Not checked in"],
                              ["CHECKED_IN", "Checked in"],
                              ["ABSENT", "Mark absent"]
                            ] as const
                          ).map(([value, label]) => (
                            <label className={styles.checkLabel} key={value}>
                              <input
                                checked={currentAttendance === value}
                                disabled={saving || savedAttendeeUnavailable}
                                name={`attendance-${card.registration_id}`}
                                onChange={() =>
                                  patchDraft(card.registration_id, {
                                    attendanceStatus: value
                                  })
                                }
                                type="radio"
                                value={value}
                              />
                              {label}
                            </label>
                          ))}
                        </fieldset>
                        <label className={styles.checkLabel}>
                          <input
                            checked={draft.waiverVerified}
                            disabled={saving || savedAttendeeUnavailable}
                            onChange={(event) =>
                              patchDraft(card.registration_id, {
                                waiverVerified: event.target.checked
                              })
                            }
                            type="checkbox"
                          />
                          Waiver verified for attending player ({attendingName})
                        </label>
                        <span
                          className={
                            card.payment.ready
                              ? `${styles.badge} ${styles.complete}`
                              : `${styles.badge} ${styles.review}`
                          }
                        >
                          Offline payment: {card.payment.status}
                        </span>
                      </div>
                      <div>
                        <label className={styles.field}>
                          Approved substitute
                          <select
                            className={styles.select}
                            disabled={
                              saving ||
                              (!card.substitution.allowed &&
                                !card.attendee.is_approved_substitute)
                            }
                            onChange={(event) =>
                              changeAttendee(card.registration_id, event.target.value)
                            }
                            value={draft.substitutePlayerId}
                          >
                            <option value="">Original registrant is attending</option>
                            {savedAttendeeUnavailable ? (
                              <option disabled value={draft.substitutePlayerId}>
                                Saved attendee is unavailable — choose again
                              </option>
                            ) : null}
                            {(card.substitution.allowed
                              ? snapshot.player_options
                              : []
                            )
                              .filter(
                                (player) => player.id !== card.original_registrant.player_id
                              )
                              .map((player) => (
                                <option key={player.id} value={player.id}>{player.name}</option>
                              ))}
                          </select>
                        </label>
                        {!card.substitution.allowed ? (
                          <p className={`${styles.notice} ${styles.error}`} role="status">
                            <strong>Substitute assignment unavailable.</strong>{" "}
                            {card.substitution.blocker.detail}
                          </p>
                        ) : null}
                        {savedAttendeeUnavailable ? (
                          <p className={`${styles.notice} ${styles.error}`} role="alert">
                            Saved attendee is unavailable. Choose the original registrant
                            or an active substitute before saving.
                          </p>
                        ) : null}
                        <label className={styles.field}>
                          Operator note
                          <textarea
                            className={styles.textarea}
                            disabled={saving}
                            maxLength={1000}
                            onChange={(event) =>
                              patchDraft(card.registration_id, {
                                notes: event.target.value
                              })
                            }
                            value={draft.notes}
                          />
                        </label>
                      </div>
                    </div>

                    {card.blockers.length ? (
                      <ul className={styles.blockerList}>
                        {card.blockers.map((blocker, index) => (
                          <li
                            className={styles.blockerRow}
                            key={`${blocker.code}:${index}`}
                          >
                            <strong>{blocker.title}</strong><br />{blocker.detail}
                          </li>
                        ))}
                      </ul>
                    ) : null}
                    <div className={styles.actions}>
                      <button
                        className={styles.button}
                        disabled={Boolean(savingId) || savedAttendeeUnavailable}
                        onClick={() => void save(card)}
                        type="button"
                      >
                        {saving ? "Saving…" : "Save check-in"}
                      </button>
                    </div>
                  </article>
                );
              })}
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

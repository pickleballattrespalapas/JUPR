"use client";

import { useCallback, useMemo, useState } from "react";
import {
  PublicRegistrationDay,
  PublicRegistrationEvent,
  PublicRegistrationPlayer,
  PublicRegistrationProfileResolutionResponse,
  PublicRegistrationSelectionPayload,
  resolveClubTournamentRegistrationProfile,
  submitClubTournamentRegistration
} from "@/lib/tournamentRegistrationApi";
import {
  TournamentCommerceCatalog,
  TournamentCommerceQuote,
  TournamentCommerceSelection
} from "@/lib/tournamentCommerceApi";
import {
  formatRegistrationRating,
  publicEventCapacityLabel,
  publicEventEligibilityReason,
  publicEventFamilyKey,
  publicEventFormatLabel,
  publicScoringLabel,
  publicTournamentDayLabel,
  publicTournamentEventLabel
} from "@/lib/tournamentRegistrationEligibility";
import { createPublicFourPlayerTeam } from "@/lib/tournamentTeamCompetitionApi";
import FourPlayerTeamRegistrationCard, {
  TEAM_SLOTS,
  TeamRegistrationDraft,
  newTeamRegistrationDraft,
  validateTeamRegistrationDraft
} from "@/components/tournaments/FourPlayerTeamRegistrationCard";
import EditLinkRequestForm from "./EditLinkRequestForm";
import TournamentCommerceChooser from "./TournamentCommerceChooser";

type TournamentRegistrationFormProps = {
  clubSlug: string;
  tournamentId: string;
  registrationSlug?: string | null;
  registrationOpen: boolean;
  registrationClosedReason?: string | null;
  timeZone?: string | null;
  days: PublicRegistrationDay[];
  events: PublicRegistrationEvent[];
  commerce?: TournamentCommerceCatalog | null;
};

type ContactState = {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  age: string;
  gender: string;
  notes: string;
};

type ProfileState = {
  candidateId: string;
  displayName: string;
  duprId: string;
  doublesSkill: string;
  singlesSkill: string;
};

type PartnerState = {
  mode: "NONE" | "HAS_PARTNER" | "NEEDS_PARTNER";
  name: string;
  email: string;
  phone: string;
  duprId: string;
  skill: string;
  age: string;
  gender: string;
  note: string;
  showOnBoard: boolean;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const inputStyle = {
  width: "100%",
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

const primaryButtonStyle = {
  padding: "0.7rem 1rem",
  borderRadius: "10px",
  border: "1px solid #0f172a",
  background: "#0f172a",
  color: "white",
  fontWeight: 800,
  cursor: "pointer"
};

const secondaryButtonStyle = {
  ...primaryButtonStyle,
  background: "white",
  color: "#0f172a",
  borderColor: "#cbd5e1"
};

const doublesEventTypes = new Set(["DOUBLES", "GENDER_DOUBLES", "MIXED_DOUBLES", "MIXED"]);

function numericValue(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function isDoublesEvent(event: PublicRegistrationEvent): boolean {
  return Boolean(event.partner_required) || doublesEventTypes.has(String(event.event_type || "").toUpperCase());
}

function emptyPartnerState(event: PublicRegistrationEvent): PartnerState {
  return {
    mode: event.partner_required ? "NEEDS_PARTNER" : "NONE",
    name: "",
    email: "",
    phone: "",
    duprId: "",
    skill: "",
    age: "",
    gender: "",
    note: "",
    showOnBoard: false
  };
}

function eventMeta(event: PublicRegistrationEvent): string {
  const pieces = [
    event.skill_label,
    event.age_label,
    publicEventFormatLabel(event.event_format),
    publicScoringLabel(event.scoring)
  ]
    .filter((item) => item && String(item).trim())
    .map(String);
  if (event.price_usd != null) pieces.push(`$${Number(event.price_usd).toFixed(2)}`);
  const capacity = publicEventCapacityLabel(event);
  if (capacity) pieces.push(capacity);
  if (
    String(event.eligibility_mode || "").toUpperCase() ===
      "COMBINED_RATING_CAP" &&
    event.combined_rating_cap != null
  ) {
    pieces.push(
      `Combined rating must be below ${formatRegistrationRating(Number(event.combined_rating_cap))}`
    );
  }
  if (
    String(event.competition_format || "").toUpperCase() ===
    "FOUR_PLAYER_TEAM"
  ) {
    pieces.push("Four-player team");
  }
  return pieces.join(" • ");
}

function scheduledDaysLabel(
  event: PublicRegistrationEvent,
  daysById: Map<string, PublicRegistrationDay>
): string {
  const scheduledIds = event.scheduled_day_ids?.length
    ? event.scheduled_day_ids
    : [event.registration_day_id];
  return scheduledIds
    .map((dayId) => daysById.get(dayId))
    .filter(Boolean)
    .map((day) =>
      day ? publicTournamentDayLabel(day.label, day.event_date) : null
    )
    .join(" · ");
}

function candidateLabel(candidate: PublicRegistrationPlayer): string {
  const doubles = candidate.doubles_skill == null
    ? "Doubles not set"
    : `Doubles ${formatRegistrationRating(Number(candidate.doubles_skill))}`;
  const singles = candidate.singles_skill == null
    ? "Singles not set"
    : `Singles ${formatRegistrationRating(Number(candidate.singles_skill))}`;
  return `${candidate.display_name} · ${doubles} · ${singles}`;
}

export default function TournamentRegistrationForm({
  clubSlug,
  tournamentId,
  registrationSlug,
  registrationOpen,
  registrationClosedReason,
  timeZone,
  days,
  events,
  commerce
}: TournamentRegistrationFormProps) {
  const [mode, setMode] = useState<"choose" | "new" | "edit">("choose");
  const [step, setStep] = useState(1);
  const [contact, setContact] = useState<ContactState>({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    age: "",
    gender: "",
    notes: ""
  });
  const [profile, setProfile] = useState<ProfileState>({
    candidateId: "",
    displayName: "",
    duprId: "",
    doublesSkill: "",
    singlesSkill: ""
  });
  const [resolution, setResolution] = useState<PublicRegistrationProfileResolutionResponse | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [partnerDetails, setPartnerDetails] = useState<Record<string, PartnerState>>({});
  const [partnerConsent, setPartnerConsent] = useState(false);
  const [termsAccepted, setTermsAccepted] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recoveryEmail, setRecoveryEmail] = useState("");
  const [commerceSelections, setCommerceSelections] = useState<
    TournamentCommerceSelection[]
  >([]);
  const [commerceQuote, setCommerceQuote] =
    useState<TournamentCommerceQuote | null>(null);
  const [commerceIdempotencyKey, setCommerceIdempotencyKey] = useState(() =>
    crypto.randomUUID()
  );
  const [teamDrafts, setTeamDrafts] = useState<
    Record<string, TeamRegistrationDraft>
  >({});
  const [savedRegistration, setSavedRegistration] = useState<{
    registrationId: string;
    confirmationToken: string;
    emailStatus?: string | null;
  } | null>(null);
  const [createdTeamEventIds, setCreatedTeamEventIds] = useState<string[]>([]);

  const selectableEvents = useMemo(() => events.filter((event) => event.selectable), [events]);
  const eventById = useMemo(() => new Map(events.map((event) => [event.id, event])), [events]);
  const daysById = useMemo(() => new Map(days.map((day) => [day.id, day])), [days]);
  const groupedEvents = useMemo(
    () =>
      days
        .map((day) => ({ day, events: selectableEvents.filter((event) => event.registration_day_id === day.id) }))
        .filter((row) => row.events.length > 0),
    [days, selectableEvents]
  );
  const eligibilityProfile = useMemo(
    () => ({
      gender: contact.gender,
      age: numericValue(contact.age),
      doublesSkill: numericValue(profile.doublesSkill),
      singlesSkill: numericValue(profile.singlesSkill)
    }),
    [contact.gender, contact.age, profile.doublesSkill, profile.singlesSkill]
  );
  const totalPrice = selectedIds.reduce((sum, id) => sum + Number(eventById.get(id)?.price_usd || 0), 0);
  const needsPartnerBoardConsent = selectedIds.some(
    (id) => partnerDetails[id]?.mode === "NEEDS_PARTNER" && partnerDetails[id]?.showOnBoard
  );
  const selectedTeamEvents = selectedIds
    .map((id) => eventById.get(id))
    .filter(
      (event): event is PublicRegistrationEvent =>
        Boolean(event) &&
        String(event?.competition_format || "").toUpperCase() ===
          "FOUR_PLAYER_TEAM"
    );

  function resetWizard(nextMode: "choose" | "new" | "edit" = "choose") {
    setMode(nextMode);
    setStep(1);
    setResolution(null);
    setSelectedIds([]);
    setPartnerDetails({});
    setPartnerConsent(false);
    setTermsAccepted(false);
    setCommerceSelections([]);
    setCommerceQuote(null);
    setCommerceIdempotencyKey(crypto.randomUUID());
    setTeamDrafts({});
    setSavedRegistration(null);
    setCreatedTeamEventIds([]);
    setError(null);
  }

  const updateCommerceReview = useCallback(
    (
      selections: TournamentCommerceSelection[],
      quote: TournamentCommerceQuote | null
    ) => {
      const priorFingerprint = commerceQuote?.quote_fingerprint || null;
      const nextFingerprint = quote?.quote_fingerprint || null;
      setCommerceSelections(selections);
      setCommerceQuote(quote);
      if (
        priorFingerprint &&
        priorFingerprint !== nextFingerprint
      ) {
        setCommerceIdempotencyKey(crypto.randomUUID());
      }
    },
    [commerceQuote?.quote_fingerprint]
  );

  function updateContact(key: keyof ContactState, value: string) {
    setContact((current) => ({ ...current, [key]: value }));
  }

  function updateProfile(key: keyof ProfileState, value: string) {
    setProfile((current) => ({ ...current, [key]: value }));
  }

  function updatePartner(eventId: string, patch: Partial<PartnerState>) {
    const event = eventById.get(eventId);
    if (!event) return;
    setPartnerDetails((current) => ({
      ...current,
      [eventId]: { ...(current[eventId] || emptyPartnerState(event)), ...patch }
    }));
  }

  function toggleEvent(eventId: string, checked: boolean) {
    const nextEvent = eventById.get(eventId);
    if (!nextEvent) return;
    setSelectedIds((current) => {
      if (!checked) return current.filter((id) => id !== eventId);
      const sameFamily = current.filter((id) => {
        const row = eventById.get(id);
        return row && publicEventFamilyKey(row) === publicEventFamilyKey(nextEvent);
      });
      if (sameFamily.length) {
        setPartnerDetails((currentDetails) => {
          const nextDetails = { ...currentDetails };
          sameFamily.forEach((id) => delete nextDetails[id]);
          return nextDetails;
        });
      }
      return [...current.filter((id) => !sameFamily.includes(id) && id !== eventId), eventId];
    });
    if (checked) updatePartner(eventId, {});
    if (
      checked &&
      String(nextEvent.competition_format || "").toUpperCase() ===
        "FOUR_PLAYER_TEAM"
    ) {
      setTeamDrafts((current) => ({
        ...current,
        [eventId]:
          current[eventId] || newTeamRegistrationDraft(contact.gender)
      }));
    }
  }

  async function resolveProfile() {
    setError(null);
    const age = numericValue(contact.age);
    if (!contact.firstName.trim() || !contact.lastName.trim() || !contact.email.trim() || age == null || !contact.gender) {
      setError("First name, last name, email, age, and gender are required.");
      return;
    }
    if (age < 1 || age > 120) {
      setError("Age must be between 1 and 120.");
      return;
    }
    setPending(true);
    const response = await resolveClubTournamentRegistrationProfile(clubSlug, {
      tournament_id: tournamentId,
      registration_slug: registrationSlug || null,
      first_name: contact.firstName,
      last_name: contact.lastName,
      email: contact.email,
      age,
      gender: contact.gender,
      website: ""
    });
    setPending(false);
    if (response.error || !response.data) {
      setError(response.error || "We couldn’t continue your registration. Please try again.");
      return;
    }
    if (!response.data.can_start_new) {
      setRecoveryEmail(contact.email.trim());
      setMode("edit");
      setError(
        response.data.status === "closed"
          ? response.data.registration_closed_reason || "Registration is closed."
          : null
      );
      return;
    }
    setResolution(response.data);
    setProfile((current) => ({
      ...current,
      displayName: current.displayName || `${contact.firstName.trim()} ${contact.lastName.trim()}`.trim()
    }));
    setStep(2);
  }

  function selectCandidate(candidate: PublicRegistrationPlayer | null) {
    if (!candidate) {
      setProfile((current) => ({ ...current, candidateId: "" }));
      return;
    }
    setProfile({
      candidateId: candidate.id,
      displayName: candidate.display_name,
      duprId: String(candidate.dupr_id || ""),
      doublesSkill: candidate.doubles_skill == null ? "" : String(candidate.doubles_skill),
      singlesSkill: candidate.singles_skill == null ? "" : String(candidate.singles_skill)
    });
  }

  function validateSelections(): string | null {
    if (!selectedIds.length) return "Select at least one event.";
    for (const id of selectedIds) {
      const event = eventById.get(id);
      if (!event) return "A selected event is no longer available.";
      const reason = publicEventEligibilityReason(event, eligibilityProfile);
      if (reason) return `${event.division_name}: ${reason}`;
      if (
        String(event.competition_format || "").toUpperCase() ===
        "FOUR_PLAYER_TEAM"
      ) {
        const teamError = validateTeamRegistrationDraft(
          teamDrafts[id],
          contact.email,
          contact.gender
        );
        if (teamError) return `${event.division_name}: ${teamError}`;
        continue;
      }
      const partner = partnerDetails[id] || emptyPartnerState(event);
      if (event.partner_required && !["HAS_PARTNER", "NEEDS_PARTNER"].includes(partner.mode)) {
        return `${event.division_name}: choose whether you have or need a partner.`;
      }
      if (String(event.event_type || "").toUpperCase() === "SINGLES" && partner.mode !== "NONE") {
        return `${event.division_name} does not accept partner information.`;
      }
      if (partner.mode === "HAS_PARTNER") {
        if (!partner.name.trim() || !partner.email.trim() || !partner.age.trim() || !partner.gender || !partner.skill.trim()) {
          return `${event.division_name}: partner name, email, age, gender, and starting skill are required.`;
        }
        const partnerSkill = numericValue(partner.skill);
        if (partnerSkill == null || partnerSkill < 1 || partnerSkill > 7) {
          return `${event.division_name}: partner starting skill must be between 1 and 7.`;
        }
        if (partner.email.trim().toLowerCase() === contact.email.trim().toLowerCase()) {
          return "A player cannot register themselves as their own partner.";
        }
        if (
          String(event.eligibility_mode || "").toUpperCase() ===
            "COMBINED_RATING_CAP" &&
          event.combined_rating_cap != null
        ) {
          const playerRating = numericValue(profile.doublesSkill);
          const partnerRating = numericValue(partner.skill);
          if (
            playerRating != null &&
            partnerRating != null &&
            Number((playerRating + partnerRating).toFixed(2)) >=
              Number(event.combined_rating_cap)
          ) {
            return `${event.division_name}: combined rating must be below ${formatRegistrationRating(
              Number(event.combined_rating_cap)
            )}.`;
          }
        }
      }
    }
    return null;
  }

  function advanceFromProfile() {
    setError(null);
    if (!profile.displayName.trim()) {
      setError("Enter the display name tournament staff should use.");
      return;
    }
    for (const [label, value] of [
      ["Doubles skill", profile.doublesSkill],
      ["Singles skill", profile.singlesSkill]
    ]) {
      const parsed = numericValue(value);
      if (parsed != null && (parsed < 1 || parsed > 7)) {
        setError(`${label} must be between 1 and 7.`);
        return;
      }
    }
    setStep(3);
  }

  function advanceFromEvents() {
    setError(null);
    const validationError = validateSelections();
    if (validationError) {
      setError(validationError);
      return;
    }
    if (commerce?.available && !commerceQuote) {
      setError(
        "Update your total before continuing. You can choose zero extras."
      );
      return;
    }
    setStep(4);
  }

  function selectionsPayload(): PublicRegistrationSelectionPayload[] {
    return selectedIds.map((eventId) => {
      const event = eventById.get(eventId)!;
      const teamFormat =
        String(event.competition_format || "").toUpperCase() ===
        "FOUR_PLAYER_TEAM";
      const partner = teamFormat
        ? { ...emptyPartnerState(event), mode: "NONE" as const }
        : partnerDetails[eventId] || emptyPartnerState(event);
      return {
        event_option_id: eventId,
        registration_day_id: event.registration_day_id,
        partner_mode: partner.mode,
        partner_name: partner.name,
        partner_email: partner.email,
        partner_phone: partner.phone,
        partner_dupr_id: partner.duprId,
        partner_skill: numericValue(partner.skill),
        partner_age: numericValue(partner.age),
        partner_gender: partner.gender,
        partner_note: partner.note,
        show_on_partner_board: partner.mode === "NEEDS_PARTNER" && partner.showOnBoard
      };
    });
  }

  function confirmationPath(
    saved: {
      registrationId: string;
      confirmationToken: string;
      emailStatus?: string | null;
    },
    teamSetupNeedsAttention = false
  ): string {
    const query = new URLSearchParams({
      confirmation_token: saved.confirmationToken
    });
    if (saved.emailStatus) query.set("email_status", saved.emailStatus);
    if (registrationSlug) query.set("tournament", registrationSlug);
    else query.set("tournament_id", tournamentId);
    if (teamSetupNeedsAttention) query.set("team_setup", "attention");
    return `/clubs/${clubSlug}/tournament-registration/confirmation?${query.toString()}`;
  }

  async function submitRegistration() {
    setError(null);
    const selectionError = validateSelections();
    if (selectionError) {
      setStep(3);
      setError(selectionError);
      return;
    }
    if (needsPartnerBoardConsent && !partnerConsent) {
      setError("Please confirm that organizers may contact you about finding a partner.");
      return;
    }
    if (!termsAccepted) {
      setError("Confirm the tournament policies before submitting.");
      return;
    }
    if (commerce?.available && !commerceQuote) {
      setStep(3);
      setError(
        "Update your total before submitting. Prices or availability may have changed."
      );
      return;
    }
    setPending(true);
    let saved = savedRegistration;
    if (!saved) {
      const response = await submitClubTournamentRegistration(clubSlug, {
        tournament_id: tournamentId,
        registration_slug: registrationSlug || null,
        first_name: contact.firstName,
        last_name: contact.lastName,
        display_name: profile.displayName,
        email: contact.email,
        phone: contact.phone,
        // Public profile suggestions are never identity proof. Staff links the row after review.
        player_id: null,
        dupr_id: profile.duprId,
        doubles_skill: numericValue(profile.doublesSkill),
        singles_skill: numericValue(profile.singlesSkill),
        age: numericValue(contact.age),
        gender: contact.gender,
        notes: contact.notes,
        wants_partner_board_contact:
          needsPartnerBoardConsent && partnerConsent,
        terms_accepted: termsAccepted,
        website: "",
        selections: selectionsPayload(),
        commerce: commerce?.available
          ? {
              item_selections: commerceSelections,
              expected_quote_fingerprint:
                commerceQuote?.quote_fingerprint || "",
              idempotency_key: commerceIdempotencyKey
            }
          : null
      });
      if (response.error || !response.data?.registration_id) {
        setPending(false);
        if (response.status === 409 && response.current_quote) {
          const nextQuote = response.current_quote;
          if (
            nextQuote.quote_fingerprint !==
            (commerceQuote?.quote_fingerprint || null)
          ) {
            setCommerceIdempotencyKey(crypto.randomUUID());
          }
          setCommerceQuote(nextQuote);
          setCommerceSelections(nextQuote.request.item_selections || []);
          setStep(3);
          setError(
            "The total changed. Review the updated price before submitting."
          );
          return;
        }
        if (response.status === 409) {
          setRecoveryEmail(contact.email.trim());
          setMode("edit");
          setError(null);
          return;
        }
        setError(response.error || "We couldn’t submit your registration. Please try again.");
        return;
      }
      if (!response.data.confirmation_token) {
        setPending(false);
        setError(
          "We saved your registration but couldn’t open the confirmation page. Contact the organizer before trying again."
        );
        return;
      }
      saved = {
        registrationId: response.data.registration_id,
        confirmationToken: response.data.confirmation_token,
        emailStatus: response.data.email_delivery?.status
      };
      setSavedRegistration(saved);
      if (selectedTeamEvents.length) {
        // From this point the saved registration and its signed bearer proof
        // are the recovery source. If the tab reloads during team creation,
        // the confirmation page resumes from durable server state.
        window.history.replaceState(null, "", confirmationPath(saved));
      }
    }

    const completedTeamEvents = new Set(createdTeamEventIds);
    for (const teamEvent of selectedTeamEvents) {
      if (completedTeamEvents.has(teamEvent.id)) continue;
      const draft = teamDrafts[teamEvent.id];
      const teamResponse = await createPublicFourPlayerTeam(clubSlug, {
        tournament_id: tournamentId,
        event_option_id: teamEvent.id,
        team_name: draft.teamName.trim(),
        captain_registration_id: saved.registrationId,
        confirmation_token: saved.confirmationToken,
        members: TEAM_SLOTS.map((slot) => {
          if (slot === draft.captainSlot) {
            return {
              slot,
              registration_id: saved.registrationId,
              email: contact.email.trim().toLowerCase(),
              display_name: profile.displayName.trim(),
              gender: contact.gender
            };
          }
          return {
            slot,
            email: draft.teammates[slot].email.trim().toLowerCase(),
            display_name: draft.teammates[slot].displayName.trim(),
            gender: slot.startsWith("MAN_") ? "Men" : "Women"
          };
        }),
        idempotency_key: draft.idempotencyKey,
        website: ""
      });
      if (teamResponse.error) {
        setPending(false);
        setCreatedTeamEventIds([...completedTeamEvents]);
        window.location.href = confirmationPath(saved, true);
        return;
      }
      completedTeamEvents.add(teamEvent.id);
      setCreatedTeamEventIds([...completedTeamEvents]);
    }
    setPending(false);

    window.location.href = confirmationPath(saved);
  }

  if (mode === "choose") {
    return (
      <section style={{ ...cardStyle, display: "grid", gap: "0.9rem" }} data-testid="registration-mode-chooser">
        <h2 style={{ margin: 0 }}>
          {registrationOpen ? "Register or make changes" : "Registration is closed"}
        </h2>
        <p style={{ color: "#475569", margin: 0 }}>
          {registrationOpen
            ? "Start a registration or update one you’ve already submitted."
            : "Already registered? You can still edit your registration."}
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
          {registrationOpen ? (
            <button type="button" onClick={() => resetWizard("new")} style={primaryButtonStyle}>
              Start a registration
            </button>
          ) : null}
          <button type="button" onClick={() => resetWizard("edit")} style={secondaryButtonStyle}>
            Edit my registration
          </button>
        </div>
        {!registrationOpen && registrationClosedReason ? (
          <p style={{ color: "#92400e", margin: 0 }}>{registrationClosedReason}</p>
        ) : null}
      </section>
    );
  }

  if (mode === "edit") {
    return (
      <section style={{ ...cardStyle, display: "grid", gap: "0.9rem" }} data-testid="registration-edit-mode">
        <div>
          <p style={{ color: "#2563eb", fontWeight: 800, margin: "0 0 0.35rem" }}>Edit my registration</p>
          <h2 style={{ margin: 0 }}>Request an edit link</h2>
        </div>
        <p style={{ color: "#475569", margin: 0 }}>
          Enter the email you registered with. We’ll send your edit link there.
        </p>
        <EditLinkRequestForm clubSlug={clubSlug} tournamentId={tournamentId} registrationSlug={registrationSlug} initialEmail={recoveryEmail} />
        {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
        <button type="button" onClick={() => resetWizard("choose")} style={secondaryButtonStyle}>Back to registration choices</button>
      </section>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }} data-testid="registration-new-wizard">
      <div style={{ ...cardStyle, background: "#f8fafc" }}>
        <p style={{ margin: 0, color: "#2563eb", fontWeight: 800 }}>New registration · Step {step} of 4</p>
        <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>Name and contact · Player profile · Events and partners · Review</p>
      </div>

      {step === 1 ? (
        <section style={cardStyle} data-testid="registration-step-contact">
          <h2 style={{ marginTop: 0 }}>1. Name and contact</h2>
          <p style={{ color: "#475569" }}>
            We use your age and gender to show eligible divisions. Only
            organizers can see your contact details.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
            <label>First name *<br /><input aria-label="First name" value={contact.firstName} onChange={(event) => updateContact("firstName", event.target.value)} style={inputStyle} /></label>
            <label>Last name *<br /><input aria-label="Last name" value={contact.lastName} onChange={(event) => updateContact("lastName", event.target.value)} style={inputStyle} /></label>
            <label>Email *<br /><input aria-label="Email" type="email" value={contact.email} onChange={(event) => updateContact("email", event.target.value)} style={inputStyle} /></label>
            <label>Phone / WhatsApp<br /><input aria-label="Phone / WhatsApp" value={contact.phone} onChange={(event) => updateContact("phone", event.target.value)} style={inputStyle} /></label>
            <label>Age *<br /><input aria-label="Age" type="number" min="1" max="120" value={contact.age} onChange={(event) => updateContact("age", event.target.value)} style={inputStyle} /></label>
            <label>Gender *<br />
              <select aria-label="Gender" value={contact.gender} onChange={(event) => updateContact("gender", event.target.value)} style={inputStyle}>
                <option value="">Select</option><option>Women</option><option>Men</option><option>Non-binary</option><option>Other</option><option>Prefer not to say</option>
              </select>
            </label>
          </div>
          <label style={{ display: "block", marginTop: "0.75rem" }}>Notes for tournament staff<br /><textarea aria-label="Notes for tournament staff" rows={3} value={contact.notes} onChange={(event) => updateContact("notes", event.target.value)} style={inputStyle} /></label>
        </section>
      ) : null}

      {step === 2 ? (
        <section style={cardStyle} data-testid="registration-step-profile">
          <h2 style={{ marginTop: 0 }}>2. Player profile</h2>
          <p style={{ color: "#475569" }}>
            {resolution?.profile_candidates.length
              ? "Is this you? Choose a profile below, or continue without one."
              : "We didn’t find a matching player profile. You can still continue."}
          </p>
          {resolution?.profile_candidates.length ? (
            <div style={{ display: "grid", gap: "0.5rem", marginBottom: "1rem" }}>
              {resolution.profile_candidates.map((candidate) => (
                <label key={candidate.id} style={{ border: "1px solid #cbd5e1", borderRadius: "10px", padding: "0.65rem" }}>
                  <input type="radio" name="profile_candidate" checked={profile.candidateId === candidate.id} onChange={() => selectCandidate(candidate)} /> {candidateLabel(candidate)}
                </label>
              ))}
              <label style={{ padding: "0.4rem" }}><input type="radio" name="profile_candidate" checked={!profile.candidateId} onChange={() => selectCandidate(null)} /> None of these is me</label>
            </div>
          ) : null}
          <aside style={{ borderLeft: "4px solid #2563eb", padding: "0.65rem 0.8rem", background: "#eff6ff", marginBottom: "1rem" }}>
            Choosing a profile only fills in this form. An organizer will
            confirm that it belongs to you.
          </aside>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
            <label>Display name *<br /><input aria-label="Display name" value={profile.displayName} onChange={(event) => updateProfile("displayName", event.target.value)} style={inputStyle} /></label>
            <label>DUPR ID<br /><input aria-label="DUPR ID" value={profile.duprId} onChange={(event) => updateProfile("duprId", event.target.value)} style={inputStyle} /></label>
            <label>Doubles skill<br /><input aria-label="Doubles skill" type="number" min="1" max="7" step="0.01" value={profile.doublesSkill} onChange={(event) => updateProfile("doublesSkill", event.target.value)} style={inputStyle} /></label>
            <label>
              Singles skill<br />
              <input aria-label="Singles skill" type="number" min="1" max="7" step="0.01" aria-describedby="singles-skill-help" value={profile.singlesSkill} onChange={(event) => updateProfile("singlesSkill", event.target.value)} style={inputStyle} />
              <span id="singles-skill-help" style={{ display: "block", color: "#64748b", fontSize: "0.9rem", marginTop: "0.35rem" }}>
                No singles rating? Enter your current level or leave it blank.
              </span>
            </label>
          </div>
        </section>
      ) : null}

      {step === 3 ? (
        <section style={cardStyle} data-testid="registration-step-events">
          <h2 style={{ marginTop: 0 }}>3. Events and partners</h2>
          <p style={{ color: "#475569" }}>
            Choose one division for each event on a given day. Picking another
            division will replace your earlier choice.
          </p>
          {groupedEvents.map(({ day, events: dayEvents }) => (
            <div key={day.id} style={{ marginBottom: "1rem" }}>
              <h3>{publicTournamentDayLabel(day.label, day.event_date)}</h3>
              <div style={{ display: "grid", gap: "0.6rem" }}>
                {dayEvents.map((eventOption) => {
                  const selected = selectedIds.includes(eventOption.id);
                  const partner = partnerDetails[eventOption.id] || emptyPartnerState(eventOption);
                  const eligibilityReason = publicEventEligibilityReason(eventOption, eligibilityProfile);
                  return (
                    <article key={eventOption.id} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: selected ? "#f8fafc" : "white" }}>
                      <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
                        <input type="checkbox" aria-label={`${eventOption.event_family_label} ${eventOption.division_name}`} checked={selected} disabled={Boolean(eligibilityReason)} onChange={(event) => toggleEvent(eventOption.id, event.target.checked)} />
                        <span><strong>{publicTournamentEventLabel(eventOption.event_family_label, eventOption.division_name)}</strong><br /><span style={{ color: "#64748b" }}>{scheduledDaysLabel(eventOption, daysById) || "Schedule TBD"}<br />{eventMeta(eventOption)}</span></span>
                      </label>
                      {eligibilityReason ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{eligibilityReason}</p> : null}
                      {selected &&
                      eventOption.partner_required &&
                      String(
                        eventOption.competition_format || ""
                      ).toUpperCase() !== "FOUR_PLAYER_TEAM" ? (
                        <div style={{ display: "grid", gap: "0.65rem", marginTop: "0.75rem" }}>
                          <label>Partner plan<br />
                            <select aria-label={`${eventOption.division_name} partner plan`} value={partner.mode} onChange={(event) => updatePartner(eventOption.id, { mode: event.target.value as PartnerState["mode"] })} style={inputStyle}>
                              <option value="HAS_PARTNER">I have a partner</option>
                              <option value="NEEDS_PARTNER">I need a partner</option>
                            </select>
                          </label>
                          {partner.mode === "HAS_PARTNER" ? (
                            <>
                              <p style={{ color: "#475569", margin: 0 }}>
                                Add your partner’s details here. They don’t need
                                to register separately.
                              </p>
                              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.5rem" }}>
                                <label>Partner name *<br /><input required aria-label={`${eventOption.division_name} partner name`} value={partner.name} onChange={(event) => updatePartner(eventOption.id, { name: event.target.value })} style={inputStyle} /></label>
                                <label>Partner email *<br /><input required aria-label={`${eventOption.division_name} partner email`} type="email" value={partner.email} onChange={(event) => updatePartner(eventOption.id, { email: event.target.value })} style={inputStyle} /></label>
                                <label>Partner age *<br /><input required aria-label={`${eventOption.division_name} partner age`} type="number" min="1" max="120" value={partner.age} onChange={(event) => updatePartner(eventOption.id, { age: event.target.value })} style={inputStyle} /></label>
                                <label>Partner gender *<br /><select required aria-label={`${eventOption.division_name} partner gender`} value={partner.gender} onChange={(event) => updatePartner(eventOption.id, { gender: event.target.value })} style={inputStyle}><option value="">Select</option><option>Women</option><option>Men</option><option>Non-binary</option><option>Other</option><option>Prefer not to say</option></select></label>
                                <label>Partner starting skill *<br /><input required aria-label={`${eventOption.division_name} partner skill`} type="number" min="1" max="7" step="0.01" value={partner.skill} onChange={(event) => updatePartner(eventOption.id, { skill: event.target.value })} style={inputStyle} /></label>
                                <label>Partner phone<br /><input aria-label={`${eventOption.division_name} partner phone`} value={partner.phone} onChange={(event) => updatePartner(eventOption.id, { phone: event.target.value })} style={inputStyle} /></label>
                                <label>Partner DUPR ID<br /><input aria-label={`${eventOption.division_name} partner DUPR ID`} value={partner.duprId} onChange={(event) => updatePartner(eventOption.id, { duprId: event.target.value })} style={inputStyle} /></label>
                              </div>
                            </>
                          ) : (
                            <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                              <input type="checkbox" checked={partner.showOnBoard} disabled={!eventOption.partner_board_enabled} onChange={(event) => updatePartner(eventOption.id, { showOnBoard: event.target.checked })} /> List me as looking for a partner in this division
                            </label>
                          )}
                          <label>Partner note<br /><textarea aria-label={`${eventOption.division_name} partner note`} rows={2} value={partner.note} onChange={(event) => updatePartner(eventOption.id, { note: event.target.value })} style={inputStyle} /></label>
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            </div>
          ))}
          {!selectableEvents.length ? <p>No events are open for registration.</p> : null}
          {selectedTeamEvents.map((teamEvent) => (
            <FourPlayerTeamRegistrationCard
              key={teamEvent.id}
              event={teamEvent}
              captainName={profile.displayName}
              captainEmail={contact.email}
              captainGender={contact.gender}
              value={
                teamDrafts[teamEvent.id] ||
                newTeamRegistrationDraft(contact.gender)
              }
              onChange={(draft) =>
                setTeamDrafts((current) => ({
                  ...current,
                  [teamEvent.id]: draft
                }))
              }
            />
          ))}
          <p><strong>Estimated total:</strong> ${totalPrice.toFixed(2)}</p>
          {commerce?.available ? (
            <TournamentCommerceChooser
              clubSlug={clubSlug}
              tournamentId={tournamentId}
              eventOptionIds={selectedIds}
              catalog={commerce}
              timeZone={timeZone}
              initialSelections={commerceSelections}
              disabled={pending}
              onReviewChange={updateCommerceReview}
            />
          ) : null}
        </section>
      ) : null}

      {step === 4 ? (
        <section style={cardStyle} data-testid="registration-step-review">
          <h2 style={{ marginTop: 0 }}>4. Review and submit</h2>
          <p><strong>Player:</strong> {profile.displayName} · {contact.age} · {contact.gender}</p>
          <p><strong>Email:</strong> {contact.email}</p>
          <ul>
            {selectedIds.map((id) => {
              const event = eventById.get(id)!;
              const partner = partnerDetails[id] || emptyPartnerState(event);
              const teamDraft = teamDrafts[id];
              const entryLabel =
                String(event.competition_format || "").toUpperCase() ===
                "FOUR_PLAYER_TEAM"
                  ? `Team: ${teamDraft?.teamName || "not named"}`
                  : event.partner_required
                ? partner.mode === "HAS_PARTNER" ? `Partner: ${partner.name}` : "Needs partner"
                : isDoublesEvent(event) ? "Registered as an individual" : "Singles";
              return <li key={id}>{scheduledDaysLabel(event, daysById) || "Schedule TBD"} · {publicTournamentEventLabel(event.event_family_label, event.division_name)} · {entryLabel}</li>;
            })}
          </ul>
          <p>
            <strong>Amount due to organizer:</strong>{" "}
            {commerceQuote
              ? `$${(commerceQuote.total_minor / 100).toFixed(2)}`
              : `$${totalPrice.toFixed(2)}`}
          </p>
          {commerceQuote?.lines.some(
            (line) => line.line_type !== "EVENT"
          ) ? (
            <>
              <h3>Extras and bundles</h3>
              <ul>
                {commerceQuote.lines
                  .filter((line) => line.line_type !== "EVENT")
                  .map((line) => (
                    <li key={line.line_key}>
                      {line.quantity} × {line.label}
                      {line.option_label ? ` — ${line.option_label}` : ""} · $
                      {(line.final_total_minor / 100).toFixed(2)}
                    </li>
                  ))}
              </ul>
            </>
          ) : null}
          {commerceQuote ? (
            <p style={{ color: "#475569" }}>
              You’ll pay the tournament organizer separately; we won’t charge
              you here.
            </p>
          ) : null}
          {needsPartnerBoardConsent ? (
            <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start", marginBottom: "0.75rem" }}>
              <input type="checkbox" checked={partnerConsent} onChange={(event) => setPartnerConsent(event.target.checked)} /> The organizers may contact me about finding a partner.
            </label>
          ) : null}
          <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
            <input type="checkbox" checked={termsAccepted} onChange={(event) => setTermsAccepted(event.target.checked)} /> My information is correct, and I agree to the tournament rules and refund policy.
          </label>
        </section>
      ) : null}

      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
        {step > 1 ? <button type="button" disabled={Boolean(savedRegistration)} onClick={() => { setError(null); setStep((current) => current - 1); }} style={secondaryButtonStyle}>Back</button> : <button type="button" onClick={() => resetWizard("choose")} style={secondaryButtonStyle}>Back</button>}
        {step === 1 ? <button type="button" onClick={resolveProfile} disabled={pending} style={primaryButtonStyle}>{pending ? "Checking…" : "Continue"}</button> : null}
        {step === 2 ? <button type="button" onClick={advanceFromProfile} style={primaryButtonStyle}>Continue to events</button> : null}
        {step === 3 ? <button type="button" onClick={advanceFromEvents} style={primaryButtonStyle}>Review registration</button> : null}
        {step === 4 ? <button type="button" onClick={submitRegistration} disabled={pending} style={primaryButtonStyle}>{pending ? "Submitting…" : savedRegistration ? "Finish team registration" : "Submit registration"}</button> : null}
      </div>
    </section>
  );
}

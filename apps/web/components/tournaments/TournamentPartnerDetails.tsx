"use client";

import { useCallback, useEffect, useId, useRef, useState, type CSSProperties } from "react";
import {
  PublicRegistrationPlayer,
  resolveClubTournamentPartnerProfile
} from "@/lib/tournamentRegistrationApi";
import {
  formatRegistrationRating,
  registrationGenderOptions
} from "@/lib/tournamentRegistrationEligibility";

import { automaticRegistrationProfile } from "@/lib/tournamentRegistrationProfile";

export type TournamentPartnerDetailsValue = {
  name: string;
  email: string;
  age: string;
  gender: string;
  skill: string;
  phone: string;
  duprId: string;
  // A browser prefill choice, never submitted as a verified player link.
  profileId?: string;
  profileHasRating?: boolean;
  profileChoiceMade?: boolean;
  profileLookupPending?: boolean;
  profileChoiceRequired?: boolean;
};

type Props = {
  clubSlug: string;
  tournamentId: string;
  registrationSlug?: string | null;
  labelPrefix: string;
  value: TournamentPartnerDetailsValue;
  onChange: (patch: Partial<TournamentPartnerDetailsValue>) => void;
};

const inputStyle: CSSProperties = {
  width: "100%", boxSizing: "border-box", padding: "0.55rem",
  border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit"
};
const gridStyle: CSSProperties = {
  display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 220px), 1fr))", gap: "0.75rem"
};
const buttonStyle: CSSProperties = {
  padding: "0.55rem 0.8rem", border: "1px solid #cbd5e1", borderRadius: "8px",
  background: "white", font: "inherit", fontWeight: 700, cursor: "pointer", justifySelf: "start"
};

export default function TournamentPartnerDetails({
  clubSlug, tournamentId, registrationSlug, labelPrefix, value, onChange
}: Props) {
  const choiceId = useId();
  const canSearch = value.name.trim().length >= 2;
  const skillReadOnly = Boolean(value.profileId && value.profileHasRating);
  const requestId = useRef(0);
  const detailsVersion = useRef(0);
  const lookupTimer = useRef<ReturnType<typeof setTimeout>>();
  const onChangeRef = useRef(onChange);
  useEffect(() => { onChangeRef.current = onChange; }, [onChange]);
  const [candidates, setCandidates] = useState<PublicRegistrationPlayer[] | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => () => { requestId.current += 1; }, []);

  function changeIdentity(patch: Partial<TournamentPartnerDetailsValue>) {
    requestId.current += 1;
    setPending(false);
    setCandidates(null);
    setError(null);
    onChange({
      ...(value.profileId && patch.name !== undefined ? { skill: "", duprId: "" } : {}),
      ...patch, profileId: patch.name !== undefined ? "" : value.profileId,
      ...(patch.name !== undefined ? { profileHasRating: false, profileChoiceMade: false, profileLookupPending: patch.name.trim().length >= 2, profileChoiceRequired: false } : {})
    });
  }

  const findProfile = useCallback(async (retry = false) => {
    if (!canSearch || value.profileId || (value.profileChoiceMade && !retry)) return;
    clearTimeout(lookupTimer.current);
    const id = ++requestId.current;
    const version = detailsVersion.current;
    setPending(true);
    onChangeRef.current({ profileLookupPending: true });
    setError(null);
    try {
      const result = await resolveClubTournamentPartnerProfile(clubSlug, {
        tournament_id: tournamentId,
        registration_slug: registrationSlug || null,
        name: value.name.trim(),
        // An email still being typed must not suppress name suggestions.
        email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.email.trim()) ? value.email.trim() : null
      });
      if (id !== requestId.current) return;
      if (result.error || !result.data) {
        setError(result.error || "We couldn’t look up your partner. Try again, or enter their details below.");
        return;
      }
      const rows = result.data.profile_candidates;
      setCandidates(rows);
      const candidate = automaticRegistrationProfile(rows, value.name, result.data.profile_match_kind);
      onChangeRef.current(candidate && version === detailsVersion.current ? {
        profileId: candidate.id, profileHasRating: candidate.doubles_skill != null, profileChoiceMade: true, profileChoiceRequired: false,
        skill: candidate.doubles_skill == null ? "" : String(candidate.doubles_skill),
        duprId: String(candidate.dupr_id || "")
      } : { profileChoiceMade: rows.length === 0, profileChoiceRequired: rows.length > 0 });
    } catch {
      if (id === requestId.current) {
        setError("We couldn’t look up your partner. Try again, or enter their details below.");
      }
    } finally {
      if (id === requestId.current) {
        setPending(false);
        onChangeRef.current({ profileLookupPending: false });
      }
    }
  }, [clubSlug, tournamentId, registrationSlug, canSearch, value.name, value.email, value.profileId, value.profileChoiceMade]);

  useEffect(() => {
    if (!canSearch || value.profileId || value.profileChoiceMade) return;
    onChangeRef.current({ profileLookupPending: true });
    lookupTimer.current = setTimeout(() => { void findProfile(); }, 250);
    return () => {
      clearTimeout(lookupTimer.current);
      requestId.current += 1;
      onChangeRef.current({ profileLookupPending: false });
    };
  }, [findProfile, canSearch, value.name, value.profileId, value.profileChoiceMade]);

  function selectProfile(candidate: PublicRegistrationPlayer | null) {
    clearTimeout(lookupTimer.current);
    requestId.current += 1;
    setPending(false);
    setError(null);
    onChange(candidate ? {
      profileId: candidate.id, profileHasRating: candidate.doubles_skill != null, profileChoiceMade: true, profileLookupPending: false, profileChoiceRequired: false,
      name: candidate.display_name,
      skill: candidate.doubles_skill == null ? "" : String(candidate.doubles_skill),
      duprId: String(candidate.dupr_id || "")
    } : { profileId: "", profileHasRating: false, ...(value.profileId ? { skill: "", duprId: "" } : {}), profileChoiceMade: true, profileLookupPending: false, profileChoiceRequired: false });
  }

  return (
    <div style={{ display: "grid", gap: "0.75rem" }}>
      <p style={{ color: "#475569", margin: 0 }}>
        Start typing your partner’s first or last name, then choose their profile from the suggestions.
        If they don’t have a profile, enter their full name and details below.
        They don’t need to register separately.
      </p>
      <div style={gridStyle}>
        <div style={{ display: "grid", gap: "0.5rem", alignContent: "start" }}>
        <label>Partner full name *<br />
          <input required aria-label={`${labelPrefix} partner name`} placeholder="Start typing a first or last name"
            autoComplete="off" aria-describedby={`${choiceId}-help`}
            value={value.name} onChange={(event) => changeIdentity({ name: event.target.value })}
            onBlur={() => { if (!candidates && !pending) void findProfile(); }} style={inputStyle} />
        </label>
        <p id={`${choiceId}-help`} style={{ margin: 0, color: "#475569", fontSize: "0.9rem" }}>Type at least 2 characters to see matching profiles.</p>
        {pending ? <p role="status" style={{ margin: 0, color: "#475569" }}>Finding partner profiles…</p> : null}
        {candidates?.length ? (
          <fieldset style={{ margin: 0, padding: "0.75rem", border: "1px solid #cbd5e1", borderRadius: "8px" }}>
            <legend>Is one of these your partner?</legend>
            <div style={{ display: "grid", gap: "0.5rem" }}>
              {candidates.map((candidate) => (
                <label key={candidate.id} style={{ display: "flex", alignItems: "center", gap: "0.5rem", minHeight: "44px", cursor: "pointer" }}>
                  <input type="radio" name={choiceId} required checked={value.profileId === candidate.id}
                    onChange={() => selectProfile(candidate)} />
                  <span><strong>{candidate.display_name}</strong><br />
                    {candidate.doubles_skill == null ? "Doubles rating not set" : `Doubles ${formatRegistrationRating(candidate.doubles_skill)}`}
                  </span>
                </label>
              ))}
              <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", minHeight: "44px", cursor: "pointer" }}><input type="radio" name={choiceId} required checked={Boolean(value.profileChoiceMade && !value.profileId)}
                onChange={() => selectProfile(null)} /> None of these is my partner</label>
            </div>
            {candidates.length >= 8 ? <p style={{ margin: "0.5rem 0 0", color: "#475569" }}>Keep typing to narrow the list.</p> : null}
          </fieldset>
        ) : candidates && !pending ? (
          <p role="status" style={{ margin: 0, color: "#475569" }}>No matching profile found. Try their first or last name, or enter their full name and details below.</p>
        ) : null}
        </div>
        <label>Partner email *<br />
          <input required aria-label={`${labelPrefix} partner email`} type="email" placeholder="name@example.com"
            value={value.email} onChange={(event) => changeIdentity({ email: event.target.value })}
            onBlur={() => { if (!candidates && !pending) void findProfile(); }} style={inputStyle} />
        </label>
      </div>
      {!value.profileId ? (
        <button type="button" onClick={() => { void findProfile(true); }} disabled={pending || !canSearch} style={buttonStyle}>
          {pending ? "Finding partner profile…" : "Find partner profile"}
        </button>
      ) : (
        <p role="status" style={{ margin: 0, color: "#166534" }}>
          Profile selected: <strong>{value.name}</strong>. Check their details below.
          {" "}<button type="button" onClick={() => selectProfile(null)} style={buttonStyle}>Change profile</button>
        </p>
      )}
      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
      <div style={gridStyle}>
        <label>Partner age *<br /><input required aria-label={`${labelPrefix} partner age`} type="number" min="1" max="120" value={value.age} onChange={(event) => onChange({ age: event.target.value })} style={inputStyle} /></label>
        <label>Partner gender *<br /><select required aria-label={`${labelPrefix} partner gender`} value={value.gender} onChange={(event) => onChange({ gender: event.target.value })} style={inputStyle}><option value="">Select</option>{registrationGenderOptions(value.gender).map((gender) => <option key={gender} value={gender}>{gender}</option>)}</select></label>
        {/* Club ratings retain their full precision; a hundredth-step rejects valid profile autofill. */}
        <label>Partner starting skill *<br /><input required aria-label={`${labelPrefix} partner skill`} type="number" min="1" max="7" step="any" inputMode="decimal" value={value.skill} readOnly={skillReadOnly} onChange={(event) => { if (skillReadOnly) return; detailsVersion.current += 1; onChange({ skill: event.target.value }); }} style={{ ...inputStyle, ...(skillReadOnly ? { background: "#f1f5f9" } : {}) }} /></label>
      </div>
      {skillReadOnly ? <p style={{ margin: 0, color: "#475569" }}>Your partner’s club profile rating can’t be changed here.</p> : null}
      <div style={gridStyle}>
        <label>Partner phone<br /><input aria-label={`${labelPrefix} partner phone`} type="tel" value={value.phone} onChange={(event) => onChange({ phone: event.target.value })} style={inputStyle} /></label>
        {!value.profileId ? <label>Partner DUPR ID<br /><input aria-label={`${labelPrefix} partner DUPR ID`} value={value.duprId} onChange={(event) => { detailsVersion.current += 1; onChange({ duprId: event.target.value }); }} style={inputStyle} /></label> : null}
      </div>
    </div>
  );
}

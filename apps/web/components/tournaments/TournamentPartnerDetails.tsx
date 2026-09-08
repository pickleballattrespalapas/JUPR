"use client";

import { useEffect, useId, useRef, useState, type CSSProperties } from "react";
import {
  PublicRegistrationPlayer,
  resolveClubTournamentPartnerProfile
} from "@/lib/tournamentRegistrationApi";
import {
  formatRegistrationRating,
  registrationGenderOptions
} from "@/lib/tournamentRegistrationEligibility";

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
  const requestId = useRef(0);
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
      ...patch, profileId: patch.name !== undefined ? "" : value.profileId
    });
  }

  async function findProfile() {
    if (!value.name.trim() || value.profileId) return;
    if (value.email.trim() && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.email.trim())) {
      setError("Enter a valid partner email address, or leave it blank while finding their profile.");
      return;
    }
    const id = ++requestId.current;
    setPending(true);
    setError(null);
    try {
      const result = await resolveClubTournamentPartnerProfile(clubSlug, {
        tournament_id: tournamentId,
        registration_slug: registrationSlug || null,
        name: value.name.trim(),
        email: value.email.trim() || null
      });
      if (id !== requestId.current) return;
      if (result.error || !result.data) {
        setError(result.error || "We couldn’t look up your partner. Try again, or enter their details below.");
        return;
      }
      setCandidates(result.data.profile_candidates);
    } catch {
      if (id === requestId.current) {
        setError("We couldn’t look up your partner. Try again, or enter their details below.");
      }
    } finally {
      if (id === requestId.current) setPending(false);
    }
  }

  function selectProfile(candidate: PublicRegistrationPlayer | null) {
    requestId.current += 1;
    setPending(false);
    setError(null);
    onChange(candidate ? {
      profileId: candidate.id,
      name: candidate.display_name,
      skill: candidate.doubles_skill == null ? "" : String(candidate.doubles_skill),
      duprId: String(candidate.dupr_id || "")
    } : { profileId: "", skill: "", duprId: "" });
  }

  return (
    <div style={{ display: "grid", gap: "0.75rem" }}>
      <p style={{ color: "#475569", margin: 0 }}>
        Enter your partner’s full name to find their club profile, or fill in their details below.
        They don’t need to register separately.
      </p>
      <div style={gridStyle}>
        <label>Partner full name *<br />
          <input required aria-label={`${labelPrefix} partner name`} placeholder="First and last name"
            value={value.name} onChange={(event) => changeIdentity({ name: event.target.value })}
            onBlur={() => { if (!candidates && !pending) void findProfile(); }} style={inputStyle} />
        </label>
        <label>Partner email *<br />
          <input required aria-label={`${labelPrefix} partner email`} type="email" placeholder="name@example.com"
            value={value.email} onChange={(event) => changeIdentity({ email: event.target.value })}
            onBlur={() => { if (!candidates && !pending) void findProfile(); }} style={inputStyle} />
        </label>
      </div>
      {!value.profileId ? (
        <button type="button" onClick={findProfile} disabled={pending || !value.name.trim()} style={buttonStyle}>
          {pending ? "Finding partner profile…" : "Find partner profile"}
        </button>
      ) : (
        <p role="status" style={{ margin: 0, color: "#166534" }}>
          Profile selected: <strong>{value.name}</strong>. Check their details below.
          {" "}<button type="button" onClick={() => { selectProfile(null); setCandidates(null); }} style={buttonStyle}>Change profile</button>
        </p>
      )}
      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
      {candidates?.length ? (
        <fieldset style={{ margin: 0, padding: "0.75rem", border: "1px solid #cbd5e1", borderRadius: "8px" }}>
          <legend>Is one of these your partner?</legend>
          <div style={{ display: "grid", gap: "0.5rem" }}>
            {candidates.map((candidate) => (
              <label key={candidate.id}>
                <input type="radio" name={choiceId} checked={value.profileId === candidate.id}
                  onChange={() => selectProfile(candidate)} /> {candidate.display_name}
                {" · "}{candidate.doubles_skill == null ? "Doubles rating not set" : `Doubles ${formatRegistrationRating(candidate.doubles_skill)}`}
              </label>
            ))}
            <label><input type="radio" name={choiceId} checked={!value.profileId}
              onChange={() => selectProfile(null)} /> None of these is my partner</label>
          </div>
        </fieldset>
      ) : candidates && !pending ? (
        <p role="status" style={{ margin: 0, color: "#475569" }}>No matching profile found. Check their full name or email, or enter their details below.</p>
      ) : null}
      <div style={gridStyle}>
        <label>Partner age *<br /><input required aria-label={`${labelPrefix} partner age`} type="number" min="1" max="120" value={value.age} onChange={(event) => onChange({ age: event.target.value })} style={inputStyle} /></label>
        <label>Partner gender *<br /><select required aria-label={`${labelPrefix} partner gender`} value={value.gender} onChange={(event) => onChange({ gender: event.target.value })} style={inputStyle}><option value="">Select</option>{registrationGenderOptions(value.gender).map((gender) => <option key={gender} value={gender}>{gender}</option>)}</select></label>
        <label>Partner starting skill *<br /><input required aria-label={`${labelPrefix} partner skill`} type="number" min="1" max="7" step="0.01" value={value.skill} onChange={(event) => onChange({ skill: event.target.value })} style={inputStyle} /></label>
      </div>
      <div style={gridStyle}>
        <label>Partner phone<br /><input aria-label={`${labelPrefix} partner phone`} type="tel" value={value.phone} onChange={(event) => onChange({ phone: event.target.value })} style={inputStyle} /></label>
        {!value.profileId ? <label>Partner DUPR ID<br /><input aria-label={`${labelPrefix} partner DUPR ID`} value={value.duprId} onChange={(event) => onChange({ duprId: event.target.value })} style={inputStyle} /></label> : null}
      </div>
    </div>
  );
}

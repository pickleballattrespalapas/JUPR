"use client";

import { useMemo, useState } from "react";
import type { PublicTeamLeagueDetail } from "@/lib/teamLeagueApi";

type Props = {
  apiBase: string | null;
  clubSlug: string;
  leagueName: string;
  detail: PublicTeamLeagueDetail;
};

const input = {
  width: "100%",
  maxWidth: "100%",
  minWidth: 0,
  boxSizing: "border-box" as const,
  padding: "0.65rem",
  border: "1px solid #cbd5e1",
  borderRadius: "9px",
  font: "inherit"
};

function operationKey(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `teamleague:${crypto.randomUUID()}`;
  }
  return `teamleague:${Date.now()}:${Math.random().toString(16).slice(2)}`;
}

export default function TeamLeagueRegistrationForm({
  apiBase,
  clubSlug,
  leagueName,
  detail
}: Props) {
  const [signupType, setSignupType] = useState<"team" | "solo">("team");
  const [search, setSearch] = useState("");
  const [playerId, setPlayerId] = useState("");
  const [partnerId, setPartnerId] = useState("");
  const [email, setEmail] = useState("");
  const [partnerEmail, setPartnerEmail] = useState("");
  const [teamName, setTeamName] = useState("");
  const [note, setNote] = useState("");
  const [key, setKey] = useState(operationKey);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageTone, setMessageTone] = useState<"success" | "error" | null>(null);
  const players = useMemo(() => {
    const query = search.trim().toLowerCase();
    return detail.registration_players.filter((player) =>
      !query || player.player_name.toLowerCase().includes(query)
    );
  }, [detail.registration_players, search]);

  async function submit() {
    if (!apiBase) {
      setMessage("Registration is unavailable right now.");
      setMessageTone("error");
      return;
    }
    setBusy(true);
    setMessage(null);
    setMessageTone(null);
    try {
      const response = await fetch(
        `${apiBase.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/team-leagues/${encodeURIComponent(leagueName)}/registrations`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            signup_type: signupType,
            player_id: Number(playerId),
            contact_email: email,
            partner_player_id: signupType === "team" ? Number(partnerId) : null,
            partner_email: signupType === "team" ? partnerEmail : "",
            team_name: signupType === "team" ? teamName : "",
            note,
            idempotency_key: key,
            confirmation_text:
              signupType === "team" ? "REGISTER TEAM" : "JOIN PARTNER WAITLIST"
          })
        }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const detailText =
          typeof payload?.detail === "object"
            ? payload.detail.message
            : payload?.detail;
        throw new Error(String(detailText || `Registration failed (${response.status}).`));
      }
      setMessage(String(payload.message || "Registration saved."));
      setMessageTone("success");
      setKey(operationKey());
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to register.");
      setMessageTone("error");
    } finally {
      setBusy(false);
    }
  }

  if (!detail.registration.open) {
    return (
      <section>
        <h2 style={{ marginBottom: "0.4rem" }}>Registration</h2>
        <p style={{ color: "#475569" }}>
          {detail.registration.unavailable_reason || "Registration is closed. The schedule and results remain available below."}
        </p>
      </section>
    );
  }

  return (
    <section style={{ display: "grid", gap: "0.8rem" }}>
      <h2 style={{ marginBottom: 0 }}>Register</h2>
      <p style={{ margin: 0, color: "#475569" }}>
        Payment is handled offline. A team becomes confirmed only after the
        invited partner accepts the private email link.
      </p>
      {detail.league.allow_substitutes ? (
        <p style={{ margin: 0, color: "#475569" }}>
          Approved one-off substitutes are allowed. Contact league staff to
          arrange a substitution for a scheduled match.
        </p>
      ) : null}
      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
        <label>
          <input type="radio" checked={signupType === "team"} onChange={() => setSignupType("team")} /> Register a team
        </label>
        <label>
          <input type="radio" checked={signupType === "solo"} onChange={() => setSignupType("solo")} /> Find me a partner
        </label>
      </div>
      <label style={{ minWidth: 0 }}>
        Search players
        <input style={input} value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Type a player name" />
      </label>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 220px), 1fr))", gap: "0.8rem", minWidth: 0 }}>
        <label style={{ minWidth: 0 }}>
          Your player profile
          <select style={input} value={playerId} onChange={(event) => setPlayerId(event.target.value)}>
            <option value="">Choose a player</option>
            {players.map((player) => <option key={player.player_id} value={player.player_id}>{player.player_name}{player.rating_jupr ? ` · ${player.rating_jupr.toFixed(2)}` : ""}</option>)}
          </select>
        </label>
        {signupType === "team" ? (
          <label style={{ minWidth: 0 }}>
            Partner
            <select style={input} value={partnerId} onChange={(event) => setPartnerId(event.target.value)}>
              <option value="">Choose a partner</option>
              {players.filter((player) => String(player.player_id) !== playerId).map((player) => <option key={player.player_id} value={player.player_id}>{player.player_name}{player.rating_jupr ? ` · ${player.rating_jupr.toFixed(2)}` : ""}</option>)}
            </select>
          </label>
        ) : null}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 220px), 1fr))", gap: "0.8rem", minWidth: 0 }}>
        <label style={{ minWidth: 0 }}>Your email<input style={input} type="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label>
        {signupType === "team" ? <label style={{ minWidth: 0 }}>Partner email<input style={input} type="email" value={partnerEmail} onChange={(event) => setPartnerEmail(event.target.value)} /></label> : null}
      </div>
      {signupType === "team" ? <label style={{ minWidth: 0 }}>Team name<input style={input} value={teamName} onChange={(event) => setTeamName(event.target.value)} maxLength={120} /></label> : null}
      <label style={{ minWidth: 0 }}>Note (optional)<textarea style={{ ...input, minHeight: "80px" }} value={note} onChange={(event) => setNote(event.target.value)} maxLength={500} /></label>
      <button type="button" disabled={busy || !playerId || (signupType === "team" && !partnerId)} onClick={submit} style={{ width: "fit-content", padding: "0.7rem 1rem", border: 0, borderRadius: "999px", background: "#0f172a", color: "white", fontWeight: 800 }}>
        {busy ? "Saving…" : signupType === "team" ? "Register team" : "Join partner waitlist"}
      </button>
      {message ? (
        <p
          role={messageTone === "error" ? "alert" : "status"}
          style={{ color: messageTone === "error" ? "#b91c1c" : "#166534" }}
        >
          {message}
        </p>
      ) : null}
    </section>
  );
}

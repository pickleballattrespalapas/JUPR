"use client";

import { useEffect, useRef, useState } from "react";
import type { ClubEntryOffer, ClubEntryRegistration } from "@/lib/interclubClubEntry";
import { managedRequest, type ManagedClub } from "@/lib/managedInterclub";
import { seasonDateLabel } from "@/lib/interclubSetup";
import styles from "../setup.module.css";

type Entry = {
  club_id: string;
  status: "invited" | "registered";
  revision: number;
  registration: ClubEntryRegistration | null;
};
export type SelectedHostOffer = { clubId: string; offer: ClubEntryOffer };

type Props = {
  root: string;
  accessToken: string;
  clubs: ManagedClub[];
  active: boolean;
  plannerOpen: boolean;
  onPlan: (offer: SelectedHostOffer) => void;
};

export default function ClubRegistrations({ root, accessToken, clubs, active, plannerOpen, onPlan }: Props) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [links, setLinks] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [issuing, setIssuing] = useState("");
  const pending = useRef(false);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    managedRequest<{ entries: Entry[] }>(root + "/club-entries", accessToken, "GET", undefined, controller.signal)
      .then(result => { if (!controller.signal.aborted) setEntries(result.entries); })
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message || "Could not load club registrations."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [root, accessToken, active, refresh]);

  async function createLink(club: ManagedClub) {
    if (pending.current) return;
    pending.current = true;
    setIssuing(club.id);
    setError("");
    setNotice("");
    try {
      const result = await managedRequest<{ token: string }>(
        root + "/club-entries/" + encodeURIComponent(club.id) + "/link", accessToken, "POST"
      );
      const href = new URL("/interclub/club-entry", window.location.origin);
      href.hash = new URLSearchParams({ token: result.token }).toString();
      setLinks(previous => ({ ...previous, [club.id]: href.toString() }));
      setNotice("Private registration link ready for " + club.name + ".");
      setRefresh(value => value + 1);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not confirm the registration link.");
    } finally {
      pending.current = false;
      setIssuing("");
    }
  }

  async function copyLink(club: ManagedClub) {
    try {
      await navigator.clipboard.writeText(links[club.id]);
      setNotice("Registration link copied for " + club.name + ".");
    } catch {
      setNotice("Select the private link and copy it.");
    }
  }

  return <section aria-label="Club registrations">
    <h2>Club registrations</h2>
    <p>Add a club in Teams &amp; Players, then share its private registration link with its captain.
      Captains can enter their contact details, divisions and hosting availability.</p>
    <button type="button" disabled={loading || !!issuing} onClick={() => setRefresh(value => value + 1)}>Refresh registrations</button>
    <p role="status">{loading ? "Loading registrations…" : notice}</p>
    {error && <p role="alert" className={styles.error}>{error}</p>}
    {plannerOpen && <p className={styles.note}>A meet is open in Schedule. Save or cancel that plan before choosing another host offer.</p>}
    {!clubs.length && <p>Add your first club in Teams &amp; Players.</p>}
    <div className={styles.cards}>{clubs.map(club => {
      const entry = entries.find(item => item.club_id === club.id);
      const reg = entry?.registration;
      const captain = reg?.captain || club.captain;
      const contact = reg?.contact || club.contact;
      const link = links[club.id];
      return <article key={club.id} className={styles.card}>
        <h3>{club.name}</h3>
        <p>{entry?.status === "registered" ? "Registered" : entry || link ? "Invited" : loading ? "Checking registration status" : error ? "Registration status unavailable" : "Awaiting invitation"}</p>
        {captain && <p><strong>Captain</strong> {captain}</p>}
        {contact && <p><strong>{reg ? contactLabel(reg.contact_method) : "Contact"}</strong> {contact}</p>}
        {reg && <>
          <p><strong>Divisions</strong> {reg.divisions.join(", ") || "Still checking"}</p>
          <p><strong>Participation</strong> {participationLabel(reg.participation)}</p>
          {reg.travel_notes && <p><strong>Travel availability</strong> {reg.travel_notes}</p>}
          {reg.host_offers.length > 0 && <div>
            <h4>Proposed host dates</h4>
            <ul>{reg.host_offers.map(offer => <li key={offer.date}>
              <strong>{seasonDateLabel(offer.date)}</strong>
              {offer.start_time && " · From " + offer.start_time}
              {offer.end_time && " · Until " + offer.end_time}
              {offer.courts != null && " · " + offer.courts + (offer.courts === 1 ? " court" : " courts")}
              <p><button type="button" disabled={plannerOpen || clubs.length < 2} onClick={() => onPlan({ clubId: club.id, offer })}>
                Plan meet at {club.name} on {seasonDateLabel(offer.date)}
              </button></p>
            </li>)}</ul>
            {clubs.length < 2 && <p>Add a visiting club in Teams &amp; Players to plan a meet.</p>}
          </div>}
        </>}
        <button type="button" disabled={!!issuing || loading || !!error}
          onClick={() => void createLink(club)}>
          {issuing === club.id ? "Preparing link…" : entry || link ? "Replace registration link" : "Create registration link"}
        </button>
        {(entry || link) && <p className={styles.muted}>A replacement link disables the previous one.</p>}
        {link && <>
          <label className={styles.field}>Private registration link for {club.name}
            <input readOnly value={link} onFocus={event => event.target.select()} />
          </label>
          <button type="button" onClick={() => void copyLink(club)}>Copy registration link</button>
        </>}
      </article>;
    })}</div>
  </section>;
}

function contactLabel(value: ClubEntryRegistration["contact_method"]) {
  return { email: "Email", whatsapp: "WhatsApp", phone: "Phone" }[value];
}

function participationLabel(value: ClubEntryRegistration["participation"]) {
  return {
    host_travel: "Hosting and traveling", host: "Hosting only", travel: "Traveling only",
    undecided: "Still checking",
  }[value];
}

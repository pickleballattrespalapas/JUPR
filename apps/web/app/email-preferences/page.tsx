import Link from "next/link";
import { getEmailPreferences } from "@/lib/emailPreferencesApi";
import EmailPreferencesPanel from "./EmailPreferencesPanel";

type EmailPreferencesPageProps = {
  searchParams?: { token?: string; ut?: string; sid?: string; subscription_id?: string };
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function EmailPreferencesPage({ searchParams }: EmailPreferencesPageProps) {
  const token = searchParams?.token || null;
  const ut = searchParams?.ut || null;
  const sid = searchParams?.sid || null;
  const subscriptionId = searchParams?.subscription_id || null;
  const hasLegacyIdOnly = !token && !ut && Boolean(sid || subscriptionId);
  const { data, error } = hasLegacyIdOnly
    ? { data: null, error: "Legacy subscription-id links are no longer accepted. Open the tokenized preference link in a recent player update email." }
    : await getEmailPreferences({ token, ut });

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Email Preferences
      </p>
      <h1 style={{ marginTop: 0 }}>Manage email preferences</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Manage optional JUPR emails from the tokenized preference link included in your player update email. Category scope disables player update digests; global scope also applies to future optional categories.
      </p>

      {error ? (
        <article style={{ ...cardStyle, background: "#fef2f2", borderColor: "#fecaca" }}>
          <h2 style={{ marginTop: 0 }}>Preferences unavailable</h2>
          <p style={{ color: "#991b1b" }}>{error}</p>
          <p style={{ marginBottom: 0 }}><Link href="/support">Contact support</Link></p>
        </article>
      ) : (
        <EmailPreferencesPanel initial={data} token={token} ut={ut} />
      )}
    </section>
  );
}

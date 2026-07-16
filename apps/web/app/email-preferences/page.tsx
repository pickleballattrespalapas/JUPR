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
  const { data, error } = await getEmailPreferences({ token, ut, sid, subscriptionId });

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Email Preferences
      </p>
      <h1 style={{ marginTop: 0 }}>Manage email preferences</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Manage optional JUPR player update emails from the preference link included in your email. Category-level and global optional unsubscribe choices are supported for the current player-update subscription system.
      </p>

      {error ? (
        <article style={{ ...cardStyle, background: "#fef2f2", borderColor: "#fecaca" }}>
          <h2 style={{ marginTop: 0 }}>Preferences unavailable</h2>
          <p style={{ color: "#991b1b" }}>{error}</p>
          <p style={{ marginBottom: 0 }}><Link href="/support">Contact support</Link></p>
        </article>
      ) : (
        <EmailPreferencesPanel initial={data} token={token} ut={ut} sid={sid} subscriptionId={subscriptionId} />
      )}
    </section>
  );
}

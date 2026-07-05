import Link from "next/link";
import { getAdminApiBaseUrl } from "@/lib/adminMatchLogApi";
import AdminPilotPreflightPanel from "./AdminPilotPreflightPanel";

export default function AdminPilotPage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin pilot
      </p>
      <h1 style={{ marginTop: 0 }}>Match Log and Replay pilot checks</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Use this page from a signed-in admin browser session before the first pilot operation. It checks the API flag status and your staff session using safe validation requests.
      </p>
      <AdminPilotPreflightPanel apiBase={getAdminApiBaseUrl()} clubId="tres_palapas" />
      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/match-log">Open Match Log</Link> · <Link href="/admin/replay-history">Open Replay History</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}

import Link from "next/link";
import AdminLoginForm from "./AdminLoginForm";

export default function AdminLoginPage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin login
      </p>
      <h1 style={{ marginTop: 0 }}>JUPR admin login</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Sign in with Supabase Auth to use Next admin pilot workflows. Write features still remain workflow-flagged, club-scoped, FastAPI-mediated, and backed by Streamlit fallback until each pilot is proven.
      </p>
      <AdminLoginForm />
      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin">Back to operations cockpit</Link>
      </p>
    </section>
  );
}

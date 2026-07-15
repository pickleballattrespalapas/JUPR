import Link from "next/link";
import AdminResetPasswordForm from "./AdminResetPasswordForm";

export default function AdminResetPasswordPage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin password reset
      </p>
      <h1 style={{ marginTop: 0 }}>Reset admin password</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Request a Supabase Auth recovery email and set a new password from the recovery session. Admin writes still require FastAPI JWT validation, role checks, and workflow-specific feature flags.
      </p>
      <AdminResetPasswordForm />
      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/login">Back to admin login</Link>
      </p>
    </section>
  );
}

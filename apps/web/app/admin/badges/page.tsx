import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import Link from "next/link";

export default async function AdminBadgesPage() {
  await requireAdminWorkspace();
  return (
    <section>
      <h1>Badges are disabled</h1>
      <p>Badge processing and public badge collections have been retired. Player ratings, match history, and trophy cases remain available.</p>
      <p><Link href="/admin">Return to admin home</Link></p>
    </section>
  );
}

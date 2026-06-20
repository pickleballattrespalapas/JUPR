import Link from "next/link";

export default function HomePage() {
  return (
    <section>
      <h1 style={{ marginBottom: "0.5rem" }}>Welcome to JUPR</h1>
      <p style={{ marginTop: 0 }}>
        Public, read-only player rating views for clubs piloting the JUPR SaaS experience.
      </p>
      <p>
        Start here: <Link href="/clubs/tres-palapas">Tres Palapas club page</Link>
      </p>
      <p>
        Or jump to <Link href="/clubs/tres-palapas/leaderboards">Tres Palapas leaderboards</Link>.
      </p>
    </section>
  );
}

import Link from "next/link";

export default function HomePage() {
  return (
    <section>
      <h1 style={{ marginBottom: "0.5rem" }}>Welcome to JUPR</h1>
      <p style={{ marginTop: 0 }}>This is the first public web shell for JUPR (read-only pages).</p>
      <p>
        Start here: <Link href="/clubs/tres-palapas">Tres Palapas club page</Link>
      </p>
    </section>
  );
}

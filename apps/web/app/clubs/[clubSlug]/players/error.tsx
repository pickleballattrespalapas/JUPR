"use client";

export default function PlayersError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <section data-testid="players-route-error-state" role="alert">
      <h1>Players are temporarily unavailable</h1>
      <p>No private player data was exposed. Try the public directory again.</p>
      <button type="button" onClick={() => reset()}>Try again</button>
    </section>
  );
}

"use client";

export default function PlayersError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <section data-testid="players-route-error-state" role="alert">
      <h1>Player profiles are unavailable</h1>
      <p>We couldn&apos;t load player profiles. Please try again.</p>
      <button type="button" onClick={() => reset()}>Try again</button>
    </section>
  );
}

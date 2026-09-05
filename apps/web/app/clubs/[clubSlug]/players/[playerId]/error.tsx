"use client";

export default function PlayerProfileError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <section data-testid="player-profile-route-error-state" role="alert">
      <h1>Player profile unavailable</h1>
      <p>We couldn&apos;t load this player profile. Please try again.</p>
      <button type="button" onClick={() => reset()}>Try again</button>
    </section>
  );
}

"use client";

export default function PlayerProfileError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <section data-testid="player-profile-route-error-state" role="alert">
      <h1>Player profile unavailable</h1>
      <p>No private player data was exposed. Try loading the public profile again.</p>
      <button type="button" onClick={() => reset()}>Try again</button>
    </section>
  );
}

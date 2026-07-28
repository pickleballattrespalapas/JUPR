import TeamInvitationReview from "./TeamInvitationReview";

type Props = {
  params: { clubSlug: string };
};

export default function TournamentTeamInvitationPage({ params }: Props) {
  return (
    <section>
      <p style={{ color: "#2563eb", fontWeight: 800 }}>Private team invitation</p>
      <h1>Review your four-player team spot</h1>
      <p style={{ color: "#475569", maxWidth: "44rem" }}>
        This private link is cleared from your browser address before the
        invitation is checked.
      </p>
      <TeamInvitationReview clubSlug={params.clubSlug} />
    </section>
  );
}

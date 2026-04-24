from __future__ import annotations

import streamlit as st


def render(ctx) -> None:  # noqa: ARG001
    st.markdown(
        """
        <section class="jupr-public-shell jupr-rules-page">
          <div class="jupr-card jupr-rules-section">
            <div class="jupr-hero-eyebrow">Public Trust &amp; Integrity</div>
            <h1 class="jupr-hero-title">Rating Rules</h1>
            <p class="jupr-hero-subtitle">How JUPR ratings, verified matches, corrections, and badges work at Tres Palapas.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>1) What does JUPR mean?</h2>
            <p>JUPR is the player rating and event tracking system used for Tres Palapas leagues, ladders, verified round robins, tournaments, player profiles, badges, and weekly updates.</p>
            <p>A JUPR rating is meant to reflect performance in recorded events, not reputation, self-rating, or one good day on court.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>2) How ratings move</h2>
            <p>JUPR is based on an Elo-style model. Before a match, the system estimates the expected result using the players’ current ratings. After the match, the actual score is compared against that expectation. A stronger-than-expected performance moves a rating up more, while an underperformance can move it down.</p>
            <p>The final score matters because a close 15–13 match does not tell the same story as a 15–3 match.</p>
            <ul class="jupr-rules-list">
              <li>Beating a stronger team or outperforming expectations can move a rating more.</li>
              <li>A winning team should always receive positive rating movement.</li>
              <li>A losing team may still move slightly up if they significantly outperform expectations against stronger opponents.</li>
              <li>Ratings are displayed on the JUPR scale rather than raw Elo points.</li>
            </ul>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>3) What counts as rated?</h2>
            <p>Only results entered through approved rated event workflows affect official JUPR ratings.</p>
            <ul class="jupr-rules-list">
              <li>Verified round robins</li>
              <li>Official leagues/ladders marked as rated</li>
              <li>Admin-entered tournament or event results marked as rated</li>
              <li>Other club-approved rated events</li>
            </ul>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>4) What does not count?</h2>
            <p>Not every game at Tres Palapas affects your rating. Casual play can still be tracked or displayed, but it is separated from official rated history unless the event is explicitly marked as rated.</p>
            <ul class="jupr-rules-list">
              <li>Open play</li>
              <li>Practice games</li>
              <li>Warmups</li>
              <li>Casual/social events marked unrated</li>
              <li>Incomplete or test events</li>
              <li>Events not approved as rated</li>
            </ul>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>5) Verified vs. casual</h2>
            <p>Verified/rated results are official match records that can impact ratings and standings.</p>
            <p>Casual/social results can still be useful for community history and participation tracking, but they do not create official rating movement unless an event is explicitly approved as rated.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>6) Can matches be edited?</h2>
            <p>Scores can be corrected by authorized admins when there is a clear data-entry mistake or event correction. Corrections should preserve the integrity of the event record and should not be used to manually manipulate ratings.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>7) What happens if scores are deleted?</h2>
            <p>Rated match deletion is an admin-controlled correction workflow. Deleted or corrected scores should be reflected in the rating record through the rating recalculation process.</p>
            <div class="jupr-rules-callout">The intended policy is straightforward: if a rated match is removed, current ratings should reflect the remaining rated match history.</div>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>8) How badges are awarded</h2>
            <p>Badges are awarded from recorded player and match data. Some badges recognize participation, some recognize performance, some recognize league or tournament results, and some recognize unusual achievements like upsets or strong runs.</p>
            <p>Badge requirements may be simplified in public view, but awards should come from recorded results rather than manual favoritism.</p>
            <a class="jupr-link-button" href="?page=badge_codex">See Badge Codex</a>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>9) Why ratings may not match self-rating</h2>
            <p>JUPR is based on recorded performance inside this system. A player’s JUPR rating may differ from DUPR, club self-rating, tournament bracket level, or personal opinion because each system uses different data.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>10) Admin integrity</h2>
            <p>Rated results are entered and corrected by authorized admins. Public users can view results and profiles, but cannot change rated history.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # TODO: Keep this page copy aligned with any future changes to match deletion/replay workflows.

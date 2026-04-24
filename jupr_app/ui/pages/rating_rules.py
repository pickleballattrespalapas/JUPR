from __future__ import annotations

import streamlit as st


def render(ctx) -> None:  # noqa: ARG001
    st.markdown(
        """
        <section class="jupr-public-shell jupr-rules-page">
          <div class="jupr-card jupr-rules-section">
            <div class="jupr-hero-eyebrow">Rating System</div>
            <h1 class="jupr-hero-title">Rating Rules</h1>
            <p class="jupr-hero-subtitle">How JUPR ratings, verified matches, corrections, and badges work at Tres Palapas.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>1) What JUPR tracks</h2>
            <p>JUPR is the player rating and event tracking system for Tres Palapas leagues, ladders, verified round robins, tournaments, player profiles, badges, and weekly updates.</p>
            <p>JUPR ratings reflect recorded performance inside the Tres Palapas system. They are based on entered match results, not reputation, self-rating, or one strong day on court.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>2) How ratings move</h2>
            <p>JUPR uses an Elo-style rating model. Before a rated match, the system estimates the expected result from the players’ current ratings. After the match, the final score is compared against that expectation.</p>
            <p>Ratings move more when a team performs above expectation. Ratings move down when a team performs below expectation. The scoreline matters because a 15–13 match and a 15–3 match tell different stories.</p>
            <ul class="jupr-rules-list">
              <li>Winners receive positive rating movement.</li>
              <li>Underdogs gain more when they beat or strongly outperform higher-rated opponents.</li>
              <li>A losing team can still gain rating when it significantly outperforms expectation against stronger opponents.</li>
              <li>Ratings display on the JUPR scale rather than raw Elo points.</li>
            </ul>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>3) What counts as rated</h2>
            <p>Only approved rated event results affect official JUPR ratings.</p>
            <ul class="jupr-rules-list">
              <li>Verified round robins</li>
              <li>Official leagues and ladders marked as rated</li>
              <li>Tournament results marked as rated</li>
              <li>Club-approved rated events entered by an authorized admin</li>
            </ul>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>4) What does not count</h2>
            <p>Not every game at Tres Palapas affects your rating. Casual play, practice games, warmups, and unrated social events stay outside official rating history.</p>
            <ul class="jupr-rules-list">
              <li>Open play</li>
              <li>Practice games</li>
              <li>Warmups</li>
              <li>Casual or social events marked unrated</li>
              <li>Incomplete or test events</li>
              <li>Events not approved as rated</li>
            </ul>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>5) Verified vs. casual</h2>
            <p>Verified/rated results are official match records. They affect JUPR ratings, standings, player profiles, and event history.</p>
            <p>Casual/social results are separated from official rating history. They can appear as participation or event history, but they do not move official ratings unless the event is approved as rated.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>6) Match corrections</h2>
            <p>Authorized admins correct scores when there is a data-entry mistake or official event correction. Corrections update the match record and rating history according to the corrected result.</p>
            <p>Public users can view results and profiles. Public users cannot change rated history.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>7) Deleted scores</h2>
            <p>Deleted rated matches are handled through the admin correction workflow. Rating-impact deletions require recalculation before the public rating record is final.</p>
            <div class="jupr-rules-callout">Do not present deleted-match rating rollback as automatic unless the code enforces it.</div>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>8) How badges are awarded</h2>
            <p>Badges are awarded from recorded player, match, league, tournament, and event data. Participation badges come from attendance or match history. Performance badges come from results. League and tournament badges come from completed event outcomes.</p>
            <p>Public badge descriptions summarize the achievement. The actual award comes from recorded JUPR data.</p>
            <a class="jupr-link-button" href="?page=badge_codex">See Badge Codex</a>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>9) Why JUPR may differ from other ratings</h2>
            <p>JUPR uses the match data recorded inside this system. It can differ from DUPR, self-rating, tournament bracket level, or personal opinion because each rating system uses different data and rules.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <h2>10) Admin integrity</h2>
            <p>Rated results are entered, corrected, and managed by authorized admins. Public users can view ratings, match history, profiles, badges, and event results, but they cannot edit official rated history.</p>
          </div>

          <div class="jupr-rules-section jupr-card">
            <div class="jupr-rules-callout">JUPR ratings follow recorded results. Verified matches count. Casual play stays separate. Corrections update the official record.</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

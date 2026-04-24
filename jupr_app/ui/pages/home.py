from __future__ import annotations

import streamlit as st


def render(ctx) -> None:  # noqa: ARG001
    st.markdown(
        """
        <section class="jupr-public-shell">
          <div class="jupr-hero jupr-card">
            <div class="jupr-hero-eyebrow">JUPR</div>
            <h1 class="jupr-hero-title">The official player rating and event system for Tres Palapas.</h1>
            <p class="jupr-hero-subtitle">
              Track ratings, standings, player profiles, events, match history, badges, and weekly updates
              from Tres Palapas pickleball.
            </p>
            <div class="jupr-trust-badge" role="note">
              Transparent rating rules • Verified event separation • Admin-controlled corrections
            </div>
            <div class="jupr-hero-actions">
              <a class="jupr-link-button" href="?page=leaderboards">View Leaderboards</a>
              <a class="jupr-link-button" href="?page=players">Find Your Player Profile</a>
              <a class="jupr-link-button" href="?page=tournament_registration">See Upcoming Events</a>
              <a class="jupr-link-button" href="?page=verified_updates_request">Subscribe to Player Updates</a>
              <a class="jupr-link-button" href="?admin=1&page=admin_login">Admin Login</a>
            </div>
          </div>

          <div class="jupr-trust-section jupr-card">
            <h2 class="jupr-section-title">Built on transparent rating rules</h2>
            <p class="jupr-home-card-body">
              JUPR separates verified results from casual play, uses match scores and opponent strength to update ratings,
              and keeps rating changes tied to recorded match history.
            </p>
            <div class="jupr-trust-grid">
              <div class="jupr-trust-card">
                <h3>Verified vs. Casual</h3>
                <p>Rated events affect standings. Casual/social results can be shown separately without changing official ratings.</p>
              </div>
              <div class="jupr-trust-card">
                <h3>Score-based movement</h3>
                <p>Ratings consider who you played, the expected result, and how the match actually finished.</p>
              </div>
              <div class="jupr-trust-card">
                <h3>Corrections matter</h3>
                <p>If a rated score is corrected or removed, the rating record should reflect the corrected match history.</p>
              </div>
            </div>
            <div class="jupr-trust-actions">
              <a class="jupr-link-button" href="?page=rating_rules">Read the Rating Rules</a>
            </div>
          </div>

          <div class="jupr-home-grid">
            <a class="jupr-home-card jupr-card jupr-card--hover jupr-home-card-link" href="?page=leaderboards" aria-label="View leaderboards">
              <h3 class="jupr-home-card-title">Ratings &amp; Standings</h3>
              <p class="jupr-home-card-body">Live player ratings, league results, match history, and verified event tracking.</p>
              <div class="jupr-home-card-cta">View leaderboards →</div>
            </a>
            <a class="jupr-home-card jupr-card jupr-card--hover jupr-home-card-link" href="?page=players" aria-label="Find player profiles">
              <h3 class="jupr-home-card-title">Player Profiles</h3>
              <p class="jupr-home-card-body">Search players, view rating history, badges, trophies, and recent match results.</p>
              <div class="jupr-home-card-cta">Find players →</div>
            </a>
            <a class="jupr-home-card jupr-card jupr-card--hover jupr-home-card-link" href="?page=tournament_registration" aria-label="View event registration and updates">
              <h3 class="jupr-home-card-title">Events &amp; Updates</h3>
              <p class="jupr-home-card-body">View public event registration, weekly recaps, partner boards, and player update subscriptions.</p>
              <div class="jupr-home-card-cta">See events →</div>
            </a>
            <a class="jupr-home-card jupr-card jupr-card--hover jupr-home-card-link" href="?page=rating_rules" aria-label="Read the rating rules">
              <h3 class="jupr-home-card-title">Rating Rules</h3>
              <p class="jupr-home-card-body">See what counts as rated, how results are handled, and why verified matches matter.</p>
              <div class="jupr-home-card-cta">Read rating rules →</div>
            </a>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

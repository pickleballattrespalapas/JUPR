from __future__ import annotations

import streamlit as st

from jupr_app.ui.branding import CLUB_NAME, PRODUCT_NAME, TAGLINE


def render(ctx) -> None:  # noqa: ARG001
    st.markdown(
        f"""
        <section class="jupr-public-shell">
          <div class="jupr-hero jupr-card">
            <div class="jupr-hero-eyebrow">{PRODUCT_NAME} at {CLUB_NAME}</div>
            <h1 class="jupr-hero-title">Official player ratings and events for {CLUB_NAME}.</h1>
            <p class="jupr-hero-subtitle">
              Find current ratings, player profiles, tournament registration, and player update subscriptions.
            </p>
            <p class="jupr-hero-subtitle">{TAGLINE}</p>
            <div class="jupr-hero-actions">
              <a class="jupr-link-button" href="?page=leaderboards">Leaderboards</a>
              <a class="jupr-link-button" href="?page=players">Player Profiles</a>
              <a class="jupr-link-button" href="?page=tournament_registration">Tournaments</a>
              <a class="jupr-link-button" href="?page=verified_updates_request">Player Updates</a>
            </div>
          </div>

          <div class="jupr-trust-section jupr-card">
            <h2 class="jupr-section-title">How JUPR ratings work</h2>
            <p class="jupr-home-card-body">
              Ratings are built from approved match results. Casual play stays separate. When an official record is corrected, affected ratings are recalculated.
            </p>
            <div class="jupr-trust-grid">
              <div class="jupr-trust-card">
                <h3>Rated results only</h3>
                <p>Only approved rated events affect JUPR ratings and standings.</p>
              </div>
              <div class="jupr-trust-card">
                <h3>Scorelines matter</h3>
                <p>Movement reflects the final score, margin, opponent strength, and expected result.</p>
              </div>
              <div class="jupr-trust-card">
                <h3>Corrections rebuild ratings</h3>
                <p>Authorized edits update match history and recalculate affected ratings.</p>
              </div>
            </div>
            <div class="jupr-trust-actions">
              <a class="jupr-link-button" href="?page=rating_rules">Read Rating Rules</a>
            </div>
          </div>

          <div class="jupr-home-grid">
            <a class="jupr-home-card jupr-card jupr-card--hover jupr-home-card-link" href="?page=leaderboards" aria-label="View leaderboards">
              <h3 class="jupr-home-card-title">Ratings &amp; Standings</h3>
              <p class="jupr-home-card-body">Current ratings, league standings, and match history.</p>
              <div class="jupr-home-card-cta">View leaderboards →</div>
            </a>
            <a class="jupr-home-card jupr-card jupr-card--hover jupr-home-card-link" href="?page=players" aria-label="Find player profiles">
              <h3 class="jupr-home-card-title">Player Profiles</h3>
              <p class="jupr-home-card-body">Profiles, recent results, badges, trophies, and update subscriptions.</p>
              <div class="jupr-home-card-cta">Find players →</div>
            </a>
            <a class="jupr-home-card jupr-card jupr-card--hover jupr-home-card-link" href="?page=tournament_registration" aria-label="View tournament registration and updates">
              <h3 class="jupr-home-card-title">Tournaments &amp; Updates</h3>
              <p class="jupr-home-card-body">Tournament registration, weekly recaps, partner boards, and player notifications.</p>
              <div class="jupr-home-card-cta">See tournaments →</div>
            </a>
            <a class="jupr-home-card jupr-card jupr-card--hover jupr-home-card-link" href="?page=rating_rules" aria-label="Read the rating rules">
              <h3 class="jupr-home-card-title">Rating Rules</h3>
              <p class="jupr-home-card-body">What counts as rated and how results affect movement.</p>
              <div class="jupr-home-card-cta">Read rating rules →</div>
            </a>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

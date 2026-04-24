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
            <div class="jupr-hero-actions">
              <a class="jupr-link-button" href="?page=leaderboards">View Leaderboards</a>
              <a class="jupr-link-button" href="?page=players">Find Your Player Profile</a>
              <a class="jupr-link-button" href="?page=tournament_registration">See Upcoming Events</a>
              <a class="jupr-link-button" href="?page=verified_updates_request">Subscribe to Player Updates</a>
              <a class="jupr-link-button" href="?admin=1&page=admin_login">Admin Login</a>
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
            <a class="jupr-home-card jupr-card jupr-card--hover jupr-home-card-link" href="?page=faqs" aria-label="Read frequently asked questions">
              <h3 class="jupr-home-card-title">Built for Tres Palapas</h3>
              <p class="jupr-home-card-body">A public-facing rating and event hub for the Tres Palapas pickleball community.</p>
              <div class="jupr-home-card-cta">Read FAQs →</div>
            </a>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

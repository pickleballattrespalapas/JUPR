(function () {
  "use strict";

  var Engine = window.LiveEngine;
  var STORAGE_KEY = "live-brackets-mvp-state";
  var app = document.getElementById("app");
  var importInput = document.getElementById("import-file");
  var flashTimer = null;
  var pendingFocusIndex = null;
  var renderScoreIndex = 0;

  var state = {
    event: null,
    setup: {
      type: "round-robin",
      rrMode: "switch-doubles",
      name: "Saturday Event",
      participantCount: 8,
      totalRounds: 3,
      courtSizesText: "",
      namesText: ""
    },
    flash: null
  };

  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function loadState() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) { return; }
      var parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        state = Object.assign({}, state, parsed);
        if (state.event && state.event.type === "tournament") {
          Engine.resolveTournament(state.event);
        }
      }
    } catch (_error) {
      // ignore corrupted local state
    }
  }

  function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      event: state.event,
      setup: state.setup
    }, null, 2));
  }

  function flash(message, tone) {
    state.flash = { message: message, tone: tone || "success" };
    render();
    if (flashTimer) {
      window.clearTimeout(flashTimer);
    }
    flashTimer = window.setTimeout(function () {
      state.flash = null;
      render();
    }, 2400);
  }

  function participantLookup() {
    return state.event ? Engine.participantsById(state.event.participants) : {};
  }

  function participantName(participantId) {
    var byId = participantLookup();
    return byId[participantId] ? byId[participantId].name : "TBD";
  }

  function teamLabel(teamIds) {
    return (teamIds || []).map(participantName).join(" / ");
  }

  function namesFromText() {
    return String(state.setup.namesText || "")
      .split(/\n|,/)
      .map(function (value) { return Engine.normalizeName(value); })
      .filter(Boolean);
  }

  function setupHelpText() {
    if (state.setup.type === "round-robin" && state.setup.rrMode === "switch-doubles") {
      return "Switch Partner Doubles currently supports 4, 5, 8, and 12 players using your uploaded chart logic.";
    }
    if (state.setup.type === "league") {
      return "League / ladder uses 4-player and 5-player courts, with one-up / one-down movement suggestions between courts.";
    }
    if (state.setup.type === "tournament") {
      return "Tournament mode is team-based in this MVP. Enter one team per line, then score the bracket live.";
    }
    return "Standard round robin supports any participant count.";
  }

  function setSetupValue(key, value, shouldRender) {
    state.setup[key] = value;

    if (key === "type") {
      if (value === "league" && !state.setup.courtSizesText) {
        var suggestion = Engine.suggestCourtSizes(Number(state.setup.participantCount || 0));
        state.setup.courtSizesText = suggestion ? suggestion.join(",") : "";
      }
      if (value === "round-robin" && state.setup.rrMode === "switch-doubles" &&
          !Engine.templateSupportedCount(Number(state.setup.participantCount || 0))) {
        state.setup.participantCount = 8;
      }
    }

    if (key === "participantCount" && state.setup.type === "league") {
      var sizes = Engine.suggestCourtSizes(Number(value || 0));
      state.setup.courtSizesText = sizes ? sizes.join(",") : state.setup.courtSizesText;
    }

    saveState();
    if (shouldRender) {
      render();
    }
  }

  function createEvent() {
    var count = Number(state.setup.participantCount || 0);
    var names = namesFromText();
    var name = state.setup.name || "";

    try {
      if (state.setup.type === "tournament" && names.length !== count) {
        throw new Error("Tournament mode requires one team name per line for all " + count + " teams.");
      }

      if (state.setup.type === "round-robin") {
        state.event = Engine.createRoundRobinEvent({
          name: name,
          mode: state.setup.rrMode,
          participantCount: count,
          names: names
        });
      } else if (state.setup.type === "league") {
        var sizes = Engine.parseCourtSizes(state.setup.courtSizesText);
        state.event = Engine.createLeagueEvent({
          name: name,
          participantCount: count,
          names: names,
          totalRounds: Number(state.setup.totalRounds || 3),
          courtSizes: sizes
        });
      } else {
        state.event = Engine.createTournamentEvent({
          name: name,
          participantCount: count,
          names: names
        });
      }
      saveState();
      flash("Event created.", "success");
    } catch (error) {
      flash(error.message || "Could not create event.", "danger");
    }
  }

  function resetEvent() {
    state.event = null;
    saveState();
    render();
  }

  function exportEvent() {
    if (!state.event) {
      flash("Create an event first.", "warning");
      return;
    }
    var blob = new Blob([Engine.serializeEvent(state.event)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = (state.event.name || "event").toLowerCase().replace(/[^a-z0-9]+/g, "-") + ".json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function importEvent(file) {
    if (!file) { return; }
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var parsed = JSON.parse(String(reader.result || "{}"));
        if (parsed && parsed.type === "tournament") {
          Engine.resolveTournament(parsed);
        }
        state.event = parsed;
        saveState();
        flash("Event imported.", "success");
      } catch (_error) {
        flash("Invalid JSON file.", "danger");
      }
    };
    reader.readAsText(file);
  }

  function printEvent() {
    if (!state.event) {
      flash("Create an event first.", "warning");
      return;
    }
    window.print();
  }

  function metric(label, value) {
    return '<div class="metric"><div class="label">' + escapeHtml(label) +
      '</div><div class="value">' + escapeHtml(value) + '</div></div>';
  }

  function renderFlash() {
    if (!state.flash) { return ""; }
    return '<div class="alert ' + escapeHtml(state.flash.tone) + '">' +
      escapeHtml(state.flash.message) + '</div>';
  }

  function formatType(value) {
    if (value === "round-robin") { return "Round Robin"; }
    if (value === "league") { return "League / Ladder"; }
    if (value === "tournament") { return "Tournament"; }
    return value;
  }

  function standingsTable(rows) {
    return (
      '<div class="table-wrap"><table>' +
        '<thead><tr><th>Rank</th><th>Player</th><th>W</th><th>L</th><th>T</th><th>GP</th><th>PF</th><th>PA</th><th>Diff</th></tr></thead>' +
        '<tbody>' +
          rows.map(function (row) {
            return (
              '<tr>' +
                '<td>' + row.rank + '</td>' +
                '<td>' + escapeHtml(row.name) + '</td>' +
                '<td>' + row.wins + '</td>' +
                '<td>' + row.losses + '</td>' +
                '<td>' + row.ties + '</td>' +
                '<td>' + row.matches + '</td>' +
                '<td>' + row.pointsFor + '</td>' +
                '<td>' + row.pointsAgainst + '</td>' +
                '<td>' + row.differential + '</td>' +
              '</tr>'
            );
          }).join('') +
        '</tbody>' +
      '</table></div>'
    );
  }


  function attrsHtml(attrs) {
    return Object.keys(attrs || {}).map(function (key) {
      return ' data-' + escapeHtml(key) + '="' + escapeHtml(attrs[key]) + '"';
    }).join('');
  }

  function nextScoreIndex() {
    renderScoreIndex += 1;
    return renderScoreIndex;
  }

  function buildScoreInput(config) {
    var inputValue = config.value === null || config.value === undefined ? '' : escapeHtml(config.value);
    return (
      '<label class="score-field">' +
        '<span>' + escapeHtml(config.label) + '</span>' +
        '<input class="score-input" type="number" min="0" inputmode="numeric" data-score-index="' + nextScoreIndex() + '" data-action="' + escapeHtml(config.action) + '"' +
          attrsHtml(config.attrs) +
          ' data-score-key="' + escapeHtml(config.scoreKey) + '" aria-label="' + escapeHtml(config.ariaLabel || config.label) + '" value="' + inputValue + '">' +
      '</label>'
    );
  }

  function scoreCardHtml(config) {
    return (
      '<div class="match-card score-card">' +
        '<div class="score-layout">' +
          '<div class="score-side">' +
            '<div class="side-tag">' + escapeHtml(config.sideALabel || "Team A") + '</div>' +
            '<div class="team-label">' + escapeHtml(config.teamA) + '</div>' +
            buildScoreInput({
              action: config.action,
              attrs: config.attrs,
              scoreKey: "scoreA",
              label: "Score",
              ariaLabel: (config.sideALabel || "Team A") + " score for " + config.teamA,
              value: config.scoreA
            }) +
          '</div>' +
          '<div class="score-vs">vs</div>' +
          '<div class="score-side">' +
            '<div class="side-tag">' + escapeHtml(config.sideBLabel || "Team B") + '</div>' +
            '<div class="team-label">' + escapeHtml(config.teamB) + '</div>' +
            buildScoreInput({
              action: config.action,
              attrs: config.attrs,
              scoreKey: "scoreB",
              label: "Score",
              ariaLabel: (config.sideBLabel || "Team B") + " score for " + config.teamB,
              value: config.scoreB
            }) +
          '</div>' +
        '</div>' +
      '</div>'
    );
  }

  function applyPendingFocus() {
    if (!pendingFocusIndex || pendingFocusIndex < 1) {
      pendingFocusIndex = null;
      return;
    }

    window.requestAnimationFrame(function () {
      var nextInput = app.querySelector('[data-score-index="' + pendingFocusIndex + '"]');
      if (nextInput) {
        nextInput.focus();
        if (typeof nextInput.select === "function") {
          nextInput.select();
        }
      }
      pendingFocusIndex = null;
    });
  }

  function renderSetup() {
    var suggestedSizes = state.setup.type === "league"
      ? (Engine.suggestCourtSizes(Number(state.setup.participantCount || 0)) || [])
      : [];

    var modes = state.setup.type === "round-robin"
      ? (
          '<div class="field span-6"><label>Round Robin mode</label><div class="option-pills">' +
            '<button type="button" class="pill ' + (state.setup.rrMode === "switch-doubles" ? 'active' : '') + '" data-action="set-rr-mode" data-value="switch-doubles">Switch Partner Doubles</button>' +
            '<button type="button" class="pill ' + (state.setup.rrMode === "standard" ? 'active' : '') + '" data-action="set-rr-mode" data-value="standard">Standard Round Robin</button>' +
          '</div></div>'
        )
      : '';

    var leagueFields = state.setup.type === "league"
      ? (
          '<div class="field span-3"><label>Total rounds</label><input type="number" min="1" max="12" step="1" data-field="totalRounds" value="' + escapeHtml(state.setup.totalRounds) + '"></div>' +
          '<div class="field span-9"><label>Court layout</label><input type="text" data-field="courtSizesText" value="' + escapeHtml(state.setup.courtSizesText) + '" placeholder="4,4,5"><div class="note">Suggested layout: <span class="kbd">' + escapeHtml(suggestedSizes.join(",") || "none") + '</span></div></div>'
        )
      : '';

    var countLabel = state.setup.type === "tournament" ? "Team count" : "Player / team count";
    var namesLabel = state.setup.type === "tournament" ? "Teams" : "Names";
    var namesPlaceholder = state.setup.type === "tournament"
      ? "One team per line&#10;Mike / Tim&#10;Jitka / Rob&#10;Woody / Priscilla"
      : "One per line&#10;Amy&#10;Brooke&#10;Chris&#10;Dana";
    var leadText = state.setup.type === "tournament"
      ? "Choose the event type, enter the number of teams, list each team, then score matches live."
      : "Choose the event type, select player count, enter names, then score matches live.";
    var tournamentNote = state.setup.type === "tournament"
      ? '<div class="field span-12"><div class="note">Tournament entries are teams in this flow. Put one team on each line, such as <span class="kbd">Mike / Tim</span>.</div></div>'
      : '';

    return (
      '<section class="grid">' +
        renderFlash() +
        '<div class="hero">' +
          '<div class="card hero-copy">' +
            '<h2>Run the live version of a printable brackets site.</h2>' +
            '<p>This MVP stays entirely in the browser: no database, no ratings, no accounts. Create a round robin, run a live ladder, or operate a bracket and export the whole event as JSON.</p>' +
          '</div>' +
          '<div class="card hero-points">' +
            '<div class="point"><strong>Round Robin</strong><br><span class="subtle">Standard singles or switch-partner doubles.</span></div>' +
            '<div class="point"><strong>League / Ladder</strong><br><span class="subtle">Live standings plus editable up/down movement.</span></div>' +
            '<div class="point"><strong>Tournament</strong><br><span class="subtle">Team-based single-elimination bracket with automatic advancement.</span></div>' +
          '</div>' +
        '</div>' +
        '<div class="card setup-panel">' +
          '<div class="section-head"><div><h2>New event</h2><p>' + escapeHtml(leadText) + '</p></div></div>' +
          '<div class="form-grid">' +
            '<div class="field span-6"><label>Event type</label><div class="option-pills">' +
              '<button type="button" class="pill ' + (state.setup.type === "round-robin" ? 'active' : '') + '" data-action="set-type" data-value="round-robin">Round Robin</button>' +
              '<button type="button" class="pill ' + (state.setup.type === "league" ? 'active' : '') + '" data-action="set-type" data-value="league">League / Ladder</button>' +
              '<button type="button" class="pill ' + (state.setup.type === "tournament" ? 'active' : '') + '" data-action="set-type" data-value="tournament">Tournament</button>' +
            '</div></div>' +
            '<div class="field span-6"><label>Event name</label><input type="text" data-field="name" value="' + escapeHtml(state.setup.name) + '" placeholder="Saturday Social"></div>' +
            modes +
            '<div class="field span-3"><label>' + escapeHtml(countLabel) + '</label><input type="number" min="2" max="64" step="1" data-field="participantCount" value="' + escapeHtml(state.setup.participantCount) + '"></div>' +
            leagueFields +
            '<div class="field span-12"><label>' + escapeHtml(namesLabel) + '</label><textarea data-field="namesText" placeholder="' + namesPlaceholder + '">' + escapeHtml(state.setup.namesText) + '</textarea></div>' +
            tournamentNote +
            '<div class="field span-12"><div class="note">' + escapeHtml(setupHelpText()) + '</div></div>' +
            '<div class="field span-12"><div class="inline-row"><button type="button" data-action="create-event">Create event</button></div></div>' +
          '</div>' +
        '</div>' +
      '</section>'
    );
  }

  function roundRobinStandings() {
    return Engine.computeParticipantStandingsFromMatches(
      Engine.allRoundRobinMatches(state.event.rounds),
      state.event.participants
    );
  }

  function roundRobinMetrics() {
    var matches = Engine.allRoundRobinMatches(state.event.rounds);
    var completed = matches.filter(function (match) {
      return match.scoreA !== null && match.scoreB !== null;
    }).length;
    var standings = roundRobinStandings();
    return [
      { label: "Format", value: state.event.mode === "switch-doubles" ? "Switch Doubles" : "Standard" },
      { label: "Participants", value: state.event.participants.length },
      { label: "Rounds", value: state.event.rounds.length },
      { label: "Scored Matches", value: completed + " / " + matches.length },
      { label: "Leader", value: standings.length ? standings[0].name : "—" }
    ];
  }

  function roundRobinScoreBlocks() {
    return state.event.rounds.map(function (round) {
      var byeBlock = round.byeParticipantId
        ? '<div class="badge">Bye: ' + escapeHtml(participantName(round.byeParticipantId)) + '</div>'
        : '';

      var matchesHtml = round.matches.map(function (match) {
        return scoreCardHtml({
          action: "rr-score",
          attrs: { "match-id": match.id },
          teamA: teamLabel(match.teamA),
          teamB: teamLabel(match.teamB),
          scoreA: match.scoreA,
          scoreB: match.scoreB,
          sideALabel: "Team A",
          sideBLabel: "Team B"
        });
      }).join('');

      return (
        '<div class="round-block">' +
          '<header><strong>Round ' + round.number + '</strong>' + byeBlock + '</header>' +
          '<div class="round-body stack">' + matchesHtml + '</div>' +
        '</div>'
      );
    }).join('');
  }

  function renderRoundRobin() {
    var metrics = roundRobinMetrics();
    var standings = roundRobinStandings();
    var winner = standings[0] ? standings[0].name : "—";

    return (
      '<section class="grid">' +
        renderFlash() +
        '<div class="card">' +
          '<div class="section-head"><div><h2>' + escapeHtml(state.event.name) + '</h2><p>' + escapeHtml(formatType(state.event.type)) + ' • ' + escapeHtml(state.event.mode === "switch-doubles" ? "Switch Partner Doubles" : "Standard") + '</p></div>' +
            '<div class="badge accent">Leader: ' + escapeHtml(winner) + '</div>' +
          '</div>' +
          '<div class="metric-grid">' + metrics.map(function (item) { return metric(item.label, item.value); }).join('') + '</div>' +
        '</div>' +
        '<div class="grid two">' +
          '<div class="card"><div class="section-head"><div><h3>Live scoring</h3><p>Enter scores as matches happen.</p></div></div><div class="stack">' + roundRobinScoreBlocks() + '</div></div>' +
          '<div class="card"><div class="section-head"><div><h3>Live standings</h3><p>Automatically sorted by wins, differential, and points for.</p></div></div>' + standingsTable(standings) + '</div>' +
        '</div>' +
      '</section>'
    );
  }

  function currentLeagueRound() {
    return state.event.rounds.find(function (round) {
      return round.number === state.event.currentRoundNumber;
    });
  }

  function leagueMetrics() {
    var round = currentLeagueRound();
    var completedMatches = Engine.ladderRoundMatches(round).filter(function (match) {
      return match.scoreA !== null && match.scoreB !== null;
    }).length;
    var totalMatches = Engine.ladderRoundMatches(round).length;
    return [
      { label: "Participants", value: state.event.participants.length },
      { label: "Courts", value: state.event.courtSizes.length },
      { label: "Round", value: state.event.currentRoundNumber + " / " + state.event.totalRounds },
      { label: "Scored Matches", value: completedMatches + " / " + totalMatches }
    ];
  }

  function leagueCourtCards(summary) {
    return summary.map(function (courtInfo) {
      var rows = courtInfo.standings.map(function (row) {
        return (
          '<tr>' +
            '<td>' + row.rank + '</td>' +
            '<td>' + escapeHtml(row.name) + '</td>' +
            '<td>' + row.wins + '</td>' +
            '<td>' + row.losses + '</td>' +
            '<td>' + row.pointsFor + '</td>' +
            '<td>' + row.pointsAgainst + '</td>' +
            '<td>' + row.differential + '</td>' +
          '</tr>'
        );
      }).join('');

      return (
        '<div class="court-card">' +
          '<header><strong>Court ' + courtInfo.courtNumber + '</strong><span class="badge">' + courtInfo.size + ' players</span></header>' +
          '<div class="body">' +
            '<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Player</th><th>W</th><th>L</th><th>PF</th><th>PA</th><th>Diff</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
          '</div>' +
        '</div>'
      );
    }).join('');
  }

  function renderLeagueScoring(round) {
    return round.courts.map(function (court) {
      var miniRoundsHtml = (court.miniRounds || []).map(function (miniRound) {
        var byeHtml = miniRound.byeParticipantId
          ? '<div class="badge">Bye: ' + escapeHtml(participantName(miniRound.byeParticipantId)) + '</div>'
          : '';

        var matchesHtml = (miniRound.matches || []).map(function (match) {
          return scoreCardHtml({
            action: "league-score",
            attrs: { "round": round.number, "match-id": match.id },
            teamA: teamLabel(match.teamA),
            teamB: teamLabel(match.teamB),
            scoreA: match.scoreA,
            scoreB: match.scoreB,
            sideALabel: "Team A",
            sideBLabel: "Team B"
          });
        }).join('');

        return (
          '<div class="round-block">' +
            '<header><strong>Court ' + court.courtNumber + ' • Mini-round ' + miniRound.number + '</strong>' + byeHtml + '</header>' +
            '<div class="round-body stack">' + matchesHtml + '</div>' +
          '</div>'
        );
      }).join('');

      return '<div class="court-stack"><h4>Court ' + court.courtNumber + '</h4>' + miniRoundsHtml + '</div>';
    }).join('');
  }

  function ensurePendingAssignments() {
    if (!state.event.pendingAssignments) {
      var movement = Engine.buildAutoMovement(state.event, state.event.currentRoundNumber);
      state.event.pendingAssignments = movement.assignments;
      saveState();
    }
  }

  function leagueMovementEditor() {
    if (!Engine.isLeagueRoundComplete(currentLeagueRound())) {
      return '<div class="alert warning">Complete every score in the current round to unlock movement suggestions.</div>';
    }

    ensurePendingAssignments();
    var movement = Engine.buildAutoMovement(state.event, state.event.currentRoundNumber);
    var rows = movement.rows.map(function (row) {
      var current = row.currentCourt;
      var next = Number(state.event.pendingAssignments[row.participantId]);
      var direction = next < current ? "⬆️ Up" : next > current ? "⬇️ Down" : "➖ Stay";
      var options = state.event.courtSizes.map(function (_size, index) {
        var courtNumber = index + 1;
        return '<option value="' + courtNumber + '"' + (courtNumber === next ? ' selected' : '') + '>Court ' + courtNumber + '</option>';
      }).join('');

      return (
        '<tr>' +
          '<td>' + escapeHtml(row.name) + '</td>' +
          '<td>' + current + '</td>' +
          '<td>' + row.currentRank + '</td>' +
          '<td>' + escapeHtml(direction) + '</td>' +
          '<td><select data-action="assignment-select" data-participant-id="' + row.participantId + '">' + options + '</select></td>' +
        '</tr>'
      );
    }).join('');

    var validation = Engine.validateAssignments(state.event, state.event.pendingAssignments);
    var validationHtml = validation.ok
      ? '<div class="alert success">' + (state.event.currentRoundNumber >= state.event.totalRounds ? 'Final round complete. League night is ready to finish.' : 'Court counts are valid. You can start the next round.') + '</div>'
      : '<div class="alert danger">' + validation.errors.map(escapeHtml).join('<br>') + '</div>';

    var isFinalRound = state.event.currentRoundNumber >= state.event.totalRounds;
    var actionsHtml = isFinalRound
      ? '<div class="badge accent">Final round</div>'
      : '<div class="inline-row"><button type="button" class="secondary small" data-action="reset-auto-movement">Reset to auto</button><button type="button" class="small" data-action="start-next-round">Start next round</button></div>';

    return (
      '<div class="stack">' +
        '<div class="section-head"><div><h3>Suggested movement</h3><p>Automatic movement uses one-up / one-down across court boundaries. Override any next-court assignment before starting the next round.</p></div>' +
          actionsHtml +
        '</div>' +
        validationHtml +
        '<div class="table-wrap"><table><thead><tr><th>Player</th><th>Current Court</th><th>Rank</th><th>Auto</th><th>Next Court</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '</div>'
    );
  }

  function leagueRoundHistory() {
    if (state.event.rounds.length <= 1) { return ''; }
    var previous = state.event.rounds.slice(0, -1).map(function (round) {
      var summary = Engine.leagueRoundSummary(state.event, round.number) || [];
      return (
        '<details class="card">' +
          '<summary><strong>Completed round ' + round.number + '</strong></summary>' +
          '<div style="margin-top:14px;" class="cards-grid">' + leagueCourtCards(summary) + '</div>' +
        '</details>'
      );
    }).join('');

    return '<div class="stack">' + previous + '</div>';
  }

  function renderLeague() {
    var round = currentLeagueRound();
    var summary = Engine.leagueRoundSummary(state.event, state.event.currentRoundNumber) || [];
    var aggregate = Engine.leagueAggregateStandings(state.event);
    var allDone = state.event.currentRoundNumber >= state.event.totalRounds &&
      Engine.isLeagueRoundComplete(round);

    return (
      '<section class="grid">' +
        renderFlash() +
        '<div class="card">' +
          '<div class="section-head"><div><h2>' + escapeHtml(state.event.name) + '</h2><p>League / ladder night with live movement between rounds.</p></div>' +
            '<div class="badge accent">' + escapeHtml("Round " + state.event.currentRoundNumber + " of " + state.event.totalRounds) + '</div>' +
          '</div>' +
          '<div class="metric-grid">' + leagueMetrics().map(function (item) { return metric(item.label, item.value); }).join('') + '</div>' +
        '</div>' +
        '<div class="grid two">' +
          '<div class="card"><div class="section-head"><div><h3>Current round scoring</h3><p>Enter court scores, then review standings and movement.</p></div></div>' + renderLeagueScoring(round) + '</div>' +
          '<div class="stack">' +
            '<div class="card"><div class="section-head"><div><h3>Current court standings</h3><p>Sorted inside each court.</p></div></div><div class="cards-grid">' + leagueCourtCards(summary) + '</div></div>' +
            '<div class="card">' + leagueMovementEditor() + '</div>' +
          '</div>' +
        '</div>' +
        '<div class="grid two">' +
          '<div class="card"><div class="section-head"><div><h3>Cumulative ladder standings</h3><p>All completed rounds combined.</p></div></div>' + standingsTable(aggregate) + '</div>' +
          '<div class="card"><div class="section-head"><div><h3>Completed round history</h3><p>Previous rounds stay visible for review and printing.</p></div></div>' + (leagueRoundHistory() || '<div class="note">No completed rounds yet.</div>') + '</div>' +
        '</div>' +
        (allDone ? '<div class="card"><div class="alert success">League night complete. Final standings are locked into this local event file. Export the JSON to archive or share it.</div></div>' : '') +
      '</section>'
    );
  }

  function renderTournament() {
    var championId = Engine.tournamentChampion(state.event);
    var championName = championId ? participantName(championId) : "TBD";
    var metrics = [
      { label: "Teams", value: state.event.participants.length },
      { label: "Bracket size", value: state.event.bracketSize },
      { label: "Rounds", value: state.event.rounds.length },
      { label: "Champion", value: championName }
    ];

    var columns = state.event.rounds.map(function (round) {
      var matchesHtml = round.matches.map(function (match) {
        var leftName = match.participantAId ? participantName(match.participantAId) : "TBD";
        var rightName = match.participantBId ? participantName(match.participantBId) : "TBD";
        var winnerId = match.winnerId;

        return (
          '<div class="bracket-match ' + (winnerId ? 'winner' : '') + '">' +
            '<div class="subtle">Match ' + match.slot + '</div>' +
            scoreCardHtml({
              action: "tournament-score",
              attrs: { "round": round.number, "slot": match.slot },
              teamA: leftName,
              teamB: rightName,
              scoreA: match.scoreA,
              scoreB: match.scoreB,
              sideALabel: "Team 1",
              sideBLabel: "Team 2"
            }) +
            '<div class="subtle">Winner: ' + escapeHtml(winnerId ? participantName(winnerId) : "Pending") + '</div>' +
          '</div>'
        );
      }).join('');

      return (
        '<div class="bracket-column">' +
          '<header><strong>Round ' + round.number + '</strong></header>' +
          '<div class="body stack">' + matchesHtml + '</div>' +
        '</div>'
      );
    }).join('');

    return (
      '<section class="grid">' +
        renderFlash() +
        '<div class="card">' +
          '<div class="section-head"><div><h2>' + escapeHtml(state.event.name) + '</h2><p>Single-elimination team bracket with live advancement.</p></div>' +
            '<div class="badge accent">Champion: ' + escapeHtml(championName) + '</div>' +
          '</div>' +
          '<div class="metric-grid">' + metrics.map(function (item) { return metric(item.label, item.value); }).join('') + '</div>' +
        '</div>' +
        '<div class="card">' +
          '<div class="section-head"><div><h3>Live bracket</h3><p>Enter both team scores. Winners advance automatically. Tied scores do not advance anyone.</p></div></div>' +
          '<div class="bracket-grid">' + columns + '</div>' +
        '</div>' +
      '</section>'
    );
  }

  function renderEvent() {
    if (!state.event) { return renderSetup(); }

    var content = '';
    if (state.event.type === "round-robin") {
      content = renderRoundRobin();
    } else if (state.event.type === "league") {
      content = renderLeague();
    } else {
      content = renderTournament();
    }

    return (
      '<section class="grid">' +
        '<div class="topbar screen-only">' +
          '<div class="status-chip"><span>Local only</span> • <span>No DB</span> • <span>Autosave on</span></div>' +
          '<div class="toolbar">' +
            '<button type="button" class="secondary" data-action="new-event">New event</button>' +
            '<button type="button" class="secondary" data-action="export-event">Export JSON</button>' +
            '<button type="button" class="secondary" data-action="import-trigger">Import JSON</button>' +
            '<button type="button" class="secondary" data-action="print-event">Print</button>' +
          '</div>' +
        '</div>' +
        content +
      '</section>'
    );
  }

  function renderAppShell() {
    return (
      '<div class="app-shell">' +
        '<header class="topbar">' +
          '<div class="branding">' +
            '<h1>Live Brackets MVP</h1>' +
            '<p>Round robins, ladders, and tournaments — run live in the browser.</p>' +
          '</div>' +
          '<div class="toolbar screen-only">' +
            (state.event ? (
              '<button type="button" class="ghost" data-action="new-event">Start over</button>' +
              '<button type="button" class="ghost" data-action="export-event">Export</button>' +
              '<button type="button" class="ghost" data-action="import-trigger">Import</button>' +
              '<button type="button" class="ghost" data-action="print-event">Print</button>'
            ) : (
              '<div class="status-chip">Static HTML • JSON backup • GitHub-ready</div>'
            )) +
          '</div>' +
        '</header>' +
        renderEvent() +
        '<footer class="app-footer">Built as a browser-only MVP so the next step can be a GitHub repo, then optional club accounts and cloud sync.</footer>' +
      '</div>'
    );
  }

  function render() {
    renderScoreIndex = 0;
    app.innerHTML = renderAppShell();
    applyPendingFocus();
  }

  document.addEventListener("click", function (event) {
    var target = event.target.closest("[data-action]");
    if (!target) { return; }
    if (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA") {
      return;
    }

    var action = target.getAttribute("data-action");

    if (action === "set-type") {
      setSetupValue("type", target.getAttribute("data-value"), true);
      return;
    }
    if (action === "set-rr-mode") {
      setSetupValue("rrMode", target.getAttribute("data-value"), true);
      return;
    }
    if (action === "create-event") {
      createEvent();
      return;
    }
    if (action === "new-event") {
      resetEvent();
      return;
    }
    if (action === "export-event") {
      exportEvent();
      return;
    }
    if (action === "import-trigger") {
      importInput.click();
      return;
    }
    if (action === "print-event") {
      printEvent();
      return;
    }
    if (action === "reset-auto-movement") {
      if (!state.event || state.event.type !== "league") { return; }
      try {
        state.event.pendingAssignments = Engine.buildAutoMovement(state.event, state.event.currentRoundNumber).assignments;
        saveState();
        render();
      } catch (error) {
        flash(error.message || "Could not reset movement.", "danger");
      }
      return;
    }
    if (action === "start-next-round") {
      if (!state.event || state.event.type !== "league") { return; }
      try {
        Engine.startNextLeagueRound(state.event);
        saveState();
        flash("Next round started.", "success");
      } catch (error) {
        flash(error.message || "Could not start next round.", "danger");
      }
      return;
    }
  });


  document.addEventListener("keydown", function (event) {
    var target = event.target;
    if (!target || !target.classList || !target.classList.contains("score-input")) { return; }

    var currentIndex = Number(target.getAttribute("data-score-index") || 0);
    if (!currentIndex) { return; }

    if (event.key === "Enter") {
      event.preventDefault();
      pendingFocusIndex = currentIndex + 1;
      target.blur();
      return;
    }

    if (event.key === "Tab") {
      pendingFocusIndex = event.shiftKey ? currentIndex - 1 : currentIndex + 1;
    }
  });

  document.addEventListener("input", function (event) {
    var target = event.target;
    var field = target.getAttribute("data-field");
    if (!field || state.event) { return; }
    setSetupValue(field, target.value, false);
  });

  document.addEventListener("change", function (event) {
    var target = event.target;
    var action = target.getAttribute("data-action");
    var field = target.getAttribute("data-field");

    if (field && !state.event) {
      setSetupValue(field, target.value, true);
      return;
    }

    if (!state.event) { return; }

    if (action === "rr-score") {
      Engine.updateRoundRobinScore(
        state.event,
        target.getAttribute("data-match-id"),
        target.getAttribute("data-score-key"),
        target.value
      );
      saveState();
      render();
      return;
    }

    if (action === "league-score") {
      Engine.updateLeagueScore(
        state.event,
        Number(target.getAttribute("data-round")),
        target.getAttribute("data-match-id"),
        target.getAttribute("data-score-key"),
        target.value
      );
      if (state.event.pendingAssignments) {
        state.event.pendingAssignments = null;
      }
      saveState();
      render();
      return;
    }

    if (action === "assignment-select") {
      state.event.pendingAssignments = state.event.pendingAssignments || {};
      state.event.pendingAssignments[target.getAttribute("data-participant-id")] = Number(target.value);
      saveState();
      render();
      return;
    }

    if (action === "tournament-score") {
      Engine.updateTournamentScore(
        state.event,
        Number(target.getAttribute("data-round")),
        Number(target.getAttribute("data-slot")),
        target.getAttribute("data-score-key"),
        target.value
      );
      saveState();
      render();
    }
  });

  importInput.addEventListener("change", function (event) {
    importEvent(event.target.files && event.target.files[0]);
    importInput.value = "";
  });

  loadState();
  render();
})();

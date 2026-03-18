(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.LiveEngine = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var SWITCH_DOUBLES_TEMPLATES = {
  "4": [
    {
      "number": 1,
      "matches": [
        [
          [
            2,
            1
          ],
          [
            3,
            4
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 2,
      "matches": [
        [
          [
            4,
            2
          ],
          [
            1,
            3
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 3,
      "matches": [
        [
          [
            4,
            1
          ],
          [
            2,
            3
          ]
        ]
      ],
      "bye": null
    }
  ],
  "5": [
    {
      "number": 1,
      "matches": [
        [
          [
            1,
            2
          ],
          [
            3,
            4
          ]
        ]
      ],
      "bye": 5
    },
    {
      "number": 2,
      "matches": [
        [
          [
            2,
            4
          ],
          [
            3,
            5
          ]
        ]
      ],
      "bye": 1
    },
    {
      "number": 3,
      "matches": [
        [
          [
            1,
            5
          ],
          [
            2,
            3
          ]
        ]
      ],
      "bye": 4
    },
    {
      "number": 4,
      "matches": [
        [
          [
            1,
            3
          ],
          [
            4,
            5
          ]
        ]
      ],
      "bye": 2
    },
    {
      "number": 5,
      "matches": [
        [
          [
            1,
            4
          ],
          [
            2,
            5
          ]
        ]
      ],
      "bye": 3
    }
  ],
  "8": [
    {
      "number": 1,
      "matches": [
        [
          [
            1,
            6
          ],
          [
            2,
            5
          ]
        ],
        [
          [
            3,
            8
          ],
          [
            4,
            7
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 2,
      "matches": [
        [
          [
            2,
            3
          ],
          [
            5,
            8
          ]
        ],
        [
          [
            1,
            4
          ],
          [
            6,
            7
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 3,
      "matches": [
        [
          [
            1,
            8
          ],
          [
            3,
            6
          ]
        ],
        [
          [
            2,
            7
          ],
          [
            4,
            5
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 4,
      "matches": [
        [
          [
            1,
            2
          ],
          [
            3,
            4
          ]
        ],
        [
          [
            5,
            6
          ],
          [
            7,
            8
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 5,
      "matches": [
        [
          [
            1,
            7
          ],
          [
            2,
            8
          ]
        ],
        [
          [
            3,
            5
          ],
          [
            4,
            6
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 6,
      "matches": [
        [
          [
            2,
            6
          ],
          [
            3,
            7
          ]
        ],
        [
          [
            1,
            5
          ],
          [
            4,
            8
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 7,
      "matches": [
        [
          [
            2,
            4
          ],
          [
            6,
            8
          ]
        ],
        [
          [
            1,
            3
          ],
          [
            5,
            7
          ]
        ]
      ],
      "bye": null
    }
  ],
  "12": [
    {
      "number": 1,
      "matches": [
        [
          [
            3,
            6
          ],
          [
            4,
            11
          ]
        ],
        [
          [
            5,
            7
          ],
          [
            9,
            10
          ]
        ],
        [
          [
            12,
            1
          ],
          [
            2,
            8
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 2,
      "matches": [
        [
          [
            6,
            9
          ],
          [
            7,
            3
          ]
        ],
        [
          [
            8,
            10
          ],
          [
            1,
            2
          ]
        ],
        [
          [
            12,
            4
          ],
          [
            5,
            11
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 3,
      "matches": [
        [
          [
            11,
            2
          ],
          [
            4,
            5
          ]
        ],
        [
          [
            12,
            7
          ],
          [
            8,
            3
          ]
        ],
        [
          [
            9,
            1
          ],
          [
            10,
            6
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 4,
      "matches": [
        [
          [
            12,
            10
          ],
          [
            11,
            6
          ]
        ],
        [
          [
            1,
            4
          ],
          [
            2,
            9
          ]
        ],
        [
          [
            3,
            5
          ],
          [
            7,
            8
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 5,
      "matches": [
        [
          [
            4,
            7
          ],
          [
            5,
            1
          ]
        ],
        [
          [
            6,
            8
          ],
          [
            10,
            11
          ]
        ],
        [
          [
            12,
            2
          ],
          [
            3,
            9
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 6,
      "matches": [
        [
          [
            9,
            11
          ],
          [
            2,
            3
          ]
        ],
        [
          [
            12,
            5
          ],
          [
            6,
            1
          ]
        ],
        [
          [
            7,
            10
          ],
          [
            8,
            4
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 7,
      "matches": [
        [
          [
            12,
            8
          ],
          [
            9,
            4
          ]
        ],
        [
          [
            10,
            2
          ],
          [
            11,
            7
          ]
        ],
        [
          [
            1,
            3
          ],
          [
            5,
            6
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 8,
      "matches": [
        [
          [
            2,
            5
          ],
          [
            3,
            10
          ]
        ],
        [
          [
            4,
            6
          ],
          [
            8,
            9
          ]
        ],
        [
          [
            12,
            11
          ],
          [
            1,
            7
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 9,
      "matches": [
        [
          [
            7,
            9
          ],
          [
            11,
            1
          ]
        ],
        [
          [
            5,
            8
          ],
          [
            6,
            2
          ]
        ],
        [
          [
            12,
            3
          ],
          [
            4,
            10
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 10,
      "matches": [
        [
          [
            12,
            6
          ],
          [
            7,
            2
          ]
        ],
        [
          [
            10,
            1
          ],
          [
            3,
            4
          ]
        ],
        [
          [
            8,
            11
          ],
          [
            9,
            5
          ]
        ]
      ],
      "bye": null
    },
    {
      "number": 11,
      "matches": [
        [
          [
            11,
            3
          ],
          [
            1,
            8
          ]
        ],
        [
          [
            12,
            9
          ],
          [
            10,
            5
          ]
        ],
        [
          [
            2,
            4
          ],
          [
            6,
            7
          ]
        ]
      ],
      "bye": null
    }
  ]
};

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function uid(prefix) {
    return String(prefix || "id") + "-" + Math.random().toString(36).slice(2, 10);
  }

  function normalizeName(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/\u00A0/g, " ")
      .trim()
      .replace(/\s+/g, " ");
  }

  function asInteger(value) {
    if (value === "" || value === null || value === undefined) {
      return null;
    }
    var number = Number(value);
    if (!Number.isFinite(number)) {
      return null;
    }
    return Math.round(number);
  }

  function participantsById(participants) {
    return (participants || []).reduce(function (accumulator, participant) {
      accumulator[participant.id] = participant;
      return accumulator;
    }, {});
  }

  function participantIdsFromParticipants(participants) {
    return (participants || []).map(function (participant) { return participant.id; });
  }

  function buildParticipants(count, names) {
    var cleanNames = Array.isArray(names) ? names.map(normalizeName).filter(Boolean) : [];
    var participants = [];
    for (var index = 0; index < count; index += 1) {
      participants.push({
        id: "p" + (index + 1),
        seed: index + 1,
        name: cleanNames[index] || ("Player " + (index + 1))
      });
    }
    return participants;
  }

  function templateSupportedCount(count) {
    return Object.prototype.hasOwnProperty.call(SWITCH_DOUBLES_TEMPLATES, String(count)) ||
      Object.prototype.hasOwnProperty.call(SWITCH_DOUBLES_TEMPLATES, count);
  }

  function parseCourtSizes(value) {
    if (Array.isArray(value)) {
      return value.map(function (item) { return Number(item); })
        .filter(function (item) { return Number.isFinite(item) && item > 0; });
    }
    return String(value || "")
      .split(/[\n,\s]+/)
      .map(function (item) { return Number(item); })
      .filter(function (item) { return Number.isFinite(item) && item > 0; });
  }

  function suggestCourtSizes(participantCount) {
    var count = Number(participantCount || 0);
    if (!Number.isInteger(count) || count < 4) { return null; }

    for (var courts = 1; courts <= Math.ceil(count / 4); courts += 1) {
      for (var numFives = 0; numFives <= courts; numFives += 1) {
        var total = (courts - numFives) * 4 + (numFives * 5);
        if (total === count) {
          var result = [];
          for (var i = 0; i < courts - numFives; i += 1) { result.push(4); }
          for (var j = 0; j < numFives; j += 1) { result.push(5); }
          return result;
        }
      }
    }
    return null;
  }

  function findMatchInRounds(rounds, matchId) {
    for (var i = 0; i < (rounds || []).length; i += 1) {
      var round = rounds[i];
      if (round.matches) {
        for (var j = 0; j < round.matches.length; j += 1) {
          if (round.matches[j].id === matchId) {
            return round.matches[j];
          }
        }
      }
      if (round.courts) {
        for (var c = 0; c < round.courts.length; c += 1) {
          var court = round.courts[c];
          var miniRounds = court.miniRounds || [];
          for (var mr = 0; mr < miniRounds.length; mr += 1) {
            var matches = miniRounds[mr].matches || [];
            for (var m = 0; m < matches.length; m += 1) {
              if (matches[m].id === matchId) {
                return matches[m];
              }
            }
          }
        }
      }
    }
    return null;
  }

  function createMatch(id, teamA, teamB) {
    return {
      id: id || uid("match"),
      teamA: teamA || [],
      teamB: teamB || [],
      scoreA: null,
      scoreB: null
    };
  }

  function mapTemplateTeam(participants, indexes) {
    return (indexes || []).map(function (index) {
      return participants[index - 1] ? participants[index - 1].id : null;
    }).filter(Boolean);
  }

  function buildSwitchDoublesRounds(participants) {
    var template = clone(SWITCH_DOUBLES_TEMPLATES[participants.length]);
    if (!template) {
      throw new Error("Switch Partner Doubles currently supports 4, 5, 8, and 12 players.");
    }

    return template.map(function (templateRound) {
      return {
        id: uid("rr-round"),
        number: templateRound.number,
        byeParticipantId: templateRound.bye ? participants[templateRound.bye - 1].id : null,
        matches: (templateRound.matches || []).map(function (match, index) {
          return createMatch(
            "rr-r" + templateRound.number + "-m" + (index + 1),
            mapTemplateTeam(participants, match[0]),
            mapTemplateTeam(participants, match[1])
          );
        })
      };
    });
  }

  function buildStandardRoundRobinRounds(participants) {
    var rotation = participantIdsFromParticipants(participants).slice();
    if (rotation.length % 2 === 1) {
      rotation.push(null);
    }

    var total = rotation.length;
    var rounds = [];

    for (var roundNumber = 1; roundNumber < total; roundNumber += 1) {
      var matches = [];
      var byeId = null;

      for (var pairIndex = 0; pairIndex < total / 2; pairIndex += 1) {
        var left = rotation[pairIndex];
        var right = rotation[total - 1 - pairIndex];

        if (left === null || right === null) {
          byeId = left === null ? right : left;
          continue;
        }

        matches.push(createMatch(
          "rr-r" + roundNumber + "-m" + (pairIndex + 1),
          [left],
          [right]
        ));
      }

      rounds.push({
        id: uid("rr-round"),
        number: roundNumber,
        byeParticipantId: byeId,
        matches: matches
      });

      var fixed = rotation[0];
      var rest = rotation.slice(1);
      rest.unshift(rest.pop());
      rotation = [fixed].concat(rest);
    }

    return rounds;
  }

  function allRoundRobinMatches(rounds) {
    return (rounds || []).reduce(function (accumulator, round) {
      return accumulator.concat(round.matches || []);
    }, []);
  }

  function computeParticipantStandingsFromMatches(matches, participants) {
    var standings = {};
    (participants || []).forEach(function (participant) {
      standings[participant.id] = {
        participantId: participant.id,
        name: participant.name,
        wins: 0,
        losses: 0,
        ties: 0,
        matches: 0,
        pointsFor: 0,
        pointsAgainst: 0,
        differential: 0
      };
    });

    (matches || []).forEach(function (match) {
      if (match.scoreA === null || match.scoreB === null ||
          !Number.isFinite(Number(match.scoreA)) || !Number.isFinite(Number(match.scoreB))) {
        return;
      }

      var scoreA = Number(match.scoreA);
      var scoreB = Number(match.scoreB);
      var result = scoreA === scoreB ? "tie" : (scoreA > scoreB ? "A" : "B");

      (match.teamA || []).forEach(function (participantIdValue) {
        var row = standings[participantIdValue];
        if (!row) { return; }
        row.matches += 1;
        row.pointsFor += scoreA;
        row.pointsAgainst += scoreB;
        row.differential += scoreA - scoreB;
        if (result === "tie") {
          row.ties += 1;
        } else if (result === "A") {
          row.wins += 1;
        } else {
          row.losses += 1;
        }
      });

      (match.teamB || []).forEach(function (participantIdValue) {
        var row = standings[participantIdValue];
        if (!row) { return; }
        row.matches += 1;
        row.pointsFor += scoreB;
        row.pointsAgainst += scoreA;
        row.differential += scoreB - scoreA;
        if (result === "tie") {
          row.ties += 1;
        } else if (result === "B") {
          row.wins += 1;
        } else {
          row.losses += 1;
        }
      });
    });

    var rows = Object.keys(standings).map(function (key) {
      return standings[key];
    });

    rows.sort(function (left, right) {
      if (right.wins !== left.wins) { return right.wins - left.wins; }
      if (right.differential !== left.differential) { return right.differential - left.differential; }
      if (right.pointsFor !== left.pointsFor) { return right.pointsFor - left.pointsFor; }
      if (left.losses !== right.losses) { return left.losses - right.losses; }
      return left.name.localeCompare(right.name);
    });

    rows.forEach(function (row, index) {
      row.rank = index + 1;
    });

    return rows;
  }

  function createRoundRobinEvent(options) {
    var count = Number((options && options.participantCount) || 0);
    if (!Number.isInteger(count) || count < 2) {
      throw new Error("Round robin needs at least 2 participants.");
    }

    var participants = buildParticipants(count, options && options.names);
    var mode = options && options.mode === "standard" ? "standard" : "switch-doubles";
    var rounds = mode === "switch-doubles"
      ? buildSwitchDoublesRounds(participants)
      : buildStandardRoundRobinRounds(participants);

    return {
      schemaVersion: 1,
      id: uid("event"),
      name: normalizeName(options && options.name) || "Round Robin",
      type: "round-robin",
      mode: mode,
      participants: participants,
      rounds: rounds,
      createdAt: new Date().toISOString()
    };
  }

  function updateRoundRobinScore(event, matchId, scoreKey, value) {
    var match = findMatchInRounds(event.rounds, matchId);
    if (!match) { return; }
    match[scoreKey] = asInteger(value);
  }

  function buildLeagueRound(roundNumber, courtGroups) {
    return {
      number: roundNumber,
      courts: courtGroups.map(function (participantIds, index) {
        var size = participantIds.length;
        if (!templateSupportedCount(size) || (size !== 4 && size !== 5)) {
          throw new Error("League courts currently support 4-player or 5-player groups only.");
        }

        var template = clone(SWITCH_DOUBLES_TEMPLATES[size]);
        return {
          courtNumber: index + 1,
          size: size,
          participantIds: participantIds.slice(),
          miniRounds: template.map(function (templateRound) {
            return {
              number: templateRound.number,
              byeParticipantId: templateRound.bye ? participantIds[templateRound.bye - 1] : null,
              matches: (templateRound.matches || []).map(function (match, matchIndex) {
                var teamA = match[0].map(function (position) { return participantIds[position - 1]; });
                var teamB = match[1].map(function (position) { return participantIds[position - 1]; });
                return createMatch(
                  "lg-r" + roundNumber + "-c" + (index + 1) + "-mr" + templateRound.number + "-m" + (matchIndex + 1),
                  teamA,
                  teamB
                );
              })
            };
          })
        };
      })
    };
  }

  function createLeagueEvent(options) {
    var count = Number((options && options.participantCount) || 0);
    var totalRounds = Number((options && options.totalRounds) || 3);
    var courtSizes = parseCourtSizes(options && options.courtSizes);

    if (!Number.isInteger(count) || count < 4) {
      throw new Error("League mode needs at least 4 participants.");
    }
    if (!Number.isInteger(totalRounds) || totalRounds < 1) {
      throw new Error("Total rounds must be at least 1.");
    }
    if (!courtSizes.length) {
      throw new Error("Enter a valid court layout such as 4,4 or 4,5.");
    }

    var totalSeats = courtSizes.reduce(function (sum, size) { return sum + size; }, 0);
    if (totalSeats !== count) {
      throw new Error("Court layout must add up to the participant count.");
    }

    courtSizes.forEach(function (size) {
      if (size !== 4 && size !== 5) {
        throw new Error("League courts currently support only 4-player and 5-player groups.");
      }
    });

    var participants = buildParticipants(count, options && options.names);
    var allIds = participantIdsFromParticipants(participants);
    var courtGroups = [];
    var cursor = 0;
    courtSizes.forEach(function (size) {
      courtGroups.push(allIds.slice(cursor, cursor + size));
      cursor += size;
    });

    return {
      schemaVersion: 1,
      id: uid("event"),
      name: normalizeName(options && options.name) || "League Night",
      type: "league",
      participants: participants,
      courtSizes: courtSizes.slice(),
      totalRounds: totalRounds,
      currentRoundNumber: 1,
      rounds: [buildLeagueRound(1, courtGroups)],
      pendingAssignments: null,
      createdAt: new Date().toISOString()
    };
  }

  function findLeagueRound(event, roundNumber) {
    return (event.rounds || []).find(function (round) { return round.number === roundNumber; }) || null;
  }

  function ladderRoundMatches(round) {
    return ((round && round.courts) || []).reduce(function (courtAccumulator, court) {
      return courtAccumulator.concat((court.miniRounds || []).reduce(function (miniAccumulator, miniRound) {
        return miniAccumulator.concat(miniRound.matches || []);
      }, []));
    }, []);
  }

  function updateLeagueScore(event, roundNumber, matchId, scoreKey, value) {
    var round = findLeagueRound(event, roundNumber);
    if (!round) { return; }
    var match = findMatchInRounds([round], matchId);
    if (!match) { return; }
    match[scoreKey] = asInteger(value);
  }

  function courtParticipants(event, participantIds) {
    var byId = participantsById(event.participants);
    return (participantIds || []).map(function (participantIdValue) {
      return byId[participantIdValue];
    }).filter(Boolean);
  }

  function leagueRoundSummary(event, roundNumber) {
    var round = findLeagueRound(event, roundNumber);
    if (!round) { return []; }
    return (round.courts || []).map(function (court) {
      var matches = ladderRoundMatches({ courts: [court] });
      var standings = computeParticipantStandingsFromMatches(matches, courtParticipants(event, court.participantIds));
      standings.forEach(function (row) {
        row.currentCourt = court.courtNumber;
      });
      return {
        courtNumber: court.courtNumber,
        size: court.participantIds.length,
        participantIds: court.participantIds.slice(),
        standings: standings
      };
    });
  }

  function leagueAggregateStandings(event) {
    var matches = (event.rounds || []).reduce(function (accumulator, round) {
      return accumulator.concat(ladderRoundMatches(round));
    }, []);
    return computeParticipantStandingsFromMatches(matches, event.participants);
  }

  function isLeagueRoundComplete(round) {
    return ladderRoundMatches(round).every(function (match) {
      return match.scoreA !== null && match.scoreB !== null;
    });
  }

  function buildAutoMovement(event, roundNumber) {
    var summary = leagueRoundSummary(event, roundNumber);
    var assignments = {};
    var rowMap = {};
    var totalCourts = event.courtSizes.length;

    summary.forEach(function (courtInfo) {
      courtInfo.standings.forEach(function (row) {
        assignments[row.participantId] = courtInfo.courtNumber;
        rowMap[row.participantId] = {
          participantId: row.participantId,
          name: row.name,
          currentCourt: courtInfo.courtNumber,
          currentRank: row.rank,
          wins: row.wins,
          losses: row.losses,
          ties: row.ties,
          pointsFor: row.pointsFor,
          pointsAgainst: row.pointsAgainst,
          differential: row.differential
        };
      });
    });

    summary.forEach(function (courtInfo) {
      var rows = courtInfo.standings;
      if (!rows.length) { return; }
      if (courtInfo.courtNumber > 1) {
        assignments[rows[0].participantId] = courtInfo.courtNumber - 1;
      }
      if (courtInfo.courtNumber < totalCourts) {
        assignments[rows[rows.length - 1].participantId] = courtInfo.courtNumber + 1;
      }
    });

    var rows = Object.keys(assignments).map(function (participantIdValue) {
      var info = rowMap[participantIdValue];
      return {
        participantId: participantIdValue,
        name: info.name,
        currentCourt: info.currentCourt,
        currentRank: info.currentRank,
        proposedCourt: Number(assignments[participantIdValue]),
        wins: info.wins,
        losses: info.losses,
        ties: info.ties,
        pointsFor: info.pointsFor,
        pointsAgainst: info.pointsAgainst,
        differential: info.differential
      };
    });

    rows.sort(function (left, right) {
      if (left.currentCourt !== right.currentCourt) { return left.currentCourt - right.currentCourt; }
      if (left.currentRank !== right.currentRank) { return left.currentRank - right.currentRank; }
      return left.name.localeCompare(right.name);
    });

    return {
      assignments: assignments,
      rows: rows
    };
  }

  function validateAssignments(event, assignments) {
    var errors = [];
    var counts = event.courtSizes.map(function () { return 0; });

    event.participants.forEach(function (participant) {
      var courtNumber = Number(assignments[participant.id]);
      if (!Number.isInteger(courtNumber) || courtNumber < 1 || courtNumber > event.courtSizes.length) {
        errors.push(participant.name + " is assigned to an invalid court.");
        return;
      }
      counts[courtNumber - 1] += 1;
    });

    counts.forEach(function (count, index) {
      if (count !== event.courtSizes[index]) {
        errors.push("Court " + (index + 1) + " needs " + event.courtSizes[index] + " players, but has " + count + ".");
      }
    });

    return {
      ok: errors.length === 0,
      counts: counts,
      errors: errors
    };
  }

  function orderParticipantsForNextCourt(event, movementRows) {
    var grouped = event.courtSizes.map(function () { return []; });
    movementRows.forEach(function (row) {
      grouped[row.proposedCourt - 1].push(row);
    });

    grouped.forEach(function (rows) {
      rows.sort(function (left, right) {
        if (left.currentCourt !== right.currentCourt) { return left.currentCourt - right.currentCourt; }
        if (left.currentRank !== right.currentRank) { return left.currentRank - right.currentRank; }
        return left.name.localeCompare(right.name);
      });
    });

    return grouped.map(function (rows) {
      return rows.map(function (row) { return row.participantId; });
    });
  }

  function startNextLeagueRound(event) {
    if (event.currentRoundNumber >= event.totalRounds) {
      throw new Error("All configured rounds are already complete.");
    }

    var currentRound = findLeagueRound(event, event.currentRoundNumber);
    if (!currentRound) {
      throw new Error("Current round not found.");
    }
    if (!isLeagueRoundComplete(currentRound)) {
      throw new Error("Complete all scores in the current round first.");
    }

    var baseMovement = buildAutoMovement(event, event.currentRoundNumber);
    var assignments = clone(event.pendingAssignments || baseMovement.assignments);
    var validation = validateAssignments(event, assignments);
    if (!validation.ok) {
      throw new Error(validation.errors.join(" "));
    }

    var rowById = {};
    baseMovement.rows.forEach(function (row) {
      rowById[row.participantId] = clone(row);
    });

    Object.keys(assignments).forEach(function (participantIdValue) {
      if (!rowById[participantIdValue]) {
        return;
      }
      rowById[participantIdValue].proposedCourt = Number(assignments[participantIdValue]);
    });

    var nextCourtGroups = orderParticipantsForNextCourt(
      event,
      Object.keys(rowById).map(function (key) { return rowById[key]; })
    );
    var nextRoundNumber = event.currentRoundNumber + 1;
    event.rounds.push(buildLeagueRound(nextRoundNumber, nextCourtGroups));
    event.currentRoundNumber = nextRoundNumber;
    event.pendingAssignments = null;
    return event;
  }

  function nextPowerOfTwo(value) {
    var result = 1;
    while (result < value) {
      result *= 2;
    }
    return result;
  }

  function buildSeedOrder(size) {
    if (size === 1) { return [1]; }
    var previous = buildSeedOrder(size / 2);
    var result = [];
    previous.forEach(function (seed) {
      result.push(seed);
      result.push(size + 1 - seed);
    });
    return result;
  }

  function createTournamentEvent(options) {
    var count = Number((options && options.participantCount) || 0);
    if (!Number.isInteger(count) || count < 2) {
      throw new Error("Tournament mode needs at least 2 participants.");
    }

    var participants = buildParticipants(count, options && options.names);
    var bracketSize = nextPowerOfTwo(participants.length);
    var seedOrder = buildSeedOrder(bracketSize);
    var seeded = {};
    participants.forEach(function (participant) {
      seeded[participant.seed] = participant.id;
    });

    var rounds = [];
    var firstRoundMatches = [];
    for (var i = 0; i < bracketSize; i += 2) {
      var seedA = seedOrder[i];
      var seedB = seedOrder[i + 1];
      firstRoundMatches.push({
        slot: (i / 2) + 1,
        participantAId: seeded[seedA] || null,
        participantBId: seeded[seedB] || null,
        scoreA: null,
        scoreB: null,
        winnerId: null,
        sourceA: null,
        sourceB: null
      });
    }
    rounds.push({
      number: 1,
      matches: firstRoundMatches
    });

    var matchesInRound = firstRoundMatches.length;
    var roundNumber = 2;
    while (matchesInRound > 1) {
      matchesInRound = matchesInRound / 2;
      var nextMatches = [];
      for (var slot = 1; slot <= matchesInRound; slot += 1) {
        nextMatches.push({
          slot: slot,
          participantAId: null,
          participantBId: null,
          scoreA: null,
          scoreB: null,
          winnerId: null,
          sourceA: { roundNumber: roundNumber - 1, slot: (slot * 2) - 1 },
          sourceB: { roundNumber: roundNumber - 1, slot: slot * 2 }
        });
      }
      rounds.push({
        number: roundNumber,
        matches: nextMatches
      });
      roundNumber += 1;
    }

    var event = {
      schemaVersion: 1,
      id: uid("event"),
      name: normalizeName(options && options.name) || "Tournament",
      type: "tournament",
      participants: participants,
      bracketSize: bracketSize,
      rounds: rounds,
      createdAt: new Date().toISOString()
    };
    resolveTournament(event);
    return event;
  }

  function findTournamentMatch(event, roundNumber, slot) {
    var round = (event.rounds || []).find(function (entry) { return entry.number === roundNumber; });
    if (!round) { return null; }
    return (round.matches || []).find(function (match) { return match.slot === slot; }) || null;
  }

  function resolveTournament(event) {
    (event.rounds || []).forEach(function (round) {
      (round.matches || []).forEach(function (match) {
        var incomingA = match.participantAId;
        var incomingB = match.participantBId;

        if (match.sourceA) {
          var sourceA = findTournamentMatch(event, match.sourceA.roundNumber, match.sourceA.slot);
          incomingA = sourceA ? sourceA.winnerId : null;
        }
        if (match.sourceB) {
          var sourceB = findTournamentMatch(event, match.sourceB.roundNumber, match.sourceB.slot);
          incomingB = sourceB ? sourceB.winnerId : null;
        }

        var participantsChanged = false;
        if (match.sourceA && match.participantAId !== incomingA) { participantsChanged = true; }
        if (match.sourceB && match.participantBId !== incomingB) { participantsChanged = true; }

        match.participantAId = incomingA;
        match.participantBId = incomingB;

        if (participantsChanged) {
          match.scoreA = null;
          match.scoreB = null;
          match.winnerId = null;
        }

        var left = match.participantAId;
        var right = match.participantBId;

        if (left && !right) {
          match.winnerId = left;
          return;
        }
        if (!left && right) {
          match.winnerId = right;
          return;
        }
        if (!left && !right) {
          match.winnerId = null;
          return;
        }
        if (match.scoreA === null || match.scoreB === null) {
          match.winnerId = null;
          return;
        }
        if (Number(match.scoreA) === Number(match.scoreB)) {
          match.winnerId = null;
          return;
        }
        match.winnerId = Number(match.scoreA) > Number(match.scoreB) ? left : right;
      });
    });
    return event;
  }

  function updateTournamentScore(event, roundNumber, slot, scoreKey, value) {
    var match = findTournamentMatch(event, roundNumber, slot);
    if (!match) { return; }
    match[scoreKey] = asInteger(value);
    resolveTournament(event);
  }

  function tournamentChampion(event) {
    if (!event.rounds || !event.rounds.length) { return null; }
    var finalRound = event.rounds[event.rounds.length - 1];
    if (!finalRound.matches || !finalRound.matches.length) { return null; }
    return finalRound.matches[0].winnerId || null;
  }

  function serializeEvent(event) {
    return JSON.stringify(event, null, 2);
  }

  return {
    normalizeName: normalizeName,
    parseCourtSizes: parseCourtSizes,
    suggestCourtSizes: suggestCourtSizes,
    participantsById: participantsById,
    templateSupportedCount: templateSupportedCount,
    allRoundRobinMatches: allRoundRobinMatches,
    computeParticipantStandingsFromMatches: computeParticipantStandingsFromMatches,
    createRoundRobinEvent: createRoundRobinEvent,
    updateRoundRobinScore: updateRoundRobinScore,
    createLeagueEvent: createLeagueEvent,
    updateLeagueScore: updateLeagueScore,
    ladderRoundMatches: ladderRoundMatches,
    leagueRoundSummary: leagueRoundSummary,
    leagueAggregateStandings: leagueAggregateStandings,
    isLeagueRoundComplete: isLeagueRoundComplete,
    buildAutoMovement: buildAutoMovement,
    validateAssignments: validateAssignments,
    startNextLeagueRound: startNextLeagueRound,
    createTournamentEvent: createTournamentEvent,
    updateTournamentScore: updateTournamentScore,
    resolveTournament: resolveTournament,
    tournamentChampion: tournamentChampion,
    serializeEvent: serializeEvent
  };
});

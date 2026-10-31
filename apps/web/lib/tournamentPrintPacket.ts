import type { jsPDF } from "jspdf";
import type {
  AdminTournamentDayDraw as Draw,
  AdminTournamentDayGame as Game,
  AdminTournamentDayRoundRobinStanding as Standing,
  AdminTournamentDaySide as Side,
  AdminTournamentDayWorkspaceSnapshot as Snapshot,
} from "./adminTournamentDayOpsApi";

export type TournamentPrintScope = "cards" | "admin" | "packet";
export type TournamentPrintOptions = {
  drawIds: string[];
  unplayedOnly?: boolean;
  includeResults?: boolean;
  blankCards?: number;
  cardLayout?: "cut-stack" | "consecutive";
};
export type PrintTeam = { id: string; number: number; name: string; standing?: Standing };
export type PrintRound = { key: string; label: string; games: Game[]; byes: PrintTeam[] };
export type PrintDraw = { draw: Draw; teams: PrintTeam[]; games: Game[]; rounds: PrintRound[]; playoffs: Game[] };
export type PrintMatch = { game: Game | null; model?: PrintDraw };
type TallySection = { round: PrintRound; games: Game[]; start: number; bands: number };
export type PrintTallyPage = { teams: PrintTeam[]; sections: TallySection[]; part: number; parts: number; final: boolean; columns: number };

const compare = (a: string, b: string) => a.localeCompare(b, "en", { numeric: true });
const completed = (game: Game) => Boolean(game.finalized_at) || game.state === "COMPLETED";
const rr = (game: Game) => game.stage === "ROUND_ROBIN";
const clean = (value: string) => value.normalize("NFC").replace(/[\u2018\u2019]/g, "'")
  .replace(/[\u201c\u201d]/g, '"').replace(/[\u2010-\u2015]/g, "-")
  .replace(/\u2026/g, "...").replace(/\u00a0/g, " ");

export function printSideName(side: Side): string {
  const names = (side.participant_names || []).filter(Boolean).join(" / ");
  // Player names identify the people sent to court, even for a custom team name.
  return names || side.name || "To be determined";
}

export function printSource(source: Record<string, unknown> | null | undefined): string {
  if (!source) return "";
  if (source.winnerOf) return `Winner of ${source.winnerOf}`;
  if (source.loserOf) return `Loser of ${source.loserOf}`;
  if (source.seed != null) return `Seed ${source.seed}`;
  if (source.type === "WINNER" && source.game_code) return `Winner of ${source.game_code}`;
  if (source.type === "LOSER" && source.game_code) return `Loser of ${source.game_code}`;
  return "";
}

export function printMatchLabel(game: Game): string {
  if (game.playoff_game_code) {
    const round=game.playoff_round || game.round_label;
    const label: Record<string,string>={R16:"Round of 16",QF:"Quarterfinal",SF:"Semifinal",BRONZE:"Bronze medal",FINAL:"Final"};
    return [label[round.toUpperCase()] || round,game.playoff_game_code].filter(Boolean).join(" / ");
  }
  return [game.round_label, game.slot_label].filter(Boolean).join(" / ") || `Match ${game.id}`;
}

function playoffPhase(game: Game): number {
  const round = String(game.playoff_round || game.round_label).toUpperCase().replaceAll("_", " ");
  if (/^(R16|ROUND OF 16)$/.test(round)) return 0;
  if (/^(QF|QUARTERFINAL|QUARTERFINALS|PLAY.IN)$/.test(round)) return 1;
  if (/^(SF|SEMIFINAL|SEMIFINALS)$/.test(round)) return 2;
  if (/BRONZE|THIRD PLACE|3RD PLACE/.test(round)) return 3;
  if (/^(FINAL|FINALS|GOLD|GOLD MEDAL|CHAMPIONSHIP)$/.test(round)) return 4;
  return 2;
}

function sourceGameCode(source: Record<string, unknown> | null | undefined): string {
  if (!source) return "";
  return String(source.winnerOf || source.loserOf || (["WINNER", "LOSER"].includes(String(source.type)) ? source.game_code || "" : ""));
}

function playoffOrder(games: Game[]): Game[] {
  const remaining = new Set(games);
  const output: Game[] = [];
  while (remaining.size) {
    const available = [...remaining].filter(game => ![game.team_a_source, game.team_b_source].some(source => {
      const code = sourceGameCode(source);
      return code && [...remaining].some(parent => parent.playoff_game_code === code);
    })).sort((a,b) => playoffPhase(a)-playoffPhase(b) || compare(a.playoff_game_code || a.id,b.playoff_game_code || b.id));
    if (!available.length) throw new Error("The saved playoff bracket has circular match dependencies. Review it before printing.");
    const next = available[0];
    output.push(next); remaining.delete(next);
  }
  return output;
}

export function printScoring(game: Game): string {
  const s = game.scoring;
  if (s.blocker) return "Scoring needs confirmation - check with the desk before play.";
  if (s.format === "BEST_2_OF_3") {
    const target = s.individual_game_target;
    const rule = s.individual_game_win_by_two;
    return `Best 2 of 3${target ? ` to ${target}` : " (target: ______)"}${rule == null ? "; win by: ____" : rule ? ", win by 2" : ", win by 1"}. Game 3 only if tied 1-1.`;
  }
  return `One game${s.target ? ` to ${s.target}` : " (target: ______)"}${s.win_by_two == null ? "; win by: ____" : s.win_by_two ? ", win by 2" : ", win by 1"}.`;
}

/** Projects saved rows only. Never creates, rotates, or renumbers matchups. */
export function tournamentPrintDraws(snapshot: Snapshot, drawIds: string[]): PrintDraw[] {
  const requested = new Set(drawIds);
  if (!requested.size) throw new Error("Select at least one division or pool to print.");
  if (drawIds.some(id => !snapshot.draws.some(draw => draw.id === id))) {
    throw new Error("A selected draw is no longer in this tournament day. Reload and select it again.");
  }
  return snapshot.draws.filter(draw => requested.has(draw.id)).map(draw => {
    const games = snapshot.games.filter(game => game.draw_id === draw.id);
    const standings = draw.round_robin_summary?.standings || [];
    const teams: PrintTeam[] = standings.map(s => ({
      id: s.team_id, number: s.team_number || 0,
      name: s.participant_names.filter(Boolean).join(" / ") || s.team_name,
      standing: s,
    }));
    // Retain named teams even if older snapshot versions omit standings.
    for (const game of games) for (const side of [game.team_a, game.team_b]) {
      if (side.team_id && !teams.some(team => team.id === side.team_id)) {
        teams.push({ id: side.team_id, number: 0, name: printSideName(side) });
      }
    }
    const used = new Set(teams.map(team => team.number).filter(Boolean));
    let next = 1;
    for (const team of teams.filter(team => !team.number).sort((a,b) => compare(a.id,b.id))) {
      while (used.has(next)) next++;
      team.number = next; used.add(next);
    }
    teams.sort((a,b) => a.number - b.number || compare(a.id,b.id));
    const byRound = new Map<string, Game[]>();
    for (const game of games.filter(rr)) {
      const key = game.rr_round_number != null ? String(game.rr_round_number) : game.round_label;
      byRound.set(key, [...(byRound.get(key) || []), game]);
    }
    const rounds = [...byRound.entries()].sort(([a],[b]) => compare(a,b)).map(([key, rows]) => {
      const sorted = [...rows].sort((a,b) =>
        (a.rr_slot_number ?? Number.MAX_SAFE_INTEGER) - (b.rr_slot_number ?? Number.MAX_SAFE_INTEGER)
        || compare(a.slot_label || "", b.slot_label || "") || compare(a.id,b.id));
      const playing = new Set(sorted.flatMap(game => [game.team_a.team_id, game.team_b.team_id]));
      return { key, label: sorted[0].round_label || `Round ${key}`, games: sorted, byes: teams.filter(team => !playing.has(team.id)) };
    });
    const playoffs = playoffOrder(games.filter(game => !rr(game)));
    return { draw, teams, games: [...rounds.flatMap(round => round.games), ...playoffs], rounds, playoffs };
  });
}

/** Native pools share one draw and playoff bracket, but need separate tallies. */
export function tournamentPrintPools(model: PrintDraw): PrintDraw[] {
  const pools=model.draw.pool_config?.pools;
  if (!pools?.length) return [model];
  const memberships=new Map<string,string>();
  for (const pool of pools) for (const id of pool.team_ids) {
    if (memberships.has(id) || !model.teams.some(team=>team.id===id)) {
      throw new Error("Pool team assignments do not match the saved draw. Reload the day before printing.");
    }
    memberships.set(id,pool.id);
  }
  if (model.teams.some(team=>!memberships.has(team.id))) {
    throw new Error("A team is missing its pool assignment. Reload the day before printing.");
  }
  for (const game of model.games.filter(rr)) {
    const pool=memberships.get(game.team_a.team_id || "");
    if (!pool || pool!==memberships.get(game.team_b.team_id || "") || (game.pool_id && game.pool_id!==pool)) {
      throw new Error("A saved match does not belong to one pool. Review the pool schedule before printing.");
    }
  }
  return pools.map(pool=>{
    const teams=model.teams.filter(team=>memberships.get(team.id)===pool.id);
    const rounds=model.rounds.map(round=>{
      const games=round.games.filter(game=>memberships.get(game.team_a.team_id || "")===pool.id);
      const playing=new Set(games.flatMap(game=>[game.team_a.team_id,game.team_b.team_id]));
      return {...round,games,byes:teams.filter(team=>!playing.has(team.id))};
    }).filter(round=>round.games.length);
    const draw={...model.draw,name:`${model.draw.name} - ${pool.name}`};
    return {draw,teams,rounds,games:rounds.flatMap(round=>round.games),playoffs:[]};
  });
}

function cardDivision(game: Game, model?: PrintDraw): string {
  const division=model?.draw.name || game.draw_name;
  if (!rr(game)) return division;
  const pool=game.pool_name || model?.draw.pool_config?.pools.find(pool=>pool.team_ids.includes(game.team_a.team_id || "") && pool.team_ids.includes(game.team_b.team_id || ""))?.name;
  return pool ? `${division} - ${pool}` : division;
}

/** Current court/queue order is authoritative. Future court dispatch is dynamic;
 * the remaining cards follow saved rounds across the selected draws. */
export function tournamentPrintMatches(snapshot: Snapshot, models: PrintDraw[], options: TournamentPrintOptions): PrintMatch[] {
  const available = models.flatMap(model => model.games.map(game => ({game,model})));
  const emitted = new Set<string>();
  const entries: PrintMatch[] = [];
  const add = (id: string) => {
    const item = available.find(entry => entry.game.id === id);
    if (!item || emitted.has(id) || (options.unplayedOnly && completed(item.game))) return;
    emitted.add(id); entries.push(item);
  };
  const roundOrder = (game: Game) => game.rr_round_number ?? Number(game.round_label.match(/\d+/)?.[0] || 0);
  const planned = [...available].sort((a,b) => {
    if (rr(a.game) !== rr(b.game)) return rr(a.game) ? -1 : 1;
    const phase = rr(a.game) ? roundOrder(a.game)-roundOrder(b.game) : playoffPhase(a.game)-playoffPhase(b.game);
    return phase || models.indexOf(a.model)-models.indexOf(b.model) || a.model.games.indexOf(a.game)-b.model.games.indexOf(b.game);
  });
  // Include already completed cards first when the operator requested all games.
  for (const item of planned.filter(item => completed(item.game)).sort((a,b) => compare(a.game.finalized_at || "",b.game.finalized_at || ""))) add(item.game.id);
  for (const court of [...(snapshot.courts || [])].sort((a,b) => a.position-b.position)) {
    if (court.current_assignment) add(court.current_assignment.game_id);
  }
  const queue = [...(snapshot.eligible_queue || []),...(snapshot.reserved_queue || [])].sort((a,b) => a.position-b.position);
  for (const entry of queue) add(entry.game_id);
  // One pass per dependency level prevents a future final printing before its
  // own semifinal, even when old data has unusual round names or match codes.
  const remaining = planned.filter(item => !emitted.has(item.game.id) && (!options.unplayedOnly || !completed(item.game)));
  while (remaining.length) {
    const next = remaining.findIndex(item => rr(item.game) || ![item.game.team_a_source,item.game.team_b_source].some(source => {
      const code = sourceGameCode(source);
      return code && remaining.some(parent => parent.game.draw_id === item.game.draw_id && parent.game.playoff_game_code === code);
    }));
    if (next < 0) throw new Error("The saved playoff bracket has circular match dependencies. Review it before printing.");
    add(remaining.splice(next,1)[0].game.id);
  }
  for (let i=0; i<Math.min(20,Math.max(0,Math.floor(options.blankCards || 0))); i++) entries.push({game:null});
  return entries;
}

/** Top pile followed by bottom pile restores play order after a single cut. */
export function tournamentCardPages<T>(entries: T[], layout: TournamentPrintOptions["cardLayout"] = "cut-stack"): Array<[T | undefined,T | undefined]> {
  const pages = Math.ceil(entries.length/2);
  return Array.from({length:pages},(_,i) => layout === "consecutive" ? [entries[i*2],entries[i*2+1]] : [entries[i],entries[i+pages]]);
}

/** Bound both dimensions, including a round too large to fit on one page.
 * Repeat the saved schedule for each group of eight team columns. */
export function tournamentTallyPages(model: PrintDraw): PrintTallyPage[] {
  const columns=Math.min(8,Math.max(1,...model.rounds.map(round=>round.games.length)));
  const teamChunks=Array.from({length:Math.max(1,Math.ceil(model.teams.length/8))},(_,i)=>model.teams.slice(i*8,i*8+8));
  const schedulePages: TallySection[][]=[[]];
  let used=0;
  for (const round of model.rounds) {
    for (let start=0; start<round.games.length; start+=columns*7) {
      const games=round.games.slice(start,start+columns*7), bands=Math.ceil(games.length/columns);
      if (used+bands>7) { schedulePages.push([]); used=0; }
      schedulePages.at(-1)!.push({round,games,start,bands}); used+=bands;
    }
  }
  return schedulePages.flatMap((sections,i)=>teamChunks.map(teams=>({teams,sections,part:i+1,parts:schedulePages.length,final:i===schedulePages.length-1,columns})));
}

function sideText(game: Game, side: "a" | "b", model?: PrintDraw): string {
  const value = game[`team_${side}`];
  const team = model?.teams.find(candidate => candidate.id === value.team_id);
  if (value.team_id) return `${team ? `Team ${team.number} - ` : ""}${printSideName(value)}`;
  return `${printSource(game[`team_${side}_source`]) || "To be determined"}\n________________________________`;
}

function roundPoints(game: Game, side: "a" | "b"): string {
  if (!completed(game)) return "";
  if (game.result_type && game.result_type !== "PLAYED") {
    return ({ FORFEIT: "F", NO_SHOW: "NS", RETIREMENT: "RET" })[game.result_type];
  }
  if (game.scoring.format === "BEST_2_OF_3") {
    return game.game_scores?.length ? String(game.game_scores.reduce((sum,row) => sum + row[`score_${side}`], 0)) : "Check";
  }
  const value = game[`score_${side}`];
  return value == null ? "" : String(value);
}

class Paper {
  started = false;
  constructor(readonly pdf: jsPDF, readonly snapshot: Snapshot, readonly includeResults: boolean) {}
  page(orientation: "portrait" | "landscape") {
    if (this.started) this.pdf.addPage("letter", orientation);
    else {
      // The constructor creates a portrait page; replace it for admin-only PDFs.
      if (orientation === "landscape") { this.pdf.addPage("letter", orientation); this.pdf.deletePage(1); }
      this.started = true;
    }
    this.pdf.setTextColor(20); this.pdf.setDrawColor(90); this.pdf.setLineWidth(0.5);
  }
  text(value: string, x: number, y: number, size = 10, bold = false, align: "left" | "center" | "right" = "left") {
    this.pdf.setFont("helvetica", bold ? "bold" : "normal"); this.pdf.setFontSize(size);
    this.pdf.text(clean(value), x, y, { align });
  }
  box(value: string, x: number, y: number, w: number, h: number, size = 10, bold = false, align: "left" | "center" = "left", fill?: number) {
    if (fill != null) { this.pdf.setFillColor(fill,fill,fill); this.pdf.rect(x,y,w,h,"FD"); }
    else this.pdf.rect(x,y,w,h);
    const padding = w < 40 ? 2 : 5;
    if (value) this.fit(value,x+padding,y+4,w-2*padding,h-8,size,bold,align);
  }
  fit(value: string, x: number, y: number, w: number, h: number, size = 10, bold = false, align: "left" | "center" = "left") {
    this.pdf.setFont("helvetica", bold ? "bold" : "normal");
    let lines: string[] = [];
    // Fit complete names; no ellipses or clipping on paper match assignments.
    do {
      this.pdf.setFontSize(size);
      lines = this.pdf.splitTextToSize(clean(value), w);
      if (lines.length * size * 1.15 <= h) break;
      if (size === 6) break;
      size = Math.max(6,size-0.5);
    } while (true);
    if (lines.length * size * 1.15 > h) throw new Error("A name or label is too long for the paper layout. Shorten it before printing.");
    this.pdf.text(lines, align === "center" ? x+w/2 : x, y+size, { align, lineHeightFactor: 1.15 });
  }
  line(x1: number, y1: number, x2: number, y2: number) { this.pdf.line(x1,y1,x2,y2); }
  header(title: string, subtitle: string) {
    this.fit(this.snapshot.tournament.name,28,22,736,26,17,true);
    this.fit(title,28,52,736,25,12,true);
    this.fit(subtitle,28,79,736,22,8);
  }
  footer(label: string) {
    const w = this.pdf.internal.pageSize.getWidth(), h = this.pdf.internal.pageSize.getHeight();
    const saved = this.snapshot.generated_at ? `Snapshot: ${this.snapshot.generated_at.replace("T"," ").replace(/\.\d+Z$/, "Z")}` : "Snapshot time unavailable";
    this.text(`${label} | ${saved}`,28,h-17,7);
    this.text(`Page ${this.pdf.getNumberOfPages()}`,w-28,h-17,7,false,"right");
  }
}

function card(paper: Paper, y: number, game: Game | null, model?: PrintDraw) {
  const p = paper;
  p.fit(p.snapshot.tournament.name,32,y+32,548,30,18,true);
  p.fit(game ? cardDivision(game,model) : "Division: ____________________________________",32,y+72,548,29,14,true);
  p.fit(game ? printMatchLabel(game) : "Round: ____________________",32,y+112,360,22,12,true);
  // Cards are prepared ahead of play. The desk writes the court at dispatch,
  // even if this snapshot happens to contain a current or reserved assignment.
  p.fit("Court: __________",422,y+112,158,22,14,true,"center");
  p.fit(game ? printScoring(game) : "Scoring: ___________________________________________",32,y+144,548,30,10);
  const series = game?.scoring.format === "BEST_2_OF_3";
  const scoreColumns = series || !game ? 3 : 1;
  for (const [index, side] of (["a","b"] as const).entries()) {
    const x=32+index*302, width=246;
    const value=game?.[`team_${side}`];
    const names=value?.team_id ? (value.participant_names.filter(Boolean).join("\n") || printSideName(value)) : game ? printSource(game[`team_${side}_source`]) || "To be determined" : "________________________\n________________________";
    p.fit(names,x,y+186,width,54,15,true,"center");
    const scoreWidth=scoreColumns === 1 ? 92 : 72;
    const gap=8, total=scoreColumns*scoreWidth+(scoreColumns-1)*gap;
    const first=x+(width-total)/2;
    for (let col=0; col<scoreColumns; col++) {
      const sx=first+col*(scoreWidth+gap);
      p.text(scoreColumns === 1 ? "SCORE" : `Game ${col+1}`,sx+scoreWidth/2,y+258,10,true,"center");
      let score="";
      if (game && p.includeResults && completed(game)) {
        if (game.result_type && game.result_type !== "PLAYED") score=roundPoints(game,side);
        else if (series) {
          const row=game.game_scores?.find(row=>row.game_number===col+1);
          score=row ? String(row[`score_${side}`]) : "";
        } else score=game[`score_${side}`] == null ? "" : String(game[`score_${side}`]);
      }
      p.pdf.setLineWidth(1);
      p.box(score,sx,y+270,scoreWidth,76,28,true,"center");
      p.pdf.setLineWidth(0.5);
    }
  }
  p.text("vs",306,y+211,12,false,"center");
}

function cards(paper: Paper, models: PrintDraw[], options: TournamentPrintOptions) {
  const entries=tournamentPrintMatches(paper.snapshot,models,options);
  for (const page of tournamentCardPages(entries,options.cardLayout)) {
    paper.page("portrait");
    for (const [half,entry] of page.entries()) {
      if (!entry) continue;
      card(paper,half*396,entry.game,entry.model);
    }
    paper.pdf.setLineDashPattern([4,4],0); paper.line(18,396,594,396); paper.pdf.setLineDashPattern([],0);
  }
}

function tally(paper: Paper, model: PrintDraw) {
  const {draw, teams, rounds} = model;
  const hasByes = rounds.some(round => round.byes.length);
  const pages=tournamentTallyPages(model);
  for (const page of pages) {
    const teamChunk=page.teams, scheduleColumns=page.columns;
    const wrapped=rounds.some(round=>round.games.length>scheduleColumns);
    paper.page("landscape");
    const continuation=pages.length>1 ? ` | Teams ${teamChunk.map(team=>team.number).join(", ")} | Schedule ${page.part} of ${page.parts}` : "";
    paper.header(`${draw.name} - round-robin admin tally`, `${paper.snapshot.day_scope.selected_day.event_date || ""}${continuation} | ${paper.includeResults ? "Saved results; update totals and seeds as play continues" : "Scores and totals"}`);
    const x=28, top=108, full=736;
    const roundW=45, gameW = scheduleColumns > 4 ? 29 : 36, byeW=hasByes ? 39 : 0;
    const scheduleW=roundW+scheduleColumns*gameW+byeW;
    const teamW=(full-scheduleW)/Math.max(1,teamChunk.length);
    const nameH=76, labelH=18, headH=18, headerH=nameH+labelH+headH;
    paper.fit("SAVED MATCH SCHEDULE\nMatchups are team numbers.\nCards use the same round / slot labels.",x,top+28,scheduleW-8,58,9,true);
    for (const [ti,team] of teamChunk.entries()) {
      const tx=x+scheduleW+ti*teamW;
      paper.box(`Team ${team.number}`,tx,top,teamW,labelH,9,true,"center",240);
      paper.box(team.name,tx,top+labelH,teamW,nameH,9,false,"center");
      paper.box("For",tx,top+labelH+nameH,teamW/2,headH,8,false,"center",240);
      paper.box("Against",tx+teamW/2,top+labelH+nameH,teamW/2,headH,8,false,"center",240);
    }
    const hy=top+headerH-headH;
    paper.box("Round",x,hy,roundW,headH,8,true,"center",240);
    // Cells retain each game's actual slot label even when slots are sparse.
    if (wrapped) paper.box("Matches in saved slot order",x+roundW,hy,scheduleColumns*gameW,headH,8,true,"center",240);
    else for (let gi=0; gi<scheduleColumns; gi++) paper.box(`Game ${gi+1}`,x+roundW+gi*gameW,hy,gameW,headH,7,true,"center",240);
    if (hasByes) paper.box("Bye",x+roundW+scheduleColumns*gameW,hy,byeW,headH,8,true,"center",240);
    let y=top+headerH;
    for (const section of page.sections) {
      const {round,games,bands,start}=section, rowH=32*bands;
      paper.box(`${round.label}${start ? "\n(cont.)" : ""}`,x,y,roundW,rowH,8,true,"center");
      for (let gi=0; gi<bands*scheduleColumns; gi++) {
        const game=games[gi];
        const a=teams.find(team=>team.id === game?.team_a.team_id)?.number;
        const b=teams.find(team=>team.id === game?.team_b.team_id)?.number;
        const slot = game?.rr_slot_number;
        const slotLabel=game?.slot_label || (slot != null ? `Slot ${slot}` : `Game ${start+gi+1}`);
        const ordinaryLabel=slotLabel === `Slot ${gi+1}` || slotLabel === `Game ${gi+1}`;
        const label=game ? `${a ?? "?"} v ${b ?? "?"}${wrapped || start || !ordinaryLabel || (slot != null && slot !== gi+1) ? `\n${slotLabel}` : ""}` : "";
        paper.box(label,x+roundW+(gi%scheduleColumns)*gameW,y+Math.floor(gi/scheduleColumns)*32,gameW,32,9,false,"center");
      }
      if (hasByes) paper.box(round.byes.length>4 ? "Shaded\ncells" : round.byes.map(team=>team.number).join(", "),x+roundW+scheduleColumns*gameW,y,byeW,rowH,8,false,"center");
      for (const [ti,team] of teamChunk.entries()) {
        const tx=x+scheduleW+ti*teamW;
        const matches=games.filter(game=>[game.team_a.team_id,game.team_b.team_id].includes(team.id));
        const isBye=round.byes.some(bye=>bye.id === team.id);
        for (const [pi,own] of [true,false].entries()) {
          const scores=paper.includeResults ? matches.map(game=>roundPoints(game, (game.team_a.team_id === team.id) === own ? "a" : "b")).filter(Boolean).join(" / ") : "";
          paper.box(matches.length ? scores : isBye ? "-" : "Other\nsheet",tx+pi*teamW/2,y,teamW/2,rowH,matches.length || isBye ? 10 : 7,false,"center",isBye ? 218 : undefined);
        }
      }
      y+=rowH;
    }
    if (!rounds.length) {
      paper.fit("No round-robin games have been generated. Generate the saved schedule before printing match cards.",x,y+10,full,35,11,true);
      y+=50;
    }
    if (page.final) {
      for (const label of ["Wins / Losses","Total points","Differential +/-","Seed / place"]) {
        paper.box(label,x,y,scheduleW,24,9,true,"left");
        for (const [ti,team] of teamChunk.entries()) {
          const tx=x+scheduleW+ti*teamW, s=paper.includeResults ? team.standing : undefined;
          if (label === "Total points") {
            paper.box(s ? String(s.points_for) : "",tx,y,teamW/2,24,10,false,"center");
            paper.box(s ? String(s.points_against) : "",tx+teamW/2,y,teamW/2,24,10,false,"center");
          } else {
            const value=!s ? "" : label === "Wins / Losses" ? `${s.wins} / ${s.losses}` : label === "Differential +/-" ? String(s.differential) : `${s.seed}${s.retired ? " (retired)" : ""}`;
            paper.box(value,tx,y,teamW,24,10,false,"center");
          }
        }
        y+=24;
      }
    } else {
      paper.text("More rounds follow. Overall totals are on the final sheet for these teams.",x,y+18,8);
      y+=28;
    }
    const policy = draw.round_robin_summary?.ranking_policy;
    const description = typeof policy === "string" ? policy : policy?.description;
    const note = paper.includeResults ? "F = forfeit; NS = no-show; RET = retirement. Round cells show played points; saved totals include official adjustments. Recheck ties after updates." : "Enter each score in both teams' For / Against cells. Shaded cells are byes; they do not count as wins or losses.";
    paper.fit(`${note}${description ? `\n${description}` : ""}`,x,Math.min(y+10,553),full,37,7.5);
    paper.footer("ADMIN TALLY");
  }
}

function playoffSheet(paper: Paper, model: PrintDraw) {
  if (!model.playoffs.length) return;
  for (let start=0; start<model.playoffs.length; start+=4) {
    paper.page("landscape");
    paper.header(`${model.draw.name} - playoff bracket worksheet`, "Follow the saved seed / winner / loser sources. Write advancing teams beside each match; keep returned scorecards.");
    const widths=[58,227,227,110,114], headers=["Match","Team A / source","Team B / source","Scores A - B","Winner / advance"];
    let x=28;
    for (let i=0;i<widths.length;i++) { paper.box(headers[i],x,108,widths[i],27,9,true,"center",240); x+=widths[i]; }
    let y=135;
    for (const game of model.playoffs.slice(start,start+4)) {
      const values=[
        `${printMatchLabel(game)}\n${game.playoff_round || game.round_label}`,
        `${sideText(game,"a",model)}${game.team_a.team_id && printSource(game.team_a_source) ? `\n(${printSource(game.team_a_source)})` : ""}`,
        `${sideText(game,"b",model)}${game.team_b.team_id && printSource(game.team_b_source) ? `\n(${printSource(game.team_b_source)})` : ""}`,
        paper.includeResults && completed(game) ? (game.game_scores?.length ? game.game_scores.map(row=>`G${row.game_number}: ${row.score_a} - ${row.score_b}`).join("\n") : `${game.score_a ?? ""} - ${game.score_b ?? ""}`) : "G1: ____ - ____\n\nG2: ____ - ____\n\nG3: ____ - ____",
        paper.includeResults && completed(game) ? game.winner_name || "" : "________________\n\nEntered: ________",
      ];
      x=28;
      for (let i=0;i<widths.length;i++) { paper.box(values[i],x,y,widths[i],90,10,i===0); x+=widths[i]; }
      y+=90;
    }
    paper.fit("Single-game matches use G1 only. For best 2 of 3, stop when a team wins two games. Non-play outcome / notes: ______________________________",28,512,736,28,9);
    paper.text("1st: ______________________________   2nd: ______________________________   3rd: ______________________________",28,558,10);
    paper.footer("PLAYOFF WORKSHEET");
  }
}

export async function buildTournamentPrintPdf(snapshot: Snapshot, options: TournamentPrintOptions, scope: TournamentPrintScope) {
  const models=tournamentPrintDraws(snapshot,options.drawIds);
  // Validate pool membership for cards too, before creating a misleading PDF.
  const poolsByDraw=models.map(model=>tournamentPrintPools(model));
  const {jsPDF}=await import("jspdf");
  const pdf=new jsPDF({orientation:"portrait",unit:"pt",format:"letter",compress:true});
  const paper=new Paper(pdf,snapshot,Boolean(options.includeResults));
  pdf.setProperties({title:`${snapshot.tournament.name} - ${scope === "cards" ? "Match scorecards" : scope === "admin" ? "Admin tally sheets" : "Paper tournament packet"}`,creator:"Pickleball Club Sandwich"});
  if (scope !== "admin") cards(paper,models,options);
  if (scope !== "cards") for (const [index,model] of models.entries()) {
    for (const pool of poolsByDraw[index]) tally(paper,pool);
    playoffSheet(paper,model);
  }
  if (!paper.started) throw new Error("No matches match your selection. Include completed matches or add spare blank cards.");
  const safe=(value:string)=>clean(value).replace(/[^a-zA-Z0-9]+/g,"-").replace(/^-|-$/g,"").slice(0,70) || "tournament";
  const day=snapshot.day_scope.selected_day;
  const filename=`${safe(snapshot.tournament.name)}-${safe(day.event_date || day.label)}-${scope}.pdf`;
  return {pdf,filename};
}

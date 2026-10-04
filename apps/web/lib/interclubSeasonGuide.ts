import type { AdminEventHistory } from "./eventSeasons";
import type { SeasonAwardPreview } from "./interclubAwards";
import type { CompetitionBatch, CompetitionContext } from "./interclubCompetition";
import type { RegistrationDetail } from "./interclubRegistration";

export const seasonStages = [
  { title: "Set up the season", link: "Setup & invitations", description: "Name the season, choose clubs and divisions, and plan the meet dates." },
  { title: "Register clubs & players", link: "Registration & player pool", description: "Clubs accept their invitations. Set registration dates and build each club’s season player pool." },
  { title: "Play regular meets", link: "Regular meets", description: "Choose available players, approve lineups, enter scores, and approve each meet’s results." },
  { title: "Run championships", link: "Championships", description: "Review who qualified, resolve any qualifying playoff, then schedule and play the finals." },
  { title: "Publish results & awards", link: "Results & awards", description: "Review the champions and recipients, then publish final results and award the trophies." },
  { title: "Start the next season", link: "History & seasons", description: "Keep this season in History and create a linked draft for the next one." },
] as const;

export function meetFinished(batch: CompetitionBatch | undefined): boolean {
  return !!batch && batch.state === "approved" && batch.approved_revision === batch.revision && batch.ratings_status === "completed" && batch.document.weather !== "rescheduled";
}

export function meetProgress(batch: CompetitionBatch | undefined): string {
  if (!batch) return "Choose players and prepare lineups";
  if (batch.document.weather === "rescheduled") return "Reschedule this meet";
  if (batch.state === "draft") return "Enter scores or finish score corrections";
  if (batch.state === "submitted") return "Awaiting organizer approval";
  if (batch.approved_revision !== batch.revision) return "Review the latest score changes";
  if (batch.ratings_status === "failed") return "Retry rating updates";
  if (batch.ratings_status !== "completed") return "Rating updates still need to finish";
  return "Results approved · ratings complete";
}

export type SeasonGuideData = { registration: RegistrationDetail; competition: CompetitionContext | null; awards: SeasonAwardPreview | null; history: AdminEventHistory | null };
export type GuideAction = { label: string; href: string };

export function buildSeasonGuide({ registration, competition, awards, history }: SeasonGuideData) {
  const { season, is_organizer: organizer } = registration;
  const query = `season=${encodeURIComponent(season.id)}`;
  const registrationHref = `/admin/interclub/registrations?${query}`;
  const competitionHref = `/admin/interclub/competition?${query}`;
  const awardsHref = `/admin/interclub/awards?${query}`;
  const historyHref = organizer ? `/admin/event-history?${new URLSearchParams({ kind: "interclub", event: season.id })}` : `/interclub/${encodeURIComponent(season.id)}?view=history`;
  const hrefs = [`/admin/interclub?${query}`, registrationHref, competitionHref, competitionHref, organizer ? awardsHref : `/interclub/${encodeURIComponent(season.id)}/final-results`, historyHref];
  const meets = [...(competition?.meets || [])].sort((a, b) => a.starts_at.localeCompare(b.starts_at) || a.id.localeCompare(b.id));
  const batches = new Map((competition?.batches || []).map(batch => [batch.meet_id, batch]));
  const regular = meets.filter(meet => !meet.competition_phase || meet.competition_phase === "regular");
  const finals = meets.filter(meet => meet.competition_phase && meet.competition_phase !== "regular");
  if (regular[0]) hrefs[2] = `${competitionHref}&meet=${encodeURIComponent(regular[0].id)}`;
  const regularDone = regular.length > 0 && regular.every(meet => meetFinished(batches.get(meet.id)));
  const competitionDone = regularDone && meets.every(meet => meetFinished(batches.get(meet.id))) && competition?.club_cup.status === "complete";
  const currentEdition = history?.seasons.find(item => item.source_id === season.id);
  const successor = currentEdition ? history?.seasons.filter(item => item.position > currentEdition.position).sort((a, b) => a.position - b.position)[0] : undefined;
  const linked = !!successor;
  let stage = 1;
  let title = "Open season registration";
  let why = "Setup is saved. Now give every club the same window to register its players.";
  let instructions = ["Check the club invitation responses.", "Set the registration opening and closing dates.", "Clubs build their player pools. Meet planning opens when registration closes."];
  let action: GuideAction = { label: "Set registration dates", href: `${registrationHref}#season-registration-heading` };
  const phase = season.registration?.status || "unconfigured";
  if (!organizer && registration.own_participation?.status !== "accepted") {
    title = "Respond to your club’s invitation";
    why = "Your club must accept before it can register players or prepare a meet.";
    instructions = ["Review the season dates and divisions.", "Accept or decline the invitation for your club."];
    action = { label: "Review club invitation", href: `${registrationHref}#invitation-title` };
  } else if (phase !== "closed" || !season.registration?.meet_planning_open) {
    if (phase !== "unconfigured" || !organizer) {
      title = phase === "open" ? "Register the season’s players" : phase === "scheduled" ? "Prepare for registration" : "Registration dates are needed";
      why = phase === "open" ? "Registration is open. This is the player pool for the whole season; lineups are chosen separately for each meet." : phase === "scheduled" ? "Registration dates are set. Clubs can review the schedule before player registration opens." : "The season organizer needs to set the registration window before players can sign up.";
      instructions = ["Review your club’s invitation and registration dates.", "Share player signup when registration opens and review the player pool.", "After registration closes, return here to prepare the first meet."];
      action = { label: "Open registration & player pool", href: registrationHref };
    }
  } else {
    stage = (organizer ? regularDone : finals.length > 0 && regular.every(meet => meetFinished(batches.get(meet.id)))) ? 3 : 2;
    const outstanding = (stage === 2 ? regular : finals).find(meet => !meetFinished(batches.get(meet.id)));
    title = stage === 2 ? "Run the regular season" : "Set up and play the championships";
    why = stage === 2 ? `${regular.filter(meet => meetFinished(batches.get(meet.id))).length} of ${regular.length} ${organizer ? "regular-season" : "your club’s regular-season"} meets have approved results and completed ratings.` : "Regular-season results are complete. The championship screen shows the qualified clubs and any playoff needed before the finals.";
    instructions = stage === 2 ? ["Choose a meet and confirm the available players and lineups.", "Prepare the court schedule, then enter and submit the scores.", "The organizer approves the results. Repeat for each regular meet."] : ["Review the qualified clubs for each division and resolve any qualifying playoff.", "Schedule the championship meet and fill its lineups.", "Play the finals, submit scores, and approve the results."];
    action = { label: stage === 2 ? "Schedule a regular meet" : "Open championship setup", href: competitionHref };
    if (outstanding) {
      const batch = batches.get(outstanding.id);
      const step = !batch ? "availability" : batch.state === "draft" && batch.document.weather !== "rescheduled" ? "run" : "approve";
      action = { label: !batch ? "Prepare next meet" : organizer && batch.state === "submitted" ? "Review & approve meet results" : meetProgress(batch), href: `/admin/interclub/${step === "availability" ? "registrations" : "competition"}?${new URLSearchParams({ season: season.id, meet: outstanding.id, step })}` };
      why += ` Next: ${meetProgress(batch).toLowerCase()}.`;
      if (batch?.state === "approved" && batch.ratings_status !== "completed") {
        title = "Finish this meet’s rating updates";
        instructions = organizer ? ["Open the meet’s results review.", "Choose “Retry rating updates” if the update is pending or failed.", "Return to this guide after the rating updates complete."] : ["The organizer needs to finish this meet’s rating updates.", "You can review the approved scores while waiting."];
      } else if (batch?.state === "submitted") {
        title = organizer ? "Review the submitted meet results" : "Waiting for the organizer’s approval";
        instructions = organizer ? ["Open the meet and review the submitted scores and substitutions.", "Choose “Review official approval”, then confirm the reviewed revision.", "Return to this guide for the next meet or championship step."] : ["Review the scores submitted for this meet.", "The organizer approves the results before the season moves on."];
      }
    } else if (!organizer) {
      title = meets.length ? "Your club’s scheduled meets are complete" : "Waiting for your club’s meet schedule";
      why = "The season organizer manages the remaining meets, championship qualification, and season awards.";
      instructions = ["Review your club’s submitted results.", "Check the public results for championship and season updates."];
      action = { label: "Review your club’s meets", href: competitionHref };
    }
    if (organizer && competitionDone) {
      stage = 4;
      title = "Finish the season: publish results & awards";
      why = "All scheduled meets are approved and the championships are decided. Review and award the season trophies to complete the season.";
      instructions = ["Review final standings and the player and club trophy recipients.", "Confirm your review, then choose “Publish final results and award trophies”.", "Return to this guide to start a linked new season."];
      action = { label: awards?.revision ? "Review final results & update awards" : "Review final results & award trophies", href: awardsHref };
      if (awards && !awards.ready) { why = "The final review still has items to resolve before the season can be finished."; instructions = awards.problems; }
    }
  }
  const complete = organizer ? !!(competitionDone && awards?.ready && awards.current) : registration.season_complete === true;
  if (complete) {
    stage = 5;
    title = linked ? "This season is finished. Continue the next one." : "Season finished — ready for the next one";
    why = organizer ? "Final results and season trophies are published. This season stays in History when you start the next one." : "Final results are published. The organizer handles awards and the next season.";
    instructions = organizer ? linked ? ["Open the next season already linked to this one.", "Complete its setup and review its new meet dates before opening invitations."] : ["Choose “Start new season” in History & seasons.", "Give the new season a label, name, and dates, then review and create its draft.", "Continue the new draft’s setup. Clubs and rules carry forward; meet dates and player signups start fresh."] : ["View the final standings and honors.", "Use History to revisit earlier seasons."];
    action = organizer ? successor?.admin_href ? { label: `Continue ${successor.label}`, href: successor.admin_href } : { label: "Start next season", href: historyHref } : { label: "View final results & awards", href: hrefs[4] };
    if (organizer && history && !history.can_start && !successor) {
      why = history.reason || "Review the season history before starting another season.";
      action = { label: "Review season history", href: historyHref };
    }
  }
  return { stage, title, why, instructions, action, hrefs, meets, batches, complete, linked, organizer, regularDone, competitionDone };
}

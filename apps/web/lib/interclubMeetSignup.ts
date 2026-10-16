import type { SignupClub, SignupMeet, SignupSeason } from "./interclubPlayerSignup";

export type SignupGender = "female" | "male" | "non_binary" | "prefer_not_to_say";
export const signupGenderLabels: Record<SignupGender, string> = {
  female: "Woman", male: "Man", non_binary: "Non-binary", prefer_not_to_say: "Prefer not to say",
};
export function signupGender(value?: string | null): SignupGender | "" {
  const gender = (value || "").trim().toLowerCase().replace(/[ -]/g, "_");
  if (["female", "f", "woman", "women"].includes(gender)) return "female";
  if (["male", "m", "man", "men"].includes(gender)) return "male";
  if (["non_binary", "nonbinary"].includes(gender)) return "non_binary";
  return gender === "prefer_not_to_say" ? gender : "";
}
export function needsGenderReview(gender: string): boolean {
  return gender === "non_binary" || gender === "prefer_not_to_say";
}

export type MeetSignupEntry = {
  id: string; name: string; division: string; gender: string; rating: number | null;
  status: "active" | "withdrawn"; placement: "confirmed" | "waitlist" | "review" | "withdrawn";
  priority: "in_band" | "play_up" | "review"; reason: string; registered_at: string; queue_position?: number;
  revision?: number; player_id?: string; email?: string; manage_url?: string;
  declared_gender?: SignupGender | null; reviewed_gender?: "female" | "male" | null;
};
export type MeetSignupBoard = {
  club: SignupClub; season: SignupSeason;
  meet: SignupMeet & { revision: number; roster_deadline: string; competition_phase: string };
  signup: { open: boolean; configured: boolean; revision: number; deadline: string; schedule_changed: boolean; url: string | null };
  entries: MeetSignupEntry[];
};
export type PrivateMeetSignup = MeetSignupBoard & { entry: MeetSignupEntry };
export function meetSignupOpen(board: MeetSignupBoard, now = Date.now()): boolean {
  return board.signup.open && now < Math.min(Date.parse(board.signup.deadline), Date.parse(board.meet.roster_deadline), Date.parse(board.meet.starts_at));
}
export function signupPlacement(entry: MeetSignupEntry): string {
  if (entry.status === "withdrawn") return "Withdrawn";
  return entry.placement === "confirmed" ? "Spot reserved" : entry.placement === "review" ? "Waiting for admin review" : `Substitute #${entry.queue_position ?? "—"}${entry.priority === "play_up" ? " · Playing up" : ""}`;
}
export function divisionPriorityHint(division: string, rating: number | null): string {
  const lower = /open/i.test(division) ? 4.5 : Number(division);
  if (rating == null) return "Your club needs to confirm your league rating.";
  return rating < lower ? `Your ${rating.toFixed(2)} league rating is below ${lower.toFixed(1)}. You’ll join the play-up waitlist; players in this rating band have priority.`
    : "The first two women and first two men in this rating band reserve spots. Later signups join the substitute queue.";
}

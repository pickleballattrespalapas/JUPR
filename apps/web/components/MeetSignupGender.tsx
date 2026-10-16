"use client";
import { needsGenderReview, SignupGender, signupGenderLabels } from "@/lib/interclubMeetSignup";

export default function MeetSignupGender({ value, onChange, disabled }: {
  value: SignupGender | ""; onChange: (value: SignupGender | "") => void; disabled?: boolean;
}) {
  return <div><label>Gender<select aria-label="Gender" required disabled={disabled} value={value} onChange={event => onChange(event.target.value as SignupGender | "")}>
    <option value="">Choose gender</option>
    {Object.entries(signupGenderLabels).map(([gender, label]) => <option key={gender} value={gender}>{label}</option>)}
  </select></label>
    {needsGenderReview(value) && <p role="status">You can still sign up. An admin will review your registration and confirm your lineup placement.</p>}
  </div>;
}

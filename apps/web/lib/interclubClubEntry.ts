export type ClubEntryContactMethod = "email" | "whatsapp" | "phone";
export type ClubEntryParticipation = "host_travel" | "host" | "travel" | "undecided";

export type ClubEntryOffer = {
  date: string;
  start_time: string | null;
  end_time: string | null;
  courts: number | null;
};

export type ClubEntryRegistration = {
  captain: string;
  contact_method: ClubEntryContactMethod;
  contact: string;
  divisions: string[];
  participation: ClubEntryParticipation;
  travel_notes: string;
  host_offers: ClubEntryOffer[];
};

export type ClubEntryContext = {
  season_name: string;
  club_name: string;
  divisions: string[];
  start_date: string | null;
  end_date: string | null;
  revision: number;
  registration: ClubEntryRegistration | null;
};

import ClubEntry from "./ClubEntry";

export const metadata = {
  title: "Interclub club entry · Registro de clubes",
  robots: { index: false, follow: false },
  referrer: "no-referrer" as const,
};

export default function ClubEntryPage() {
  return <ClubEntry />;
}

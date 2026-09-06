import Link from "next/link";

export default function InterclubPlanningPage() {
  return <section style={{ maxWidth: 900, margin: "0 auto" }}>
    <p>Southern BCS</p><h1>Interclub league planning</h1>
    <p>The first season will bring clubs from La Paz and south together from January through March.</p>
    <dl>
      <dt>Clubs and meets</dt><dd>Eight clubs, four clubs per meet, with two hosts able to run meets at the same time.</dd>
      <dt>Divisions</dt><dd>3.5 and 4.0, with an optional 4.5/Open division.</dd>
      <dt>Teams and play</dt><dd>Four-player teams. Three games to 11, win by two, no cap. Meets last no more than three hours.</dd>
      <dt>Club responsibilities</dt><dd>Each club administrator submits and updates their own team rosters. Hosts run their assigned meets under the league rules.</dd>
      <dt>Ratings</dt><dd>The organizer approves each meet before it updates league ratings and players’ ratings at the clubs they represent.</dd>
    </dl>
    <h2>Coming next</h2><p>Season setup, participating clubs, team rosters, and meet scheduling. This page is a planning overview; season registration and match operations are not open yet.</p>
    <p><Link href="/admin/staff">Manage club staff</Link> · <Link href="/admin">Club operations</Link></p>
  </section>;
}

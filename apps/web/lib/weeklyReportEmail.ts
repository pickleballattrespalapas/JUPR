import type { WeeklyRecapJson } from "./weeklyRecapApi";
import type { RecapItem } from "./recapBuilder";

const font = "font-family:Arial,Helvetica,sans-serif;";
const body = `${font}font-size:16px;line-height:26px;color:#263e3a;`;
const h = `${font}font-size:21px;line-height:28px;color:#123e37;margin:0 0 8px;`;
const esc = (value: unknown) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
export function emailUrl(value: string | undefined): string {
  if (!value || /\s/.test(value)) return "";
  try {
    const u = new URL(value);
    return ["https:", "http:"].includes(u.protocol) &&
      !u.username &&
      !u.password
      ? u.href
      : "";
  } catch {
    return "";
  }
}
export function reportLink(
  origin: string,
  clubSlug: string,
  path: string,
): string {
  const base = emailUrl(origin);
  const target = reportPath(clubSlug, path);
  return base && target ? emailUrl(`${new URL(base).origin}${target}`) : "";
}
export function reportPath(clubSlug: string, path: string | undefined): string {
  if (!clubSlug || !path || /\s|\.\.|\\|:|#/.test(path)) return "";
  const segment = "[A-Za-z0-9_.%~-]+";
  const routes = [
    `leagues/${segment}/standings`,
    `leagues/${segment}/weekly-history(?:\\?week=[1-9][0-9]*)?`,
    `round-robin-generator/shared/${segment}/standings`,
    `tournament-team-results/${segment}/${segment}`,
    `tournament-results\\?tournament=${segment}`,
    `weekly-recap\\?week=[0-9]{4}-[0-9]{2}-[0-9]{2}`,
  ];
  if (new RegExp(`^/interclub/${segment}$`).test(path)) return path;
  return routes.some((pattern) => new RegExp(`^${pattern}$`).test(path))
    ? `/clubs/${encodeURIComponent(clubSlug)}/${path}`
    : "";
}
function p(value: unknown) {
  return `<p style="${body}margin:0 0 12px;">${esc(value).replace(/\n/g, "<br>")}</p>`;
}
function list(values: string[]) {
  return `<ul style="${body}margin:0 0 12px;padding-left:22px;">${values.map((v) => `<li style="margin:0 0 5px;">${esc(v)}</li>`).join("")}</ul>`;
}
function section(label: string, content: string, sand = false) {
  return `<tr data-recap-block><td bgcolor="${sand ? "#f8f3e9" : "#ffffff"}" style="padding:24px;border-top:1px solid #e6e8e1;"><h2 style="${font}font-size:12px;line-height:18px;font-weight:bold;letter-spacing:1.7px;color:#17665c;margin:0 0 14px;">${esc(label)}</h2>${content}</td></tr>`;
}
function button(label: string, url: string) {
  return url
    ? `<table role="presentation" border="0" cellspacing="0" cellpadding="0"><tr><td bgcolor="#17665c" style="border-radius:4px;mso-padding-alt:13px 16px;"><a href="${esc(url)}" style="${font}display:inline-block;padding:13px 16px;font-size:15px;line-height:21px;font-weight:bold;text-decoration:none;color:#ffffff;">${esc(label)}</a></td></tr></table>`
    : "";
}

export type WeeklyEmailOptions = {
  compact?: boolean;
  document?: boolean;
  origin: string;
  clubSlug: string;
  published?: boolean;
};
function story(item: RecapItem, options: WeeklyEmailOptions) {
  const image =
    emailUrl(item.image_url) ||
    (/^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(
      item.image_url || "",
    )
      ? item.image_url
      : "");
  const styled = item.layout && item.layout !== "plain";
  const decoration = styled
    ? `padding:18px;border:1px solid #e6e8e1;border-radius:10px;background:${item.layout === "callout" ? "#f8f3e9" : "#ffffff"};${item.layout === "callout" ? "border-left:4px solid #cb794f;" : item.layout === "promotion" ? "border-top:4px solid #17665c;" : ""}`
    : "";
  const width = styled ? 532 : 572;
  const caption = [item.caption, item.credit ? `Photo: ${item.credit}` : ""]
    .filter(Boolean)
    .join(" · ");
  const photo = image
    ? `<img src="${esc(image)}" alt="${esc(item.alt || item.title)}" width="${width}" style="display:block;width:100%;max-width:${width}px;${item.image_fit === "cover" ? `height:${Math.round((width * 9) / 16)}px;object-fit:cover;` : "height:auto;"}border-radius:8px;margin:0 0 8px;" />${caption ? `<p style="${font}font-size:13px;line-height:19px;color:#526762;margin:0 0 12px;">${esc(caption)}</p>` : ""}`
    : "";
  const url = emailUrl(item.url);
  return `<div data-recap-item="${esc(item.id)}" data-card-style="${esc(item.layout || "plain")}" style="${decoration}margin:0 0 16px;overflow-wrap:anywhere;">${styled && item.kind ? `<p style="${font}font-size:12px;line-height:18px;color:#17665c;margin:0 0 6px;">${esc(item.kind)}</p>` : ""}${item.title ? `<h3 style="${h}">${esc(item.title)}</h3>` : ""}${photo}${item.text ? p(item.text) : ""}${item.lines?.length ? list(item.lines) : ""}${options.published && reportPath(options.clubSlug, item.href) ? pLink("View results", reportLink(options.origin, options.clubSlug, item.href!)) : ""}${options.published ? (item.links || []).map((link) => pLink(link.label, reportLink(options.origin, options.clubSlug, link.href))).join("") : ""}${url ? (item.layout === "promotion" ? button(item.link_label || "Learn more", url) : pLink(item.link_label || "Learn more", url)) : ""}</div>`;
}
/** Self-contained table fragment fits inside the Tres 620px newsletter. No footer/signoff duplication. */
export function buildWeeklyReportEmail(
  recap: Partial<WeeklyRecapJson>,
  options: WeeklyEmailOptions,
): string {
  const report = recap.report || {};
  const compact = options.compact ?? true;
  const published = Boolean(options.published);
  const week = recap.week_start || recap.start_date || "";
  const end = recap.week_end || recap.end_date || "";
  const title = report.headline || "This week on the courts";
  const fullUrl = published
    ? reportLink(
        options.origin,
        options.clubSlug,
        `weekly-recap?week=${encodeURIComponent(week)}`,
      )
    : "";
  let rows = `<tr><td bgcolor="#f8f3e9" style="padding:24px;border-top:4px solid #cb794f;">${!published ? p("DRAFT PREVIEW · Review before sharing") : ""}<h2 style="${h}font-size:26px;line-height:32px;">${esc(title)}</h2><p style="${font}font-size:14px;line-height:21px;color:#17665c;margin:0;">${esc(week)} – ${esc(end)}</p></td></tr>`;
  if (report.scope_label && report.scope_label !== "Whole club")
    rows += section("LEAGUE", p(report.scope_label));
  if (report.edition) {
    const blocks = compact
      ? report.email_edition || report.edition
      : report.edition;
    for (const block of blocks) {
      const content = block.items.map((item) => story(item, options)).join("");
      rows += section(
        block.title,
        content,
        ["spotlight", "notes", "custom"].includes(block.key),
      );
    }
    if (fullUrl)
      rows += section(
        "KEEP UP WITH THE CLUB",
        button("View the full weekly report", fullUrl),
      );
    return wrapEmail(rows, title, options.document);
  }
  if (report.opening) rows += section("FROM THE COURTS", p(report.opening));
  const n = recap.numbers || {};
  rows += section(
    "THE WEEK IN NUMBERS",
    `<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr>${[
      [n.matches ?? 0, "Games"],
      [n.players ?? 0, "Players"],
      [n.new_faces ?? 0, "New faces"],
    ]
      .map(
        ([v, label]) =>
          `<td width="33%" valign="top" style="${font}text-align:center;color:#123e37;padding:8px 3px;"><strong style="font-size:26px;line-height:32px;">${esc(v)}</strong><br><span style="font-size:13px;line-height:20px;">${label}</span></td>`,
      )
      .join("")}</tr></table>`,
  );
  const events = (report.event_results || []).slice(0, compact ? 3 : 100);
  if (events.length)
    rows += section(
      "EVENT RESULTS",
      events
        .map(
          (e) =>
            `<h3 style="${h}">${esc(e.title)}</h3>${e.summary ? p(e.summary) : ""}${!compact && e.details?.length ? list(e.details) : ""}${!compact && published && reportPath(options.clubSlug, e.href) ? pLink("View results", reportLink(options.origin, options.clubSlug, e.href!)) : ""}${
              published
                ? (e.weekly_reports || [])
                    .filter((link) => reportPath(options.clubSlug, link.href))
                    .map((link) =>
                      pLink(
                        link.label,
                        reportLink(options.origin, options.clubSlug, link.href),
                      ),
                    )
                    .join("")
                : ""
            }`,
        )
        .join(""),
    );
  if (report.event_notes)
    rows += section("AROUND THE CLUB", p(report.event_notes), true);
  const spotlights = (recap.spotlight || [])
    .filter((s) => s.include !== false && s.players?.length)
    .slice(0, compact ? 2 : 20);
  if (spotlights.length)
    rows += section(
      "PLAYER HIGHLIGHTS",
      spotlights
        .map(
          (s) =>
            `<h3 style="${h}">${esc(s.label)}</h3>${list(compact ? s.players.slice(0, 1) : s.players)}${!compact && s.description ? p(s.description) : ""}`,
        )
        .join(""),
      true,
    );
  const races = (report.season_races || []).slice(0, compact ? 2 : 60);
  if (races.length)
    rows += section(
      "SEASON AWARDS RACE",
      races
        .map(
          (r) =>
            `<h3 style="${h}">${esc(r.title)}</h3>${list(compact ? r.lines.slice(0, 2) : r.lines)}`,
        )
        .join(""),
    );
  if (report.achievements?.length)
    rows += section(
      "NEW ACHIEVEMENTS",
      list(report.achievements.slice(0, compact ? 3 : 100)),
      true,
    );
  if (report.new_faces?.length)
    rows += section("WELCOME TO THE COURTS", p(report.new_faces.join(", ")));
  const upcoming = report.upcoming || [];
  if (upcoming.length || recap.looking_ahead?.length)
    rows += section(
      "COMING UP",
      upcoming
        .map((item) =>
          emailUrl(item.url)
            ? pLink(item.text, emailUrl(item.url))
            : p(item.text),
        )
        .join("") +
        (recap.looking_ahead?.length ? list(recap.looking_ahead) : ""),
    );
  if (fullUrl)
    rows += section(
      "KEEP UP WITH THE CLUB",
      button("View the full weekly report", fullUrl),
    );
  return wrapEmail(rows, title, options.document);
}
function wrapEmail(rows: string, title: string, document?: boolean): string {
  const fragment = `<table role="presentation" lang="en" width="100%" border="0" cellspacing="0" cellpadding="0" bgcolor="#ffffff" style="width:100%;max-width:620px;border-collapse:collapse;background-color:#ffffff;">${rows}</table>`;
  return document
    ? `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(title)}</title></head><body style="margin:0;padding:0;background:#f4f0e8;"><table role="presentation" width="100%" cellspacing="0" cellpadding="0" bgcolor="#f4f0e8"><tr><td align="center" style="padding:20px 8px;">${fragment}</td></tr></table></body></html>`
    : fragment;
}
function pLink(label: string, url: string) {
  return url
    ? `<p style="${body}margin:0 0 12px;"><a href="${esc(url)}" style="color:#17665c;text-decoration:underline;">${esc(label)}</a></p>`
    : p(label);
}

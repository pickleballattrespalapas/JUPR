import type { RecapBlock } from "@/lib/recapBuilder";
import { reportPath, emailUrl } from "@/lib/weeklyReportEmail";
import Link from "@/components/PublicClubLink";

export default function RecapEdition({
  blocks,
  clubSlug,
  scopeLabel,
}: {
  blocks: RecapBlock[];
  clubSlug: string;
  scopeLabel?: string;
}) {
  return (
    <div className="recap-edition" data-testid="recap-edition">
      {scopeLabel && scopeLabel !== "Whole club" ? <p>League: {scopeLabel}</p> : null}
      {blocks.map((block) => (
        <section
          key={block.key}
          data-section={block.key}
          style={{ marginBottom: "1.8rem", breakInside: "avoid" }}
        >
          <h2
            style={{
              fontSize: "1.2rem",
              color: "var(--text-primary, #123e37)",
            }}
          >
            {block.title}
          </h2>
          <div
            style={
              block.key === "numbers"
                ? { display: "flex", flexWrap: "wrap", gap: "1.5rem" }
                : { display: "grid", gap: "1rem" }
            }
          >
            {block.items.map((item) => (
              <article key={item.id} style={{ breakInside: "avoid" }}>
                {item.kind && block.key === "events" ? (
                  <small>{item.kind}</small>
                ) : null}
                {item.title ? (
                  <h3 style={{ margin: "0 0 .4rem" }}>{item.title}</h3>
                ) : null}
                {item.image_url ? (
                  <img
                    src={item.image_url}
                    alt={item.alt || item.title}
                    style={{
                      width: "100%",
                      maxWidth: 620,
                      maxHeight: 380,
                      objectFit: "contain",
                      borderRadius: 10,
                    }}
                  />
                ) : null}
                {item.text ? (
                  <p
                    style={{
                      whiteSpace: "pre-line",
                      margin: "0 0 .6rem",
                      fontSize: block.key === "numbers" ? "1.6rem" : undefined,
                    }}
                  >
                    {item.text}
                  </p>
                ) : null}
                {item.lines?.length ? (
                  <ul style={{ paddingLeft: "1.3rem", lineHeight: 1.7 }}>
                    {item.lines.map((line, i) => (
                      <li key={i}>{line}</li>
                    ))}
                  </ul>
                ) : null}
                {reportPath(clubSlug, item.href) ? (
                  <Link href={reportPath(clubSlug, item.href)}>
                    View results
                  </Link>
                ) : null}
                {item.links?.map((link, i) =>
                  reportPath(clubSlug, link.href) ? (
                    <p key={i}>
                      <Link href={reportPath(clubSlug, link.href)}>
                        {link.label}
                      </Link>
                    </p>
                  ) : null,
                )}
                {emailUrl(item.url) ? (
                  <p>
                    <a href={emailUrl(item.url)}>
                      {item.link_label || "Learn more"}
                    </a>
                  </p>
                ) : null}
              </article>
            ))}
          </div>
        </section>
      ))}
      {!blocks.length ? <p>No sections selected for this edition.</p> : null}
    </div>
  );
}

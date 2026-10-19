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
      {scopeLabel && scopeLabel !== "Whole club" ? (
        <p>League: {scopeLabel}</p>
      ) : null}
      {blocks.map((block) => (
        <section
          key={block.key}
          data-section={block.key}
          style={{ marginBottom: "1.8rem", breakInside: "avoid" }}
        >
          {block.title ? (
            <h2
              style={{
                fontSize: "1.2rem",
                color: "var(--text-primary, #123e37)",
              }}
            >
              {block.title}
            </h2>
          ) : null}
          <div
            style={
              block.key === "numbers"
                ? { display: "flex", flexWrap: "wrap", gap: "1.5rem" }
                : { display: "grid", gap: "1rem" }
            }
          >
            {block.items.map((item) => (
              <article
                key={item.id}
                data-recap-item={item.id}
                data-card-style={item.layout || "plain"}
                style={{
                  breakInside: "avoid",
                  overflowWrap: "anywhere",
                  ...(item.layout && item.layout !== "plain"
                    ? {
                        padding: "1.1rem",
                        borderRadius: 12,
                        color: "#263e3a",
                        background:
                          item.layout === "callout" ? "#f8f3e9" : "#ffffff",
                        border: "1px solid #e6e8e1",
                        ...(item.layout === "callout"
                          ? { borderLeft: "4px solid #cb794f" }
                          : {}),
                        ...(item.layout === "promotion"
                          ? { borderTop: "4px solid #17665c" }
                          : {}),
                      }
                    : {}),
                }}
              >
                {item.kind && ["events", "custom"].includes(block.key) ? (
                  <small
                    style={{
                      display: "block",
                      marginBottom: ".4rem",
                      color: "#17665c",
                    }}
                  >
                    {item.kind}
                  </small>
                ) : null}
                {item.title ? (
                  <h3
                    style={{
                      margin: "0 0 .4rem",
                      ...(item.layout && item.layout !== "plain"
                        ? { color: "#123e37" }
                        : {}),
                    }}
                  >
                    {item.title}
                  </h3>
                ) : null}
                {item.image_url ? (
                  <figure style={{ margin: "0 0 .8rem" }}>
                    <img
                      src={item.image_url}
                      alt={item.alt || item.title}
                      style={{
                        width: "100%",
                        maxWidth: 620,
                        maxHeight: 380,
                        objectFit: item.image_fit || "contain",
                        ...(item.image_fit === "cover"
                          ? { aspectRatio: "16 / 9" }
                          : {}),
                        borderRadius: 10,
                      }}
                    />
                    {item.caption || item.credit ? (
                      <figcaption
                        style={{ fontSize: ".85rem", marginTop: ".4rem" }}
                      >
                        {item.caption}
                        {item.caption && item.credit ? " · " : ""}
                        {item.credit ? `Photo: ${item.credit}` : ""}
                      </figcaption>
                    ) : null}
                  </figure>
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
                    <a
                      href={emailUrl(item.url)}
                      style={
                        item.layout === "promotion"
                          ? {
                              display: "inline-block",
                              padding: ".8rem 1rem",
                              background: "#17665c",
                              color: "#ffffff",
                              borderRadius: 6,
                              textDecoration: "none",
                              fontWeight: 700,
                            }
                          : undefined
                      }
                    >
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

"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import {
  formatCommerceMoney,
  quoteTournamentCommerce,
  TournamentCommerceBundleComponent,
  TournamentCommerceCatalog,
  TournamentCommerceQuote,
  TournamentCommerceSelection,
  TournamentCommerceVariant
} from "@/lib/tournamentCommerceApi";

type TournamentCommerceChooserProps = {
  clubSlug: string;
  tournamentId: string;
  registrationId?: string | null;
  eventOptionIds: string[];
  timeZone?: string | null;
  catalog: TournamentCommerceCatalog;
  initialSelections?: TournamentCommerceSelection[];
  disabled?: boolean;
  onReviewChange: (
    selections: TournamentCommerceSelection[],
    quote: TournamentCommerceQuote | null
  ) => void;
};

const inputStyle = {
  width: "100%",
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

const buttonStyle = {
  padding: "0.7rem 1rem",
  borderRadius: "10px",
  border: "1px solid #0f172a",
  background: "#0f172a",
  color: "white",
  fontWeight: 800,
  cursor: "pointer"
};

function initialQuantityMap(
  selections: TournamentCommerceSelection[] | undefined
): Record<string, number> {
  return Object.fromEntries(
    (selections || [])
      .filter((row) => row.variant_id && Number(row.quantity) > 0)
      .map((row) => [row.variant_id, Number(row.quantity)])
  );
}

function dateWindow(
  start?: string | null,
  end?: string | null,
  timeZone?: string | null
): string | null {
  if (!start && !end) return null;
  let formatter: Intl.DateTimeFormat;
  try {
    formatter = new Intl.DateTimeFormat("en-US", {
      dateStyle: "medium",
      timeStyle: "short",
      timeZone: timeZone || "UTC"
    });
  } catch {
    formatter = new Intl.DateTimeFormat("en-US", {
      dateStyle: "medium",
      timeStyle: "short",
      timeZone: "UTC"
    });
  }
  const format = (value: string) => formatter.format(new Date(value));
  if (start && end) return `Available ${format(start)} through ${format(end)}`;
  return start ? `Available from ${format(start)}` : `Available through ${format(end!)}`;
}

function componentLabel(component: TournamentCommerceBundleComponent): string {
  const label =
    component.label ||
    component.option_label ||
    (component.component_type === "EVENT_OPTION"
      ? "required event"
      : "required extra");
  if (
    component.label &&
    component.option_label &&
    component.option_label !== component.label
  ) {
    return `${component.label} — ${component.option_label}`;
  }
  return label;
}

export default function TournamentCommerceChooser({
  clubSlug,
  tournamentId,
  registrationId,
  eventOptionIds,
  timeZone,
  catalog,
  initialSelections,
  disabled = false,
  onReviewChange
}: TournamentCommerceChooserProps) {
  const [quantities, setQuantities] = useState<Record<string, number>>(() =>
    initialQuantityMap(initialSelections)
  );
  const [quote, setQuote] = useState<TournamentCommerceQuote | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const onReviewChangeRef = useRef(onReviewChange);
  onReviewChangeRef.current = onReviewChange;

  const eventKey = useMemo(
    () => [...eventOptionIds].sort().join(","),
    [eventOptionIds]
  );
  const initialSelectionKey = useMemo(
    () =>
      JSON.stringify(
        [...(initialSelections || [])].sort((left, right) =>
          left.variant_id.localeCompare(right.variant_id)
        )
      ),
    [initialSelections]
  );
  const catalogKey = `${catalog.catalog_revision || 0}:${
    catalog.catalog_fingerprint || ""
  }`;
  const selections = useMemo<TournamentCommerceSelection[]>(
    () =>
      Object.entries(quantities)
        .filter(([, quantity]) => Number(quantity) > 0)
        .map(([variant_id, quantity]) => ({
          variant_id,
          quantity: Number(quantity)
        }))
        .sort((left, right) => left.variant_id.localeCompare(right.variant_id)),
    [quantities]
  );
  const variantsByItem = useMemo(() => {
    const grouped = new Map<string, TournamentCommerceVariant[]>();
    for (const variant of catalog.variants) {
      const rows = grouped.get(variant.item_id) || [];
      rows.push(variant);
      grouped.set(variant.item_id, rows);
    }
    return grouped;
  }, [catalog.variants]);
  const componentsByBundle = useMemo(() => {
    const grouped = new Map<string, TournamentCommerceBundleComponent[]>();
    for (const component of catalog.bundle_components) {
      const rows = grouped.get(String(component.bundle_id || "")) || [];
      rows.push(component);
      grouped.set(String(component.bundle_id || ""), rows);
    }
    return grouped;
  }, [catalog.bundle_components]);

  useEffect(() => {
    setQuote(null);
    setMessage(null);
    onReviewChangeRef.current(selections, null);
    // A quote is bound to the chosen events and exact catalog revision.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eventKey, catalogKey]);

  useEffect(() => {
    setQuantities(initialQuantityMap(initialSelections));
    // The parent may replace selections after an authoritative quote conflict.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialSelectionKey]);

  function setQuantity(variantId: string, rawValue: string, maximum: number) {
    const parsed = Number.parseInt(rawValue, 10);
    const next = Number.isFinite(parsed)
      ? Math.max(0, Math.min(maximum, parsed))
      : 0;
    setQuantities((current) => ({ ...current, [variantId]: next }));
    setQuote(null);
    setMessage("Update your total before continuing.");
    const nextSelections = Object.entries({ ...quantities, [variantId]: next })
      .filter(([, quantity]) => Number(quantity) > 0)
      .map(([variant_id, quantity]) => ({
        variant_id,
        quantity: Number(quantity)
      }))
      .sort((left, right) => left.variant_id.localeCompare(right.variant_id));
    onReviewChange(nextSelections, null);
  }

  async function reviewExtras() {
    setBusy(true);
    setMessage(null);
    const response = await quoteTournamentCommerce(clubSlug, {
      tournament_id: tournamentId,
      registration_id: registrationId || null,
      event_option_ids: eventOptionIds,
      item_selections: selections
    });
    setBusy(false);
    if (response.error || !response.data?.quote) {
      setQuote(null);
      onReviewChange(selections, null);
      setMessage(response.error || "We couldn’t update your total right now. Please try again.");
      return;
    }
    setQuote(response.data.quote);
    onReviewChange(selections, response.data.quote);
    setMessage(
      selections.length
        ? "Your extras and total are updated."
        : "Your total is updated."
    );
  }

  if (!catalog.available) {
    return null;
  }

  return (
    <section
      style={{
        border: "1px solid #e2e8f0",
        borderRadius: "14px",
        padding: "1rem",
        background: "#f8fafc",
        display: "grid",
        gap: "1rem"
      }}
      aria-labelledby="tournament-extras-heading"
      data-testid="tournament-commerce-chooser"
    >
      <div>
        <h3 id="tournament-extras-heading" style={{ margin: 0 }}>
          Tournament extras
        </h3>
        <p style={{ color: "#475569", margin: "0.35rem 0 0" }}>
          Add shirts, lodging, meals, drink packs, or other tournament options.
          You can also continue without extras.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(min(100%, 260px), 1fr))",
          gap: "0.75rem"
        }}
      >
        {catalog.items.map((item) => {
          const itemVariants = variantsByItem.get(item.id) || [];
          const availability = dateWindow(
            item.available_from,
            item.available_until,
            timeZone
          );
          return (
            <article
              key={item.id}
              style={{
                border: "1px solid #cbd5e1",
                borderRadius: "12px",
                padding: "0.8rem",
                background: "white"
              }}
            >
              <h4 style={{ margin: 0 }}>{item.name}</h4>
              {item.description ? (
                <p style={{ color: "#475569", margin: "0.35rem 0" }}>
                  {item.description}
                </p>
              ) : null}
              {availability ? (
                <p style={{ color: "#64748b", fontSize: "0.9rem" }}>
                  {availability}
                </p>
              ) : null}
              <div style={{ display: "grid", gap: "0.65rem" }}>
                {itemVariants.map((variant) => {
                  const stockMaximum =
                    variant.remaining == null
                      ? item.remaining == null
                        ? item.max_per_registration
                        : Math.min(
                            item.max_per_registration,
                            Number(item.remaining)
                          )
                      : Math.min(
                          item.max_per_registration,
                          Number(variant.remaining),
                          item.remaining == null
                            ? item.max_per_registration
                            : Number(item.remaining)
                        );
                  return (
                    <label key={variant.id}>
                      <span>
                        {variant.name} ·{" "}
                        <strong>{formatCommerceMoney(variant.price_minor)}</strong>
                        {variant.remaining != null
                          ? ` · ${variant.remaining} left`
                          : ""}
                      </span>
                      <input
                        aria-label={`${item.name} ${variant.name} quantity`}
                        type="number"
                        min={0}
                        max={stockMaximum}
                        value={quantities[variant.id] || 0}
                        disabled={disabled || busy || stockMaximum <= 0}
                        onChange={(event) =>
                          setQuantity(
                            variant.id,
                            event.target.value,
                            stockMaximum
                          )
                        }
                        style={{ ...inputStyle, marginTop: "0.2rem" }}
                      />
                    </label>
                  );
                })}
              </div>
              {item.requires_fulfillment &&
              item.fulfillment_instructions ? (
                <p style={{ color: "#64748b", fontSize: "0.9rem" }}>
                  Pickup: {item.fulfillment_instructions}
                </p>
              ) : null}
            </article>
          );
        })}
      </div>

      {catalog.bundles.length ? (
        <section
          style={{
            border: "1px solid #bfdbfe",
            borderRadius: "12px",
            padding: "0.8rem",
            background: "#eff6ff"
          }}
        >
          <h4 style={{ margin: 0 }}>Bundle savings</h4>
          <p style={{ color: "#475569", margin: "0.35rem 0 0.65rem" }}>
            You’ll automatically get the best bundle price for the events and
            extras you choose.
          </p>
          <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
            {catalog.bundles.map((bundle) => (
              <li key={bundle.id}>
                <strong>{bundle.name}</strong> —{" "}
                {formatCommerceMoney(bundle.price_minor)}
                {bundle.description ? ` · ${bundle.description}` : ""}
                {(componentsByBundle.get(bundle.id) || []).length ? (
                  <ul>
                    {(componentsByBundle.get(bundle.id) || []).map(
                      (component, index) => (
                        <li key={component.id || `${bundle.id}-${index}`}>
                          {component.quantity || 1} ×{" "}
                          {componentLabel(component)}
                        </li>
                      )
                    )}
                  </ul>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {catalog.promotions.length ? (
        <p style={{ margin: 0, color: "#166534" }}>
          If you qualify for a giveaway, we’ll add it while supplies last.
        </p>
      ) : null}

      <div>
        <button
          type="button"
          onClick={reviewExtras}
          disabled={disabled || busy}
          style={buttonStyle}
        >
          {busy ? "Updating…" : "Update total"}
        </button>
      </div>

      {quote ? (
        <section
          style={{
            borderTop: "1px solid #cbd5e1",
            paddingTop: "0.8rem"
          }}
          data-testid="tournament-commerce-quote"
        >
          <h4 style={{ margin: "0 0 0.5rem" }}>Registration total</h4>
          <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
            {quote.lines.map((line) => (
              <li key={line.line_key}>
                {line.quantity} × {line.label}
                {line.option_label ? ` — ${line.option_label}` : ""}:{" "}
                {formatCommerceMoney(line.final_total_minor)}
                {line.savings_minor > 0
                  ? ` (${formatCommerceMoney(line.savings_minor)} saved)`
                  : ""}
                {line.component_snapshot?.length ? (
                  <ul>
                    {line.component_snapshot.map((component, index) => (
                      <li
                        key={
                          component.id ||
                          `${line.line_key}-${component.component_type}-${index}`
                        }
                      >
                        {component.total_quantity ||
                          component.quantity ||
                          1}{" "}
                        × {componentLabel(component)}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </li>
            ))}
          </ul>
          {quote.discount_minor > 0 ? (
            <p style={{ color: "#166534", marginBottom: "0.25rem" }}>
              Savings: {formatCommerceMoney(quote.discount_minor)}
            </p>
          ) : null}
          <p style={{ fontSize: "1.1rem", margin: "0.25rem 0" }}>
            <strong>
              Amount due to organizer: {formatCommerceMoney(quote.total_minor)}
            </strong>
          </p>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            You’ll pay the tournament organizer separately; we won’t charge you
            here.
          </p>
        </section>
      ) : null}

      <p
        aria-live="polite"
        role={message?.startsWith("We couldn’t") ? "alert" : "status"}
        style={{
          color: message?.startsWith("We couldn’t") ? "#b91c1c" : "#475569",
          margin: 0
        }}
      >
        {message ||
          "Update your total before continuing. Prices and availability may change."}
      </p>
    </section>
  );
}

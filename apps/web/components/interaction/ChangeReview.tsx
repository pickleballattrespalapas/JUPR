import type { ReactNode } from "react";

import styles from "./InteractionDialog.module.css";

export type ChangeReviewRow = {
  label: ReactNode;
  before: ReactNode;
  after: ReactNode;
  changed?: boolean;
};

export type ChangeReviewProps = {
  caption?: ReactNode;
  rows: ChangeReviewRow[];
  beforeLabel?: string;
  afterLabel?: string;
};

export function ChangeReview({
  caption = "Review changes",
  rows,
  beforeLabel = "Before",
  afterLabel = "After"
}: ChangeReviewProps) {
  return (
    <div className={styles.reviewWrap}>
      <table className={styles.reviewTable}>
        <caption className={styles.reviewCaption}>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Field</th>
            <th scope="col">{beforeLabel}</th>
            <th scope="col">{afterLabel}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className={row.changed === false ? styles.reviewNoChange : undefined}>
              <th scope="row">{row.label}</th>
              <td>{row.before}</td>
              <td>{row.changed === false ? <>No change ({row.after})</> : row.after}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

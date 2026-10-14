import type { ReactNode } from "react";
import styles from "./PublicControls.module.css";

type Props = {
  children: ReactNode;
  context?: ReactNode;
  label?: string;
  resetKey?: string;
  initiallyOpen?: boolean;
};

/** Native disclosure keeps controls accessible and forms mounted, even without JavaScript. */
export default function PublicControls({ children, context, label = "View & filters", resetKey, initiallyOpen = false }: Props) {
  return (
    <div className={styles.shell} data-public-controls>
      <details key={resetKey} className={styles.panel} open={initiallyOpen || undefined} data-public-controls-panel>
        <summary className={styles.toggle}>{label}</summary>
        <div className={styles.content}>{children}</div>
      </details>
      {context ? <div className={styles.context} data-public-view-context>{context}</div> : null}
    </div>
  );
}

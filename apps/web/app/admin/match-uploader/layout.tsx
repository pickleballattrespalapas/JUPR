import type { ReactNode } from "react";
import styles from "./layout.module.css";

type MatchUploaderLayoutProps = {
  children: ReactNode;
};

export default function MatchUploaderLayout({ children }: MatchUploaderLayoutProps) {
  return <div className={styles.root}>{children}</div>;
}

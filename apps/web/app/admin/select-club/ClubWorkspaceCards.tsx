"use client";

import type { AvailableWorkspace } from "@/lib/adminWorkspace";
import styles from "./selector.module.css";

type Props = {
  workspaces: AvailableWorkspace[];
  opening: string;
  onOpen: (workspace: AvailableWorkspace) => void;
};

export default function ClubWorkspaceCards({ workspaces, opening, onOpen }: Props) {
  return (
    <ul className={styles.clubs} aria-label="Your club workspaces">
      {workspaces.map(workspace => (
        <li key={workspace.club_id}>
          <button
            type="button"
            className={styles.club}
            aria-label={`Open ${workspace.club_name}`}
            disabled={Boolean(opening)}
            data-opening={opening === workspace.club_id}
            onClick={() => onOpen(workspace)}
          >
            <span className={styles.details}>
              <strong className={styles.name}>{workspace.club_name}</strong>
              <span className={styles.roles}>
                {workspace.roles.map(role => role.replaceAll("_", " ")).join(", ")}
              </span>
            </span>
            <span className={styles.action} aria-hidden="true">
              {opening === workspace.club_id ? "Opening…" : <>Open club <span>→</span></>}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

import { Suspense } from "react";
import PublicationEditor from "./PublicationEditor";
export default function Page() {
  return (
    <Suspense fallback={<p>Loading league publication…</p>}>
      <PublicationEditor />
    </Suspense>
  );
}

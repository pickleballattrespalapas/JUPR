import { notFound } from "next/navigation";

// Badge collections are retired. Keep old links from loading the badge service.
export default function BadgeCodexPage() {
  notFound();
}

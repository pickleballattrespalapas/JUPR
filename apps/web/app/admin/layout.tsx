import type { Metadata } from "next";
import type { ReactNode } from "react";
import AdminShell from "@/components/AdminShell";

export const metadata: Metadata = {
  robots: {
    index: false,
    follow: false,
    nocache: true
  }
};

export default function AdminLayout({ children }: { children: ReactNode }) {
  return <AdminShell>{children}</AdminShell>;
}

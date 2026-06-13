import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MannSaathi — Talk it out",
  description:
    "A safe, AI-powered companion to talk through what you're feeling. Not a therapist replacement.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

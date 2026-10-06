import type { Metadata } from "next";
import { connection } from "next/server";
import { cookies } from "next/headers";
import { Bricolage_Grotesque, JetBrains_Mono, Source_Sans_3 } from "next/font/google";
import { Sidebar } from "@/components/Sidebar";
import "./globals.css";

const display = Bricolage_Grotesque({ subsets: ["latin"], variable: "--font-display-face" });
const body = Source_Sans_3({ subsets: ["latin"], variable: "--font-body" });
const code = JetBrains_Mono({ subsets: ["latin"], variable: "--font-code" });

export const metadata: Metadata = { title: "Mentora", description: "Mentora, your personal programming tutor" };

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  await connection(); // the sidebar shows live progress
  const collapsed = (await cookies()).get("sidebar")?.value === "collapsed";
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${code.variable}`}>
      <body className={`font-sans antialiased lg:grid lg:h-screen ${collapsed ? "lg:grid-cols-[60px_minmax(0,1fr)]" : "lg:grid-cols-[320px_minmax(0,1fr)]"} lg:overflow-hidden`}>
        <Sidebar collapsed={collapsed} />
        <div className="min-w-0 lg:h-screen lg:overflow-y-auto">{children}</div>
      </body>
    </html>
  );
}

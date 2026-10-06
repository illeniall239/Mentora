import type { Metadata } from "next";
import { Fraunces, JetBrains_Mono, Plus_Jakarta_Sans } from "next/font/google";
import { Topbar } from "@/components/Topbar";
import "./globals.css";

const body = Plus_Jakarta_Sans({ subsets: ["latin"], variable: "--font-body" });
const code = JetBrains_Mono({ subsets: ["latin"], variable: "--font-code" });
const logo = Fraunces({ subsets: ["latin"], variable: "--font-logo-face", axes: ["opsz"] }); // the wordmark only

export const metadata: Metadata = { title: "Mentora", description: "Mentora, your personal programming tutor" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${body.variable} ${code.variable} ${logo.variable}`}>
      <body className="font-sans antialiased lg:flex lg:h-screen lg:flex-col lg:overflow-hidden">
        <Topbar />
        <div className="min-w-0 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">{children}</div>
      </body>
    </html>
  );
}

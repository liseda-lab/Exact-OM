import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Newsreader } from "next/font/google";

import "@/styles/base.css";
import "@/styles/app.css";
import "@/styles/study.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin", "latin-ext"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

const newsreader = Newsreader({
  variable: "--font-newsreader",
  subsets: ["latin", "latin-ext"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Exact Explain",
  description: "Inspect ontology entities, Exact's saved matching decisions and prepared explanations.",
  referrer: "no-referrer",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

// Applies stored text size and theme before first paint. Static content, so the server
// can allow it by hash under a strict Content-Security-Policy.
const PREFERENCES_BOOTSTRAP = `try{var s=Number(localStorage.getItem("exact.textScale"));if([1,1.125,1.25,1.5,1.75,2].indexOf(s)>=0)document.documentElement.style.setProperty("--text-scale",String(s));var t=localStorage.getItem("exact.theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t);}catch(e){}`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable} ${newsreader.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: PREFERENCES_BOOTSTRAP }} />
      </head>
      <body>{children}</body>
    </html>
  );
}

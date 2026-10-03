"use client";

// Development-only preview of the proposed exact-study/2.0 participant flow against an
// in-browser synthetic service. Production builds compile this page to a notice only; no
// deployment profile serves it.

import dynamic from "next/dynamic";

const Preview =
  process.env.NODE_ENV === "production"
    ? null
    : dynamic(() => import("@/components/study/preview/PreviewParticipant").then((module) => module.PreviewParticipant), { ssr: false });

export default function PreviewPage() {
  if (!Preview) {
    return (
      <main className="page-message">
        <h1>Development preview only</h1>
        <p>This page exists only in development builds.</p>
      </main>
    );
  }
  return <Preview />;
}

"use client";

import { CompareView } from "@/components/explore/CompareView";
import { ExploreShell } from "@/components/shell/ExploreShell";

export default function ComparePage() {
  return (
    <ExploreShell active="compare">
      <CompareView />
    </ExploreShell>
  );
}

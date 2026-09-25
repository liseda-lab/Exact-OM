"use client";

import { BrowseView } from "@/components/explore/BrowseView";
import { ExploreShell } from "@/components/shell/ExploreShell";

export default function BrowsePage() {
  return (
    <ExploreShell active="browse">
      <BrowseView />
    </ExploreShell>
  );
}

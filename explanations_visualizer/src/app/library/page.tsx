"use client";

import { LibraryView } from "@/components/library/LibraryView";
import { ExploreShell } from "@/components/shell/ExploreShell";

export default function LibraryPage() {
  return (
    <ExploreShell active="library" allowWithoutBundle>
      <LibraryView />
    </ExploreShell>
  );
}

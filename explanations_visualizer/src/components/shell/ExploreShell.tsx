"use client";

import { ExploreProvider, useExplore } from "@/components/explore/ExploreContext";
import { Skeleton } from "@/components/common/ErrorNote";
import { AppHeader, type NavKey } from "@/components/shell/AppHeader";

function Gate({ active, children, allowWithoutBundle }: { active: NavKey; children: React.ReactNode; allowWithoutBundle?: boolean }) {
  const state = useExplore();
  if (state.phase === "legacy") {
    return (
      <div className="page-message" id="main">
        <h1>Opening the historical run viewer…</h1>
      </div>
    );
  }
  if (state.phase === "unreachable") {
    return (
      <div className="page-message" id="main" role="alert">
        <h1>The Exact service is not responding</h1>
        <p className="muted">{state.error} Start it with exact-inspect serve, then try again.</p>
        <button type="button" className="btn btn-primary" onClick={state.reload}>
          Try again
        </button>
      </div>
    );
  }
  if (state.phase === "loading") {
    return (
      <div className="page-message" id="main" aria-busy="true">
        <Skeleton lines={4} />
      </div>
    );
  }
  if (state.phase === "no_bundle" && !allowWithoutBundle) {
    return (
      <div className="page-message" id="main">
        <h1>No bundle is open</h1>
        <p className="muted">Import an inspection bundle or choose one from the library to start.</p>
        {state.health?.profile === "local_app" && (
          <a className="btn btn-primary" href="/library/">
            Open the library
          </a>
        )}
      </div>
    );
  }
  void active;
  return <>{children}</>;
}

export function ExploreShell({ active, children, allowWithoutBundle = false }: { active: NavKey; children: React.ReactNode; allowWithoutBundle?: boolean }) {
  return (
    <ExploreProvider>
      <div className="app-root">
        <AppHeader active={active} />
        <Gate active={active} allowWithoutBundle={allowWithoutBundle}>
          {children}
        </Gate>
      </div>
    </ExploreProvider>
  );
}

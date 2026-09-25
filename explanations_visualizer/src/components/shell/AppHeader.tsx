"use client";

import { useState } from "react";

import { useExplore } from "@/components/explore/ExploreContext";
import { IconMenu } from "@/components/common/Icons";
import { HelpDialog } from "@/components/shell/HelpDialog";
import { TextSizeControl, ThemeControl } from "@/components/shell/Preferences";
import { shortHash } from "@/lib/iri";

export type NavKey = "compare" | "browse" | "library";

export function BrandMark() {
  return (
    <span className="brand">
      <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
        <circle cx="9" cy="13" r="7" fill="var(--source)" />
        <rect x="12" y="6" width="13" height="14" rx="3" fill="var(--target)" opacity="0.92" />
      </svg>
      <span className="brand-name">Exact Explain</span>
    </span>
  );
}

export function AppHeader({ active }: { active: NavKey }) {
  const state = useExplore();
  const [help, setHelp] = useState(false);
  const [menu, setMenu] = useState(false);
  const profile = state.health?.profile;
  const local = profile === "local_app";
  const run = state.runs[0];
  const links: { key: NavKey; href: string; label: string }[] = [
    { key: "compare", href: "/", label: "Compare" },
    { key: "browse", href: "/browse/", label: "Browse ontologies" },
    ...(local ? [{ key: "library" as NavKey, href: "/library/", label: "Library" }] : []),
  ];
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="app-header">
        <BrandMark />
        <nav aria-label="Main" className={menu ? "app-nav open" : "app-nav"} id="app-nav">
          {links.map((link) => (
            <a key={link.key} href={link.href} aria-current={active === link.key ? "page" : undefined} className="app-nav-link">
              {link.label}
            </a>
          ))}
          <button type="button" className="app-nav-link nav-help" onClick={() => setHelp(true)}>
            How to read this screen
          </button>
        </nav>
        <div className="app-header-spacer" />
        {state.health?.package_id && (
          <span className="bundle-chip" title={state.health.package_id}>
            <span className="bundle-dot" aria-hidden="true" />
            <span className="bundle-chip-text">
              {run ? <strong>{run.run_id}</strong> : <strong>Ontology bundle</strong>}
              <span className="meta"> · bundle {shortHash(state.health.package_id, 8)}</span>
            </span>
          </span>
        )}
        {profile && <span className="pill profile-pill">{local ? "Local app" : "Public demo"}</span>}
        <div className="header-prefs">
          <TextSizeControl />
          <ThemeControl />
        </div>
        <button type="button" className="icon-btn help-btn" aria-label="How to read this screen" onClick={() => setHelp(true)}>
          ?
        </button>
        <button type="button" className="icon-btn menu-btn" aria-label="Menu" aria-expanded={menu} aria-controls="app-nav" onClick={() => setMenu((open) => !open)}>
          <IconMenu />
        </button>
      </header>
      {profile === "public_demo" && (
        <div className="demo-banner" role="note">
          Public demo of a fixed development bundle. Importing bundles and study features are not available here.
        </div>
      )}
      {help && <HelpDialog onClose={() => setHelp(false)} />}
    </>
  );
}

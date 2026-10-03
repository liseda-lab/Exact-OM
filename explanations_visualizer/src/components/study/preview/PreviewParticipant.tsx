"use client";

// Runs the real participant application against the in-browser v2 preview service.

import { useEffect, useState } from "react";

import { ParticipantApp } from "@/components/study/ParticipantApp";
import { installFixtureServer, previewOffline, resetPreview, setPreviewOffline } from "@/study/v2/fixtureServer";

export function PreviewParticipant() {
  const [ready, setReady] = useState(false);
  const [offline, setOffline] = useState(false);
  useEffect(() => {
    let undo: (() => void) | null = null;
    let cancelled = false;
    installFixtureServer().then((restore) => {
      if (cancelled) restore();
      else {
        undo = restore;
        setOffline(previewOffline());
        setReady(true);
      }
    });
    return () => {
      cancelled = true;
      undo?.();
    };
  }, []);
  if (!ready) return null;
  return (
    <>
      <div className="preview-banner" role="note">
        <strong>Development preview.</strong> The proposed exact-study/2.0 flow runs against a synthetic service inside this browser. Nothing is sent anywhere or scored.
        <label className="check">
          <input
            type="checkbox"
            checked={offline}
            onChange={(event) => {
              setPreviewOffline(event.target.checked);
              setOffline(event.target.checked);
            }}
          />
          Simulate a lost connection
        </label>
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => {
            resetPreview();
            try {
              window.sessionStorage.clear();
            } catch {
              /* preview only */
            }
            window.location.reload();
          }}
        >
          Start over
        </button>
      </div>
      <ParticipantApp />
    </>
  );
}

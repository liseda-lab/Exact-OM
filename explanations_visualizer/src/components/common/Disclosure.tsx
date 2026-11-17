"use client";

// A small keyboard/touch disclosure for explanations that are useful once but noisy when
// repeated (21 F04, S0-C1): the summary is a real button with aria-expanded, never hover-only.

import { useId, useState } from "react";

export function Disclosure({ label, openLabel, children, className }: { label: string; openLabel?: string; children: React.ReactNode; className?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className={className ? `disclosure ${className}` : "disclosure"}>
      <button type="button" className="link-btn disclosure-toggle" aria-expanded={open} aria-controls={id} onClick={() => setOpen((value) => !value)}>
        {open ? openLabel ?? label : label}
      </button>
      {open && (
        <span id={id} className="disclosure-body">
          {children}
        </span>
      )}
    </span>
  );
}

/** What a recorded score means, its stage and calibration, on request. */
export function ScoreMeaning({ meaning, stage, calibration, label = "What this score means" }: { meaning: string; stage?: string; calibration?: string; label?: string }) {
  return (
    <Disclosure label={label} openLabel="Hide score meaning" className="score-meaning">
      {meaning}
      {stage || calibration ? (
        <span className="meta">
          {stage ? ` · stage: ${stage}` : ""}
          {calibration ? ` · calibration: ${calibration.replace(/_/g, " ")}` : ""}
        </span>
      ) : null}
    </Disclosure>
  );
}

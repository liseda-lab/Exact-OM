"use client";

import { Dialog } from "@/components/common/Dialog";
import { SideMarker } from "@/components/common/Icons";

/** Persistent explicit help: the legend is always reachable, never hover-only. */
export function HelpDialog({ onClose }: { onClose: () => void }) {
  return (
    <Dialog title="How to read this screen" onClose={onClose} wide>
      <div className="help-grid">
        <section>
          <h3>Two sides</h3>
          <p className="help-row">
            <SideMarker side="source" /> <strong className="text-source">Source</strong> entities have a circle marker.
          </p>
          <p className="help-row">
            <SideMarker side="target" /> <strong className="text-target">Target</strong> entities have a square marker.
          </p>
          <p className="muted">Colour is never the only cue; every card also names its side and kind.</p>
        </section>
        <section>
          <h3>Where text comes from</h3>
          <p className="help-row">
            <span className="origin-tag">Original</span> Exactly as the ontology file states it, with its predicate.
          </p>
          <p className="help-row">
            <span className="origin-tag origin-generated">Generated</span> Prepared in advance; every claim cites the facts it rests on.
          </p>
          <p className="help-row">
            <span className="origin-tag origin-matcher">Matcher record</span> What Exact saved when it ran.
          </p>
        </section>
        <section>
          <h3>Evidence lines</h3>
          <ul className="legend-list">
            <li>
              <svg width="44" height="10" aria-hidden="true">
                <line x1="0" y1="5" x2="44" y2="5" stroke="currentColor" strokeWidth="2.5" />
              </svg>
              Ontology assertion
            </li>
            <li>
              <svg width="44" height="10" aria-hidden="true">
                <line x1="0" y1="5" x2="44" y2="5" stroke="currentColor" strokeWidth="2.5" strokeDasharray="8 5" />
              </svg>
              Structural navigation (documented rule)
            </li>
            <li>
              <svg width="44" height="10" aria-hidden="true">
                <line x1="0" y1="5" x2="44" y2="5" stroke="currentColor" strokeWidth="2.5" strokeDasharray="10 4 2 4" />
              </svg>
              Feature Exact selected when scoring
            </li>
            <li>
              <svg width="44" height="12" aria-hidden="true">
                <line x1="0" y1="3" x2="44" y2="3" stroke="var(--bridge)" strokeWidth="2" />
                <line x1="0" y1="9" x2="44" y2="9" stroke="var(--bridge)" strokeWidth="2" />
              </svg>
              Comparison across the two ontologies
            </li>
          </ul>
        </section>
        <section>
          <h3>Numbers</h3>
          <p>
            Scores are <strong>matching scores</strong> with the meaning the run recorded. None is shown as a probability of being correct unless the bundle
            carries a validated calibration.
          </p>
          <p className="muted">Matching weights on evidence stay hidden until you ask for them.</p>
        </section>
      </div>
    </Dialog>
  );
}

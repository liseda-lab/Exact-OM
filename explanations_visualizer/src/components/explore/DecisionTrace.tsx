"use client";

// The decision trace for one saved candidate pair (21 F03): a short summary of the recorded
// route first, then the recorded sequence in order. Exact matching is named as such; runs of
// unrecorded or intentionally skipped stages are collapsed with their stages one click away;
// raw codes and implementation IDs stay available as technical details.

import { useState } from "react";

import { Disclosure, ScoreMeaning } from "@/components/common/Disclosure";
import { formatScore } from "@/components/explore/candidates";
import { GROUP_TEXT, eventMeaning, groupTrace, routeSummary, stageName } from "@/lib/trace";
import type { Candidate, PairTrace } from "@/lib/types";

type TraceEvent = PairTrace["events"][number];

const RELATION_NAMES: Record<string, string> = {
  equivalent: "Equivalent",
  source_narrower: "Source is narrower",
  source_broader: "Source is broader",
  not_equivalent: "Not equivalent",
  unresolved: "Unresolved",
};

const BASIS_NAMES: Record<string, string> = {
  default_convention: "The run's default convention, not a measured confidence.",
  graph_closure_with_anchors: "Derived from hierarchy closure with stated cross-ontology anchors.",
  recorded_other: "Recorded by another relation method.",
  unavailable: "How the relation was set was not recorded.",
};

function EventRow({ event, index }: { event: TraceEvent; index: number }) {
  const [details, setDetails] = useState(false);
  const meaning = eventMeaning(event);
  const failed = event.status === "failed";
  return (
    <li className={`trace-item trace-${event.status}`}>
      <span className="trace-dot" aria-hidden="true">
        {index + 1}
      </span>
      <div className="trace-body">
        <div className="trace-head">
          <h4>{stageName(event.stage)}</h4>
          {failed && <span className="status status-bad">Failed</span>}
        </div>
        <p className="trace-reason">{meaning.text}</p>
        {event.scores.length > 0 && (
          <ul className="score-rows">
            {event.scores.map((score) => (
              <li key={`${score.name}-${score.stage}`}>
                <span className="score-chip">
                  <strong>{scoreName(score.name)}</strong> {formatScore(score.value)}
                </span>
                <ScoreMeaning meaning={score.meaning} />
              </li>
            ))}
          </ul>
        )}
        <button type="button" className="link-btn trace-more" aria-expanded={details} onClick={() => setDetails((value) => !value)}>
          {details ? "Hide technical details" : "Technical details"}
        </button>
        {details && (
          <dl className="kv trace-tech">
            <div>
              <dt>Recorded</dt>
              <dd className="mono">
                {event.stage} · {event.status} · {event.outcome} · {event.reason_code}
              </dd>
            </div>
            {event.competitor_pair_ids.length > 0 && (
              <div>
                <dt>Compared against</dt>
                <dd>
                  {event.competitor_pair_ids.length} competing {event.competitor_pair_ids.length === 1 ? "pair" : "pairs"}
                </dd>
              </div>
            )}
            {event.implementation_id && (
              <div>
                <dt>Implementation</dt>
                <dd className="mono">{event.implementation_id}</dd>
              </div>
            )}
          </dl>
        )}
      </div>
    </li>
  );
}

function GroupRow({ reason, events }: { reason: keyof typeof GROUP_TEXT; events: { event: TraceEvent; index: number }[] }) {
  const [open, setOpen] = useState(false);
  const first = events[0].index + 1;
  const last = events[events.length - 1].index + 1;
  return (
    <li className="trace-item trace-group">
      <span className="trace-dot" aria-hidden="true">
        {first}–{last}
      </span>
      <div className="trace-body">
        <p className="trace-reason">
          <strong>{events.map((item) => stageName(item.event.stage)).join(", ")}</strong> · {GROUP_TEXT[reason]}
        </p>
        <button type="button" className="link-btn trace-more" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
          {open ? "Hide these stages" : `Show these ${events.length} stages`}
        </button>
        {open && (
          <ol className="trace-list trace-sublist">
            {events.map(({ event, index }) => (
              <EventRow key={`${event.stage}-${index}`} event={event} index={index} />
            ))}
          </ol>
        )}
      </div>
    </li>
  );
}

export function DecisionTrace({ trace }: { trace: PairTrace }) {
  const summary = routeSummary(trace.events);
  const selection = trace.events.find((event) => event.stage === "selection" && event.status === "completed");
  const items = groupTrace(trace.events);
  return (
    <div className="trace">
      <div className="trace-summary">
        <h3 className="trace-heading" tabIndex={-1}>
          How this pair was decided
        </h3>
        <p className="trace-route">{summary.text}</p>
        <dl className="kv">
          <div>
            <dt>Selection</dt>
            <dd>{selection ? eventMeaning(selection).text : "No selection record"}</dd>
          </div>
          <div>
            <dt>Saved alignment</dt>
            <dd>{trace.saved_alignment_member === null ? "Not known" : trace.saved_alignment_member ? "Included" : "Not included"}</dd>
          </div>
          <div>
            <dt>Relation</dt>
            <dd>
              {trace.relation ? RELATION_NAMES[trace.relation] : "None recorded"}
              {trace.relation && BASIS_NAMES[trace.relation_basis] ? <span className="meta block">{BASIS_NAMES[trace.relation_basis]}</span> : null}
            </dd>
          </div>
        </dl>
      </div>
      <h3 className="subhead">Recorded sequence</h3>
      <ol className="trace-list">
        {items.map((item) =>
          item.kind === "event" ? <EventRow key={`${item.event.stage}-${item.index}`} event={item.event} index={item.index} /> : <GroupRow key={`group-${item.events[0].index}`} reason={item.reason} events={item.events} />,
        )}
      </ol>
    </div>
  );
}

/** The concise default label: the final decision score is "Matching score"; others keep their recorded names. */
function scoreName(name: string): string {
  return name === "S_final" ? "Matching score" : name;
}

export function ScoresPanel({ candidate }: { candidate: Candidate }) {
  const ranks = candidate.ordinal_ranks;
  const nil = Object.entries(candidate.nil);
  return (
    <div className="scores-panel">
      <Disclosure label="About these scores" className="scores-about">
        Each value names the stage that produced it. Scores rank candidates; they are not probabilities of being correct unless marked calibrated.
      </Disclosure>
      <ul className="score-rows">
        {candidate.scores.map((score) => (
          <li key={`${score.name}-${score.stage}`}>
            <span className="score-chip">
              <strong>{scoreName(score.name)}</strong> {formatScore(score.value)}
            </span>
            {score.name === "S_final" && <span className="meta">{score.name}</span>}
            <ScoreMeaning meaning={score.meaning} stage={score.stage} calibration={score.calibration_status} />
          </li>
        ))}
        {candidate.scores.length === 0 && <li className="meta">No score was recorded for this pair.</li>}
      </ul>
      <h3 className="subhead">Ranks</h3>
      <dl className="kv">
        <div>
          <dt>Candidate order</dt>
          <dd>{ranks.candidate_joint_rank ?? "Not recorded"}{ranks.joint_ordering ? <span className="meta"> · {ranks.joint_ordering}</span> : null}</dd>
        </div>
        <div>
          <dt>Retrieval order</dt>
          <dd>
            {ranks.retrieval_rank ?? "Not recorded"}
            {ranks.retrieval_ordering ? <span className="meta"> · {ranks.retrieval_ordering.replace(/_/g, " ")}</span> : null}
            {ranks.retrieval_tie_rule ? <span className="meta"> · ties: {ranks.retrieval_tie_rule.replace(/_/g, " ")}</span> : null}
          </dd>
        </div>
        <div>
          <dt>No-match rank</dt>
          <dd>{ranks.nil_rank ?? "Not recorded"}</dd>
        </div>
      </dl>
      {nil.length > 0 && (
        <>
          <h3 className="subhead">No-match values</h3>
          <ul className="score-rows">
            {nil.map(([name, value]) => (
              <li key={name}>
                <span className="score-chip">
                  <strong>{name}</strong> {typeof value === "number" ? formatScore(value) : String(value)}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="meta">Membership in the saved alignment: {candidate.membership_provenance.status === "derived_from_saved_artifacts" ? "read from the saved alignment file" : "unavailable"}.</p>
    </div>
  );
}

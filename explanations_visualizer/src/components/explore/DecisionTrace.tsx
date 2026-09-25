"use client";

import { formatScore } from "@/components/explore/candidates";
import type { Candidate, PairTrace } from "@/lib/types";

const STAGE_NAMES: Record<string, string> = {
  retrieval: "Retrieval",
  prefilter: "Prefilter",
  pair_scoring: "Pair scoring",
  llm_signal: "Language-model signal",
  selection: "Selection",
  threshold: "Threshold",
  cardinality: "Cardinality",
  extraction: "Extraction",
  relation_typing: "Relation typing",
  repair: "Repair",
};

const STATUS_NAMES: Record<string, string> = {
  completed: "Completed",
  not_run: "Not run",
  not_recorded: "Not recorded",
  failed: "Failed",
};

const OUTCOME_NAMES: Record<string, string> = {
  retained: "Retained",
  rejected: "Rejected",
  abstained: "Abstained",
  selected: "Selected",
  not_selected: "Not selected",
  not_applicable: "Not applicable",
  unknown: "Unknown",
};

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

function reasonText(code: string): string {
  if (code === "historical_stage_record_not_exported") return "The saved run did not keep a record of this stage, so whether it ran is unknown.";
  return code.replace(/_/g, " ");
}

export function DecisionTrace({ trace }: { trace: PairTrace }) {
  return (
    <div className="trace">
      <ol className="trace-list">
        {trace.events.map((event, index) => (
          <li key={`${event.stage}-${index}`} className={`trace-item trace-${event.status}`}>
            <span className="trace-dot" aria-hidden="true">
              {index + 1}
            </span>
            <div className="trace-body">
              <div className="trace-head">
                <h3>{STAGE_NAMES[event.stage] ?? event.stage}</h3>
                <span className={event.status === "completed" ? "status status-ok" : event.status === "failed" ? "status status-bad" : "status"}>{STATUS_NAMES[event.status] ?? event.status}</span>
                <span className="muted">{OUTCOME_NAMES[event.outcome] ?? event.outcome}</span>
              </div>
              <p className="trace-reason">{reasonText(event.reason_code)}</p>
              {event.scores.length > 0 && (
                <ul className="score-rows">
                  {event.scores.map((score) => (
                    <li key={`${score.name}-${score.stage}`}>
                      <span className="score-chip">
                        <strong>{score.name}</strong> {formatScore(score.value)}
                      </span>
                      <span className="muted">{score.meaning}</span>
                    </li>
                  ))}
                </ul>
              )}
              {event.competitor_pair_ids.length > 0 && <p className="meta">Compared against {event.competitor_pair_ids.length} competing {event.competitor_pair_ids.length === 1 ? "pair" : "pairs"} at this stage.</p>}
              {event.implementation_id && <p className="meta">Implementation: {event.implementation_id}</p>}
            </div>
          </li>
        ))}
      </ol>
      <div className="trace-summary card-inset">
        <h3>Final result</h3>
        <dl className="kv">
          <div>
            <dt>Saved alignment</dt>
            <dd>{trace.saved_alignment_member === null ? "Not known: no saved alignment was bound" : trace.saved_alignment_member ? "Member" : "Not a member"}</dd>
          </div>
          <div>
            <dt>Relation</dt>
            <dd>
              {trace.relation ? RELATION_NAMES[trace.relation] : "None recorded"}
              <span className="meta block">{BASIS_NAMES[trace.relation_basis]}</span>
            </dd>
          </div>
        </dl>
      </div>
    </div>
  );
}

export function ScoresPanel({ candidate }: { candidate: Candidate }) {
  const ranks = candidate.ordinal_ranks;
  const nil = Object.entries(candidate.nil);
  return (
    <div className="scores-panel">
      <p className="muted">Each value keeps the stage that produced it and what it means. None is a probability of being correct unless marked as calibrated.</p>
      <ul className="score-rows">
        {candidate.scores.map((score) => (
          <li key={`${score.name}-${score.stage}`}>
            <span className="score-chip">
              <strong>{score.name}</strong> {formatScore(score.value)}
            </span>
            <span>
              {score.meaning} <span className="meta">· stage: {score.stage} · calibration: {score.calibration_status.replace(/_/g, " ")}</span>
            </span>
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

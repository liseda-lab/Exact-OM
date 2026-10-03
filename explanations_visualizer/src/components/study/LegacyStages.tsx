"use client";

// exact-study/1.0 setup, controls practice and per-case reporting, kept for legacy
// publications and sessions exactly as they were frozen: the v1 service still requires its
// installation self-report, tutorial indices and final-only consultation. Corrected
// publications use the v2 stages instead; this module is never used to imply v2 behaviour.

import { useEffect, useMemo, useRef, useState } from "react";

import { IconCheck } from "@/components/common/Icons";
import { RankingPanel, type RankingValue } from "@/components/study/RankingPanel";
import { ResourceDownloads } from "@/components/study/Stages";
import { Paragraphs } from "@/components/study/StudyChrome";
import { orderedEntries, orderedKeys } from "@/study/formOrder";
import { FALLBACK_PRACTICE, practiceActionComplete } from "@/study/practice";
import type { StudySession } from "@/study/session";
import { legacySetup, type SetupReceipt, type StudyState } from "@/study/types";

const SETUP_CHECKS: { key: keyof SetupReceipt; label: string; group: number }[] = [
  { key: "protege_installed", label: "Protégé is installed on this computer", group: 1 },
  { key: "source_opened", label: "The first ontology file opens in Protégé", group: 2 },
  { key: "target_opened", label: "The second ontology file opens in Protégé", group: 2 },
  { key: "practice_source_located", label: "I found the practice class named in the setup instructions", group: 3 },
  { key: "practice_definition_parents_inspected", label: "I looked at its definition and its parents", group: 3 },
];

function setupDraft(state: StudyState): SetupReceipt {
  const setup = legacySetup(state);
  return {
    protege_installed: setup?.protege_installed ?? false,
    source_opened: setup?.source_opened ?? false,
    target_opened: setup?.target_opened ?? false,
    practice_source_located: setup?.practice_source_located ?? false,
    practice_definition_parents_inspected: setup?.practice_definition_parents_inspected ?? false,
    protege_version: setup?.protege_version ?? null,
    completed_tutorial_steps: setup?.completed_tutorial_steps ?? [],
  };
}

export function LegacySetupStage({ state, session }: { state: StudyState; session: StudySession }) {
  const [draft, setDraft] = useState<SetupReceipt>(() => setupDraft(state));
  const edited = useRef(false);
  useEffect(() => {
    if (!edited.current) setDraft(setupDraft(state));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.setup]);
  const [busy, setBusy] = useState(false);
  const ready = SETUP_CHECKS.every((check) => draft[check.key] === true);
  const remaining = SETUP_CHECKS.filter((check) => !draft[check.key]).length;
  const send = (receipt: SetupReceipt, transition: boolean) =>
    session.mutate({ method: "PUT", path: "/api/v1/study/setup", body: { ...receipt, protege_version: receipt.protege_version || null }, coalesce: transition ? undefined : "setup", debounceMs: transition ? 0 : 500, transition });
  // Partial drafts autosave; the v1 service advances once every confirmation is true, so a
  // complete set is only sent by the explicit Continue button.
  const updateDraft = (next: SetupReceipt) => {
    edited.current = true;
    setDraft(next);
    if (!SETUP_CHECKS.every((check) => next[check.key] === true)) void send(next, false).catch(() => undefined);
  };
  useEffect(() => {
    if (session.save.kind === "conflict") {
      edited.current = false;
      setDraft(setupDraft(state));
    }
  }, [session.save, state]);
  const toggle = (key: keyof SetupReceipt) => updateDraft({ ...draft, [key]: !draft[key] });
  return (
    <div className="study-page">
      <div className="study-intro">
        <h1>Set up Protégé and the two ontologies</h1>
        <p className="lead">This study version asks you to confirm that Protégé and both files open. We only ask you to confirm each step; nothing on your computer is checked.</p>
      </div>
      <section className="study-card">
        <h2>Instructions</h2>
        <div className="prose">
          <Paragraphs text={state.setup_instructions} />
        </div>
      </section>
      <section className="study-card">
        <h2>1 · Install Protégé</h2>
        <label className="check-row">
          <input type="checkbox" checked={draft.protege_installed} onChange={() => toggle("protege_installed")} />
          <span>{SETUP_CHECKS[0].label}</span>
        </label>
        <div className="field narrow-field">
          <label htmlFor="protege-version">
            Protégé version, if you know it <span className="meta">Optional</span>
          </label>
          <input id="protege-version" className="input" maxLength={80} placeholder="for example 5.6" value={draft.protege_version ?? ""} onChange={(event) => updateDraft({ ...draft, protege_version: event.target.value })} />
        </div>
      </section>
      <section className="study-card">
        <h2>2 · Download both ontology files</h2>
        <p className="muted">These are the exact files used in the study. Open them from your downloads folder in Protégé.</p>
        <ResourceDownloads state={state} />
        {SETUP_CHECKS.filter((check) => check.group === 2).map((check) => (
          <label key={check.key} className="check-row">
            <input type="checkbox" checked={Boolean(draft[check.key])} onChange={() => toggle(check.key)} />
            <span>{check.label}</span>
          </label>
        ))}
      </section>
      <section className="study-card">
        <h2>3 · A quick navigation check</h2>
        <p className="muted">Not scored. The setup instructions above name a practice class to look up in Protégé.</p>
        {SETUP_CHECKS.filter((check) => check.group === 3).map((check) => (
          <label key={check.key} className="check-row">
            <input type="checkbox" checked={Boolean(draft[check.key])} onChange={() => toggle(check.key)} />
            <span>{check.label}</span>
          </label>
        ))}
      </section>
      <div className="study-actions">
        <button
          type="button"
          className="btn btn-primary btn-large"
          disabled={!ready || busy}
          aria-describedby="setup-remaining"
          onClick={async () => {
            setBusy(true);
            try {
              await send(draft, true);
            } catch {
              /* shown in header */
            } finally {
              setBusy(false);
            }
          }}
        >
          Continue
        </button>
        <span id="setup-remaining" className="muted">
          {ready ? "All steps confirmed." : `${remaining} ${remaining === 1 ? "confirmation" : "confirmations"} left`}
        </span>
      </div>
      <p className="meta">Something not working? Your progress is saved; close this page and return later with your private link. This study version cannot continue to scored cases without these steps.</p>
    </div>
  );
}

export function LegacyPracticeStage({ state, session }: { state: StudyState; session: StudySession }) {
  const steps = state.tutorial_steps;
  const setup = legacySetup(state);
  const [done, setDone] = useState<Set<number>>(() => new Set(setup?.completed_tutorial_steps ?? []));
  const [sandbox, setSandbox] = useState<RankingValue>({ responseType: null, ranked: [] });
  const cases = state.practice_cases?.length ? state.practice_cases : FALLBACK_PRACTICE;
  const [practiceIndex, setPracticeIndex] = useState(0);
  const [finished, setFinished] = useState<Set<string>>(() => new Set());
  const [feedback, setFeedback] = useState<string | null>(null);
  const current = cases[practiceIndex];
  const [inspecting, setInspecting] = useState<string | null>(null);
  const inspected = current.candidates.find((candidate) => candidate.candidate_id === inspecting) ?? current.candidates[0];
  const allPractised = cases.every((item) => finished.has(item.practice_id));
  const [busy, setBusy] = useState(false);
  const all = useMemo(() => steps.every((_, index) => done.has(index)), [steps, done]);
  const start = async () => {
    if (!setup) return;
    setBusy(true);
    try {
      await session.mutate({
        method: "PUT",
        path: "/api/v1/study/setup",
        body: {
          protege_installed: setup.protege_installed,
          source_opened: setup.source_opened,
          target_opened: setup.target_opened,
          practice_source_located: setup.practice_source_located,
          practice_definition_parents_inspected: setup.practice_definition_parents_inspected,
          protege_version: setup.protege_version,
          completed_tutorial_steps: steps.map((_, index) => index),
        },
        transition: true,
      });
    } catch {
      /* header shows it */
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="study-page study-page-wide">
      <div className="study-intro">
        <span className="pill pill-warn">Not scored</span>
        <h1>How the study works</h1>
        <p className="lead">Read each step, then practise from simple definitions to qualified descriptions, partial rankings and an explicit none answer. Nothing on this page is scored.</p>
        <p className="note">
          All practice concepts are synthetic. They are separate from the study’s ontology files and scored cases.{" "}
          {state.practice_cases?.length ? "These examples are frozen with this study revision." : "This older study publication uses the interface’s built-in controls tutorial; owner-specific practice content has not been supplied."}
        </p>
      </div>
      <ol className="tutorial-steps">
        {steps.map((text, index) => (
          <li key={index} className={done.has(index) ? "tutorial-step done" : "tutorial-step"}>
            <span className="tutorial-number" aria-hidden="true">
              {done.has(index) ? <IconCheck /> : index + 1}
            </span>
            <div className="tutorial-body">
              <div className="prose">
                <Paragraphs text={text} />
              </div>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={done.has(index)}
                  onChange={() =>
                    setDone((value) => {
                      const next = new Set(value);
                      if (next.has(index)) next.delete(index);
                      else next.add(index);
                      return next;
                    })
                  }
                />
                <span>I have read step {index + 1}</span>
              </label>
            </div>
          </li>
        ))}
      </ol>
      <section className="study-card" aria-labelledby="sandbox-h">
        <h2 id="sandbox-h">{current.title}</h2>
        <p>{current.instructions}</p>
        <p className="meta">
          Practice {practiceIndex + 1} of {cases.length}. No matcher was run. Definitions below are authored examples, not generated explanations or clinical guidance.
        </p>
        <article className="entity-card entity-card-source compact">
          <h3>Practice source: {current.source.label}</h3>
          <p>{current.source.description}</p>
        </article>
        <article className="entity-card entity-card-target compact" aria-live="polite">
          <h3>Practice candidate: {inspected.label}</h3>
          <p>{inspected.description}</p>
        </article>
        <RankingPanel
          key={current.practice_id}
          practice
          candidates={current.candidates.map((candidate, index) => ({
            id: candidate.candidate_id,
            position: index + 1,
            label: candidate.label,
            identifier: `practice:${candidate.candidate_id}`,
            score: "—",
            scoreMeaning: "No matcher scores exist for these synthetic practice concepts. In scored cases, a matching score is a suggestion and never a probability of correctness.",
          }))}
          value={sandbox}
          inspecting={inspected.candidate_id}
          onInspect={setInspecting}
          onChange={(next) => {
            setSandbox(next);
            setFeedback(null);
          }}
          locked={false}
          onSubmit={() => {
            if (!practiceActionComplete(current.kind, sandbox)) {
              setFeedback(
                current.kind === "partial_ranking"
                  ? "For this controls lesson, leave at least one candidate unranked and rank at least one."
                  : current.kind === "none_of_these"
                    ? "For this lesson, explicitly choose None of these. Insufficient information remains a distinct valid response in the study."
                    : "For this lesson, add at least one candidate to practise ranking.",
              );
              return;
            }
            setFinished((previous) => new Set([...previous, current.practice_id]));
            setFeedback("Controls practice completed. This checks the response action only, not correctness. Nothing was submitted as a study answer.");
          }}
          submitting={false}
          submitLabel="Check practice action"
        />
        {feedback && (
          <p className="note note-info" role="status">
            {feedback}
          </p>
        )}
        <div className="study-actions">
          <button
            type="button"
            className="btn"
            disabled={practiceIndex === 0}
            onClick={() => {
              setPracticeIndex((index) => index - 1);
              setSandbox({ responseType: null, ranked: [] });
              setInspecting(null);
              setFeedback(null);
            }}
          >
            Previous practice
          </button>
          <button
            type="button"
            className="btn"
            disabled={!finished.has(current.practice_id) || practiceIndex === cases.length - 1}
            onClick={() => {
              setPracticeIndex((index) => index + 1);
              setSandbox({ responseType: null, ranked: [] });
              setInspecting(null);
              setFeedback(null);
            }}
          >
            Next practice
          </button>
        </div>
      </section>
      <div className="study-actions">
        <button type="button" className="btn btn-primary btn-large" disabled={!all || !allPractised || busy || !setup} onClick={start}>
          Start the scored cases
        </button>
        <span className="muted">{all && allPractised ? "Your cases will be assigned when you start." : `Confirm all ${steps.length} tutorial steps and complete all ${cases.length} practice actions to continue.`}</span>
      </div>
    </div>
  );
}

interface LocalConsultation {
  consulted: boolean | null;
  methods: string[];
  otherEditor: string;
  otherResource: string;
}

function localKey(state: StudyState) {
  return `exact.study.consultation.${state.session_id}.${state.current_presentation_id}`;
}

/** v1 has no server draft: an unfinished answer is kept only in this tab until it is saved. */
function readLocal(state: StudyState): LocalConsultation | null {
  try {
    const raw = window.sessionStorage.getItem(localKey(state));
    return raw ? (JSON.parse(raw) as LocalConsultation) : null;
  } catch {
    return null;
  }
}

export function LegacyConsultationStage({ state, session, sourceLabel }: { state: StudyState; session: StudySession; sourceLabel: string | null }) {
  const form = state.forms.consultation;
  const labelOf = (id: string) => form.find((question) => question.id === id);
  const methodsQuestion = labelOf("methods");
  const restored = useMemo(() => (typeof window === "undefined" ? null : readLocal(state)), [state]);
  const [consulted, setConsulted] = useState<boolean | null>(restored?.consulted ?? null);
  const [methods, setMethods] = useState<string[]>(restored?.methods ?? []);
  const [otherEditor, setOtherEditor] = useState(restored?.otherEditor ?? "");
  const [otherResource, setOtherResource] = useState(restored?.otherResource ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    try {
      window.sessionStorage.setItem(localKey(state), JSON.stringify({ consulted, methods, otherEditor, otherResource }));
    } catch {
      /* a lost local draft only means answering again */
    }
  }, [consulted, methods, otherEditor, otherResource, state]);
  const methodOptions = methodsQuestion ? orderedEntries(methodsQuestion.options, orderedKeys(methodsQuestion, "options", state.forms.version)) : [];
  const valid = consulted === false || (consulted === true && methods.length > 0);
  const save = async () => {
    if (!valid || !state.current_case_id) {
      setError(consulted === true ? "Choose at least one way you consulted them, or answer No." : "Please answer Yes or No.");
      return;
    }
    setBusy(true);
    try {
      await session.mutate({
        method: "PUT",
        path: `/api/v1/study/cases/${encodeURIComponent(state.current_case_id)}/consultation`,
        body: {
          consulted_external_ontologies: consulted,
          methods: consulted ? methods : [],
          other_editor: consulted && methods.includes("other_editor") && otherEditor.trim() ? otherEditor.trim() : null,
          other_resource: consulted && methods.includes("other_resource") && otherResource.trim() ? otherResource.trim() : null,
        },
        transition: true,
      });
      setError(null);
      try {
        window.sessionStorage.removeItem(localKey(state));
      } catch {
        /* nothing to clear */
      }
    } catch {
      /* header shows it */
    } finally {
      setBusy(false);
    }
  };
  const next = state.completed_cases + 2 <= state.assigned_case_count ? `Save and go to case ${state.completed_cases + 2}` : "Save and continue";
  return (
    <div className="study-page">
      <div className="study-intro">
        <h1>One question about this case</h1>
        <p className="lead">Your answer{sourceLabel ? ` for “${sourceLabel}”` : ""} is saved and can no longer change. This question is not timed as part of the case.</p>
        {restored && <p className="meta">Your unfinished answer to this question was restored from this browser tab. It is not saved on the study server until you continue.</p>}
      </div>
      <fieldset className="question">
        <legend className="question-label">{labelOf("consulted_external_ontologies")?.label ?? "For this case, did you consult the ontology resources outside the study's explanation panels?"}</legend>
        <div className="option-grid">
          {[
            [true, "Yes"],
            [false, "No"],
          ].map(([value, label]) => (
            <label key={String(value)} className={consulted === value ? "option on" : "option"}>
              <input
                type="radio"
                name="consulted"
                checked={consulted === value}
                onChange={() => {
                  setConsulted(value as boolean);
                  if (!value) setMethods([]);
                  setError(null);
                }}
              />
              <span>{label as string}</span>
            </label>
          ))}
        </div>
      </fieldset>
      {consulted && methodsQuestion && (
        <fieldset className="question">
          <legend className="question-label">
            {methodsQuestion.label} <span className="meta">Select all that apply</span>
          </legend>
          <div className="option-grid option-grid-column">
            {methodOptions.map(([code, label]) => (
              <label key={code} className={methods.includes(code) ? "option on" : "option"}>
                <input type="checkbox" checked={methods.includes(code)} onChange={() => setMethods((value) => (value.includes(code) ? value.filter((item) => item !== code) : [...value, code]))} />
                <span>{label}</span>
              </label>
            ))}
          </div>
          {methods.includes("other_editor") && (
            <div className="field">
              <label htmlFor="other-editor">
                {labelOf("other_editor")?.label ?? "Which other ontology editor did you use?"} <span className="meta">Optional</span>
              </label>
              <input id="other-editor" className="input" maxLength={160} value={otherEditor} onChange={(event) => setOtherEditor(event.target.value)} />
            </div>
          )}
          {methods.includes("other_resource") && (
            <div className="field">
              <label htmlFor="other-resource">
                {labelOf("other_resource")?.label ?? "Which other ontology resource or viewer did you use?"} <span className="meta">Optional</span>
              </label>
              <input id="other-resource" className="input" maxLength={160} value={otherResource} onChange={(event) => setOtherResource(event.target.value)} />
            </div>
          )}
        </fieldset>
      )}
      <div className="study-actions">
        <button type="button" className="btn btn-primary btn-large" disabled={busy} onClick={save}>
          {busy ? "Saving…" : next}
        </button>
        {error && (
          <span className="field-error" role="alert">
            {error}
          </span>
        )}
      </div>
      <p className="meta">Using the study&apos;s own panels is recorded separately; this asks only about outside resources.</p>
    </div>
  );
}

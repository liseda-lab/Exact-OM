"use client";

// Participant steps before and after the scored cases. Every answer is saved to the study
// server; nothing is considered done until the server acknowledges it.

import { useEffect, useMemo, useRef, useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { IconCheck, IconCopy, IconDownload, IconLink } from "@/components/common/Icons";
import { RankingPanel, type RankingValue } from "@/components/study/RankingPanel";
import { missingRequired, pruneHidden, QuestionnaireForm, type Answers } from "@/components/study/QuestionnaireForm";
import { Paragraphs } from "@/components/study/StudyChrome";
import { ApiError } from "@/lib/api";
import { shortHash } from "@/lib/iri";
import { formatBytes } from "@/lib/zipManifest";
import type { StudySession } from "@/study/session";
import type { SetupReceipt, StudyState } from "@/study/types";

function useDebounced(callback: () => void, delay: number, deps: unknown[]) {
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const timer = window.setTimeout(callback, delay);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

export function WelcomeStage({ state, session }: { state: StudyState; session: StudySession }) {
  const [confirmDecline, setConfirmDecline] = useState(false);
  const [busy, setBusy] = useState(false);
  const respond = async (accepted: boolean) => {
    setBusy(true);
    try {
      await session.mutate({ method: "PUT", path: "/api/v1/study/consent", body: { information_version: state.information_version, accepted } });
    } catch {
      /* the header shows the save outcome */
    } finally {
      setBusy(false);
      setConfirmDecline(false);
    }
  };
  return (
    <div className="study-page">
      <div className="study-intro">
        <h1>Welcome, and thank you for considering this study</h1>
        <p className="lead">
          You will judge which concepts from one ontology mean the same as a concept from another, using two different sets of tools. You can pause at any point and come back with your private link.
        </p>
      </div>
      <section className="study-card">
        <h2>What you will do</h2>
        <ol className="plain-steps">
          <li>Check that Protégé is installed and both ontology files open.</li>
          <li>Answer a few questions about your background. No name, email or institution is asked.</li>
          <li>Practise with the controls. Practice is not scored.</li>
          <li>Rank candidates for {state.assigned_case_count || "a number of"} source concepts, in two blocks that use different tools.</li>
          <li>Tell us what helped and what was hard.</li>
        </ol>
        <p className="muted">We record your answers, the actions you take in the study pages and how long each step takes. We do not record what you do inside Protégé.</p>
      </section>
      <section className="study-card study-card-info">
        <IconLink className="icon study-card-icon" />
        <div>
          <h2>Keep your private link</h2>
          <p>
            Your link is the only way back to your answers, on this or any other device. Anyone who has it can continue as you, so please do not share it. We cannot look you up by name: if both the link and
            this browser&apos;s session are lost, the session cannot be recovered.
          </p>
        </div>
      </section>
      <section className="study-card" aria-labelledby="consent-h">
        <div className="section-head">
          <h2 id="consent-h">Information and consent</h2>
          <span className="meta">Version {state.information_version}</span>
        </div>
        <div className="consent-text prose" tabIndex={0} aria-label="Participant information and consent text">
          <Paragraphs text={state.information_text} />
          <Paragraphs text={state.consent_text} />
        </div>
        <div className="study-actions">
          <button type="button" className="btn btn-primary btn-large" disabled={busy} onClick={() => respond(true)}>
            I agree to take part
          </button>
          <button type="button" className="btn btn-large" disabled={busy} onClick={() => setConfirmDecline(true)}>
            I do not want to take part
          </button>
        </div>
        <p className="meta">Choosing not to take part closes this link. Nothing further is collected.</p>
      </section>
      {confirmDecline && (
        <Dialog title="Close this link?" onClose={() => setConfirmDecline(false)}>
          <p>If you do not want to take part, this private link will be closed and no further information will be collected.</p>
          <div className="study-actions">
            <button type="button" className="btn btn-danger" disabled={busy} onClick={() => respond(false)}>
              Yes, I do not want to take part
            </button>
            <button type="button" className="btn" onClick={() => setConfirmDecline(false)}>
              Go back
            </button>
          </div>
        </Dialog>
      )}
    </div>
  );
}

const SETUP_CHECKS: { key: keyof SetupReceipt; label: string; group: number }[] = [
  { key: "protege_installed", label: "Protégé is installed on this computer", group: 1 },
  { key: "source_opened", label: "The first ontology file opens in Protégé", group: 2 },
  { key: "target_opened", label: "The second ontology file opens in Protégé", group: 2 },
  { key: "practice_source_located", label: "I found the practice class named in the setup instructions", group: 3 },
  { key: "practice_definition_parents_inspected", label: "I looked at its definition and its parents", group: 3 },
];

function setupDraft(state: StudyState): SetupReceipt {
  return {
    protege_installed: state.setup?.protege_installed ?? false,
    source_opened: state.setup?.source_opened ?? false,
    target_opened: state.setup?.target_opened ?? false,
    practice_source_located: state.setup?.practice_source_located ?? false,
    practice_definition_parents_inspected: state.setup?.practice_definition_parents_inspected ?? false,
    protege_version: state.setup?.protege_version ?? null,
    completed_tutorial_steps: state.setup?.completed_tutorial_steps ?? [],
  };
}

export function ResourceDownloads({ state, compact = false }: { state: StudyState; compact?: boolean }) {
  const [copied, setCopied] = useState<string | null>(null);
  if (!state.ontology_resources.length) return <p className="note">No ontology file is attached to this study.</p>;
  return (
    <ul className={compact ? "downloads compact" : "downloads"}>
      {state.ontology_resources.map((asset) => (
        <li key={asset.asset_id} className="download-card">
          <span className="download-name">{asset.asset_id}</span>
          <span className="meta">
            {asset.media_type === "application/owl-functional" ? "OWL Functional Syntax" : asset.media_type} · {formatBytes(asset.size_bytes)}
          </span>
          {!compact && (
            <span className="download-hash">
              <span className="iri">SHA-256 {shortHash(asset.sha256, 16)}</span>
              <button
                type="button"
                className="btn btn-quiet btn-sm"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(asset.sha256);
                    setCopied(asset.asset_id);
                  } catch {
                    setCopied(null);
                  }
                }}
              >
                <IconCopy /> {copied === asset.asset_id ? "Copied" : "Copy full hash"}
              </button>
            </span>
          )}
          <a className="btn" href={`/api/v1/study/resources/${encodeURIComponent(asset.asset_id)}`} download>
            <IconDownload /> Download
          </a>
        </li>
      ))}
    </ul>
  );
}

export function SetupStage({ state, session }: { state: StudyState; session: StudySession }) {
  const [draft, setDraft] = useState<SetupReceipt>(() => setupDraft(state));
  const [busy, setBusy] = useState(false);
  const ready = SETUP_CHECKS.every((check) => draft[check.key] === true);
  const remaining = SETUP_CHECKS.filter((check) => !draft[check.key]).length;
  const send = (receipt: SetupReceipt) =>
    session.mutate({ method: "PUT", path: "/api/v1/study/setup", body: { ...receipt, protege_version: receipt.protege_version || null }, coalesce: "setup" });
  // Save partial progress; the complete confirmation is sent only by Continue.
  useDebounced(
    () => {
      if (!ready) void send(draft).catch(() => undefined);
    },
    800,
    [draft],
  );
  const toggle = (key: keyof SetupReceipt) => setDraft((current) => ({ ...current, [key]: !current[key] }));
  return (
    <div className="study-page">
      <div className="study-intro">
        <h1>Set up Protégé and the two ontologies</h1>
        <p className="lead">The scored cases need a desktop computer with Protégé and both files open. We only ask you to confirm each step; nothing on your computer is checked.</p>
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
          <input id="protege-version" className="input" maxLength={80} placeholder="for example 5.6" value={draft.protege_version ?? ""} onChange={(event) => setDraft((current) => ({ ...current, protege_version: event.target.value }))} />
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
              await send(draft);
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
      <p className="meta">Something not working? Your progress is saved; close this page and return later with your private link. The study cannot continue to scored cases without these steps.</p>
    </div>
  );
}

function serverFieldError(error: unknown): { field: string; message: string } | null {
  if (!(error instanceof ApiError)) return null;
  const match = /^(?:Required answer|Invalid choice|Invalid choices|Special choice must be exclusive|Complete all ratings|Invalid matrix|Invalid rating|Invalid optional text): (\w+)/.exec(error.message);
  return match ? { field: match[1], message: error.message.split(":")[0] } : null;
}

export function FormStage({
  state,
  session,
  formId,
  title,
  intro,
  submitLabel,
  onSubmitted,
}: {
  state: StudyState;
  session: StudySession;
  formId: "background" | "final";
  title: string;
  intro: string;
  submitLabel: string;
  onSubmitted?: () => Promise<void>;
}) {
  const questions = state.forms[formId];
  const saved = state.questionnaires[formId]?.answers ?? {};
  const [answers, setAnswers] = useState<Answers>(saved);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const touched = useRef(false);
  const handledConflict = useRef(0);
  useEffect(() => {
    if (session.save.kind !== "conflict" || session.save.at === handledConflict.current) return;
    handledConflict.current = session.save.at;
    touched.current = false;
    setAnswers(state.questionnaires[formId]?.answers ?? {});
  }, [session.save, state.questionnaires, formId]);
  const save = (submitted: boolean) =>
    session.mutate({
      method: "PUT",
      path: `/api/v1/study/questionnaires/${formId}`,
      body: { form_version: state.forms.version, answers: pruneHidden(questions, answers), submitted },
      coalesce: `form:${formId}`,
      persist: !submitted,
    });
  useDebounced(
    () => {
      if (touched.current) void save(false).catch(() => undefined);
    },
    900,
    [answers],
  );
  const submit = async () => {
    const missing = missingRequired(questions, answers);
    if (missing.length) {
      setErrors(Object.fromEntries(missing.map((question) => [question.id, "Please answer this question. “Prefer not to say” is available where it applies."])));
      document.getElementById(`q-${missing[0].id}`)?.scrollIntoView({ block: "center" });
      return;
    }
    setBusy(true);
    try {
      await save(true);
      if (onSubmitted) await onSubmitted();
    } catch (error) {
      const field = serverFieldError(error);
      if (field) setErrors({ [field.field]: field.message });
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="study-page">
      <div className="study-intro">
        <h1>{title}</h1>
        <p className="lead">{intro}</p>
        {formId === "background" && state.forms.experience_note && <p className="muted">{state.forms.experience_note}</p>}
      </div>
      <QuestionnaireForm
        questions={questions}
        answers={answers}
        errors={errors}
        onChange={(next, changed) => {
          touched.current = true;
          setAnswers(pruneHidden(questions, next));
          setErrors((current) => {
            const rest = { ...current };
            delete rest[changed];
            return rest;
          });
        }}
        revealNote={(question) => (question.show_if ? "Shown because of your previous answer." : null)}
      />
      <div className="study-actions">
        <button type="button" className="btn btn-primary btn-large" disabled={busy} onClick={submit}>
          {busy ? "Saving…" : submitLabel}
        </button>
        {Object.keys(errors).length > 0 && (
          <span className="field-error" role="alert">
            {Object.keys(errors).length} required {Object.keys(errors).length === 1 ? "question needs" : "questions need"} an answer.
          </span>
        )}
      </div>
    </div>
  );
}

const SANDBOX = ["A", "B", "C", "D", "E"].map((letter, index) => ({
  id: `practice-${letter}`,
  position: index + 1,
  label: `Placeholder candidate ${letter}`,
  identifier: `practice:${letter}`,
  score: "—",
  scoreMeaning: "Practice placeholders have no scores and no correct answer.",
}));

export function PracticeStage({ state, session }: { state: StudyState; session: StudySession }) {
  const steps = state.tutorial_steps;
  const [done, setDone] = useState<Set<number>>(() => new Set(state.setup?.completed_tutorial_steps ?? []));
  const [sandbox, setSandbox] = useState<RankingValue>({ responseType: null, ranked: [] });
  const [tried, setTried] = useState(false);
  const [busy, setBusy] = useState(false);
  const all = useMemo(() => steps.every((_, index) => done.has(index)), [steps, done]);
  const start = async () => {
    if (!state.setup) return;
    setBusy(true);
    try {
      await session.mutate({
        method: "PUT",
        path: "/api/v1/study/setup",
        body: {
          protege_installed: state.setup.protege_installed,
          source_opened: state.setup.source_opened,
          target_opened: state.setup.target_opened,
          practice_source_located: state.setup.practice_source_located,
          practice_definition_parents_inspected: state.setup.practice_definition_parents_inspected,
          protege_version: state.setup.protege_version,
          completed_tutorial_steps: steps.map((_, index) => index),
        },
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
        <p className="lead">Read each step, then try the ranking controls below. Nothing on this page is scored.</p>
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
                    setDone((current) => {
                      const next = new Set(current);
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
        <h2 id="sandbox-h">Try the controls</h2>
        <p className="muted">
          These placeholder candidates only demonstrate the controls: add candidates in the order you prefer, move or remove them, undo, keep the initial order, or choose None of these or Insufficient
          information. Scored cases have real concepts and some have no equivalent candidate at all.
        </p>
        <RankingPanel
          candidates={SANDBOX}
          value={sandbox}
          onChange={(next) => {
            setSandbox(next);
            setTried(true);
          }}
          locked={false}
          onSubmit={() => setTried(true)}
          submitting={false}
          submitLabel="Check (practice only)"
        />
        {tried && <p className="note note-info">In scored cases, “Submit answer” saves your answer permanently. Here nothing was sent.</p>}
      </section>
      <div className="study-actions">
        <button type="button" className="btn btn-primary btn-large" disabled={!all || busy || !state.setup} onClick={start}>
          Start the scored cases
        </button>
        <span className="muted">{all ? "Your cases will be assigned when you start." : `Confirm all ${steps.length} steps to continue.`}</span>
      </div>
    </div>
  );
}

export function ConsultationStage({ state, session, sourceLabel }: { state: StudyState; session: StudySession; sourceLabel: string | null }) {
  const form = state.forms.consultation;
  const labelOf = (id: string) => form.find((question) => question.id === id);
  const methodsQuestion = labelOf("methods");
  const [consulted, setConsulted] = useState<boolean | null>(null);
  const [methods, setMethods] = useState<string[]>([]);
  const [otherEditor, setOtherEditor] = useState("");
  const [otherResource, setOtherResource] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
      });
      setError(null);
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
        <p className="lead">
          Your answer{sourceLabel ? ` for “${sourceLabel}”` : ""} is saved and can no longer change. This question is not timed as part of the case.
        </p>
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
            {Object.entries(methodsQuestion.options ?? {}).map(([code, label]) => (
              <label key={code} className={methods.includes(code) ? "option on" : "option"}>
                <input type="checkbox" checked={methods.includes(code)} onChange={() => setMethods((current) => (current.includes(code) ? current.filter((item) => item !== code) : [...current, code]))} />
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

export function PausedStage({ session, onResume }: { session: StudySession; onResume: (activity: "external_work" | "break" | "unknown") => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  void session;
  return (
    <div className="study-page">
      <section className="study-card">
        <h1 className="study-card-title">Paused</h1>
        <p>Everything up to now is saved. You can close this window; your private link brings you back here.</p>
        <div className="study-actions">
          <button
            type="button"
            className="btn btn-primary btn-large"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await onResume("break");
              } finally {
                setBusy(false);
              }
            }}
          >
            Resume
          </button>
        </div>
      </section>
    </div>
  );
}

export function GapDialog({ onAnswer }: { onAnswer: (activity: "external_work" | "break" | "unknown") => void }) {
  const [choice, setChoice] = useState<"external_work" | "break" | "unknown" | null>(null);
  return (
    <Dialog title="Welcome back" onClose={() => onAnswer("unknown")} closeLabel="Skip this question">
      <p>This page was closed for a while during a case. What were you doing in that time?</p>
      <div className="option-grid option-grid-column">
        {(
          [
            ["external_work", "Working on this case in Protégé or the ontology files"],
            ["break", "Taking a break"],
            ["unknown", "Not sure"],
          ] as const
        ).map(([value, label]) => (
          <label key={value} className={choice === value ? "option on" : "option"}>
            <input type="radio" name="gap" checked={choice === value} onChange={() => setChoice(value)} />
            <span>{label}</span>
          </label>
        ))}
      </div>
      <div className="study-actions">
        <button type="button" className="btn btn-primary" disabled={!choice} onClick={() => choice && onAnswer(choice)}>
          Continue
        </button>
      </div>
      <p className="meta">Your draft answer is exactly as you left it.</p>
    </Dialog>
  );
}

export function MessagePage({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="study-page">
      <section className="study-card">
        <h1 className="study-card-title">{title}</h1>
        {children}
      </section>
    </div>
  );
}

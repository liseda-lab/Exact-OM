"use client";

// Participant steps shared by every study protocol version. Every answer is saved to the
// study server; nothing is considered done until the server acknowledges it. Version-specific
// setup, practice/tutorial and per-case reporting live in LegacyStages (exact-study/1.0) and
// the v2 stage modules.

import { useEffect, useRef, useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { IconLink } from "@/components/common/Icons";
import { missingRequired, pruneHidden, QuestionnaireForm, type Answers } from "@/components/study/QuestionnaireForm";
import { ResourceList } from "@/components/study/Resources";
import { Paragraphs } from "@/components/study/StudyChrome";
import { ApiError } from "@/lib/api";
import { formOrderProblems } from "@/study/formOrder";
import type { StudySession } from "@/study/session";
import type { StudyState } from "@/study/types";

export type Protocol = "v1" | "v2";

export function WelcomeStage({ state, session, protocol }: { state: StudyState; session: StudySession; protocol: Protocol }) {
  const [confirmDecline, setConfirmDecline] = useState(false);
  const [busy, setBusy] = useState(false);
  const respond = async (accepted: boolean) => {
    setBusy(true);
    try {
      await session.mutate({ method: "PUT", path: "/api/v1/study/consent", body: { information_version: state.information_version, accepted }, transition: true });
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
        {protocol === "v2" ? (
          <ol className="plain-steps">
            <li>Read the task and check that you can get the two ontology files. You do not need to install or use any particular program.</li>
            <li>Answer a few questions about your background. No name, email or institution is asked.</li>
            <li>Learn the tools in a short interactive tutorial with practice examples, then answer five short questions about the task. Practice is not scored, and you can retry.</li>
            <li>Rank candidates for {state.assigned_case_count || "a number of"} source concepts, in two blocks that use different tools. After each case, say which inspection methods, if any, you used for it.</li>
            <li>Tell us what helped and what was hard.</li>
          </ol>
        ) : (
          <ol className="plain-steps">
            <li>Check that Protégé is installed and both ontology files open.</li>
            <li>Answer a few questions about your background. No name, email or institution is asked.</li>
            <li>Practise with the controls. Practice is not scored.</li>
            <li>Rank candidates for {state.assigned_case_count || "a number of"} source concepts, in two blocks that use different tools.</li>
            <li>Tell us what helped and what was hard.</li>
          </ol>
        )}
        <p className="muted">
          We record your answers, the actions you take in the study pages and how long each step takes. We do not record what you do outside the study pages{protocol === "v1" ? ", including in Protégé" : ""}.
        </p>
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
        <div className="consent-text prose" role="region" tabIndex={0} aria-label="Participant information and consent text">
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

export function ResourceDownloads({ state, compact = false }: { state: StudyState; compact?: boolean }) {
  return <ResourceList resources={state.ontology_resources} compact={compact} />;
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
  const edited = useRef(false);
  useEffect(() => {
    if (!edited.current) setAnswers(state.questionnaires[formId]?.answers ?? {});
  }, [state.questionnaires, formId]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const handledConflict = useRef(0);
  useEffect(() => {
    if (session.save.kind !== "conflict" || session.save.at === handledConflict.current) return;
    handledConflict.current = session.save.at;
    edited.current = false;
    setAnswers(state.questionnaires[formId]?.answers ?? {});
  }, [session.save, state.questionnaires, formId]);
  const problems = formOrderProblems(questions, state.forms.version);
  const save = (submitted: boolean, value: Answers = answers) =>
    session.mutate({
      method: "PUT",
      path: `/api/v1/study/questionnaires/${formId}`,
      body: { form_version: state.forms.version, answers: pruneHidden(questions, value), submitted },
      coalesce: submitted ? undefined : `form:${formId}`,
      debounceMs: submitted ? 0 : 900,
      transition: submitted,
    });
  const submit = async () => {
    const missing = missingRequired(questions, answers);
    if (missing.length) {
      setErrors(Object.fromEntries(missing.map((question) => [question.id, "Please answer this question. “Prefer not to say” is available where it applies."])));
      const first = document.getElementById(`q-${missing[0].id}`);
      first?.scrollIntoView({ block: "center" });
      first?.querySelector<HTMLElement>("input, textarea")?.focus({ preventScroll: true });
      return;
    }
    setBusy(true);
    try {
      if (!state.questionnaires[formId]?.submitted) await save(true);
      if (onSubmitted) await onSubmitted();
    } catch (error) {
      const field = serverFieldError(error);
      if (field) setErrors({ [field.field]: field.message });
    } finally {
      setBusy(false);
    }
  };
  if (problems.length) {
    return (
      <MessagePage title="This questionnaire cannot be shown as published">
        <p>This version of the study interface cannot show the questionnaire in its declared order, so it does not collect answers here. Nothing you saved earlier is lost.</p>
        <p className="meta">Form version {state.forms.version}. Please tell the study team; they can check the publication.</p>
      </MessagePage>
    );
  }
  return (
    <div className="study-page">
      <div className="study-intro">
        <h1>{title}</h1>
        <p className="lead">{intro}</p>
        {formId === "background" && state.forms.experience_note && <p className="muted">{state.forms.experience_note}</p>}
      </div>
      <QuestionnaireForm
        questions={questions}
        version={state.forms.version}
        answers={answers}
        disabled={busy || state.questionnaires[formId]?.submitted}
        errors={errors}
        onChange={(next, changed) => {
          edited.current = true;
          const visible = pruneHidden(questions, next);
          setAnswers(visible);
          void save(false, visible).catch(() => undefined);
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

export function PausedStage({ onResume }: { onResume: (activity: "external_work" | "break" | "unknown") => Promise<void> }) {
  const [busy, setBusy] = useState(false);
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
            ["external_work", "Working on this case outside the study pages (for example, in an ontology tool or the files)"],
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

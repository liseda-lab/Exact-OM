"use client";

// exact-study/2.0 tool-neutral setup (14, 16 B2). The participant acknowledges the task,
// confirms access to the supplied files and that outside inspection is optional. Nothing
// requires installing, opening or using a named program; an access problem offers help and
// save-and-return, never a false confirmation or a different study condition. Drafts never
// advance; only the explicit Continue submits.

import { useEffect, useRef, useState } from "react";

import { OPEN_HELP, ResourceList, SCOPE_NOTICE } from "@/components/study/Resources";
import { Paragraphs } from "@/components/study/StudyChrome";
import { orderedEntries, orderedKeys } from "@/study/formOrder";
import type { StudySession } from "@/study/session";
import { isReviewProtocol, isSetupV2, type ResourceAccess, type SetupV2, type StudyState } from "@/study/types";

export const METHOD_LABELS: [string, string][] = [
  ["protege", "Protégé"],
  ["other_editor", "Another ontology editor"],
  ["plain_files", "The ontology files directly (for example, in a text editor)"],
  ["other_resource", "Another ontology resource or viewer"],
  ["queries_scripts", "Queries or scripts"],
  ["reasoner", "A reasoner"],
  ["other_method", "Another method"],
];

/** study-methods/3 (22 B06): Protégé is a recommendation, never a requirement or an assumed use. */
export const METHOD_FREEDOM_V3 =
  "Inspecting the supplied ontologies outside this site is optional. If you want a tool, we recommend Protégé, but you do not need to install or use it; a text editor, another method or no outside inspection at all are equally fine. You can combine methods and change them between cases. After each case we ask which methods, if any, you actually used for that case.";

/** Legacy (setup/2) wording, kept for historical publications. */
export const METHOD_FREEDOM =
  "You may inspect the supplied ontologies using any method or combination of methods. Protégé is one option; installing or using it is not required. You may also complete a case without external tools. Use the supplied ontology versions and follow the information guidance above. You can change methods between cases. After each ranking, we will ask which methods, if any, you used for that source and its candidates.";

function draftOf(state: StudyState): SetupV2 {
  const setup = isSetupV2(state.setup) ? state.setup : null;
  return {
    setup_version: setup?.setup_version ?? state.protocol_versions?.setup ?? "setup/2",
    instructions_acknowledged: setup?.instructions_acknowledged ?? false,
    external_inspection_optional_understood: setup?.external_inspection_optional_understood ?? false,
    resource_access: setup?.resource_access ?? "not_checked",
    familiar_methods: setup?.familiar_methods ?? [],
    familiar_other_method: setup?.familiar_other_method ?? null,
    submitted: false,
  };
}

export function SetupStageV2({ state, session }: { state: StudyState; session: StudySession }) {
  const v3 = isReviewProtocol(state);
  const familiarQuestion = state.forms.setup?.find((question) => question.id === "familiar_methods") ?? null;
  const otherQuestion = state.forms.setup?.find((question) => question.id === "familiar_other_method") ?? null;
  const familiarOptions: [string, string][] = v3 && familiarQuestion ? orderedEntries(familiarQuestion.options, orderedKeys(familiarQuestion, "options", state.forms.version)) : METHOD_LABELS;
  const otherInput = useRef<HTMLInputElement>(null);
  const [otherError, setOtherError] = useState<string | null>(null);
  const [draft, setDraft] = useState<SetupV2>(() => draftOf(state));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const edited = useRef(false);
  useEffect(() => {
    if (!edited.current) setDraft(draftOf(state));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.setup]);
  useEffect(() => {
    if (session.save.kind === "conflict") {
      edited.current = false;
      setDraft(draftOf(state));
    }
  }, [session.save, state]);

  const body = (value: SetupV2, submitted: boolean) => ({
    setup_version: value.setup_version,
    instructions_acknowledged: value.instructions_acknowledged,
    external_inspection_optional_understood: value.external_inspection_optional_understood,
    resource_access: value.resource_access,
    familiar_methods: value.familiar_methods,
    // setup/3: the named Other methods travel only while Other is selected (the server clears it too).
    ...(v3 ? { familiar_other_method: value.familiar_methods.includes("other_method") ? value.familiar_other_method?.trim() || null : null } : {}),
    submitted,
  });
  /** Familiarity choices; "None of these" is exclusive. Familiarity never implies use in a case. */
  const toggleFamiliar = (code: string) => {
    const selected = draft.familiar_methods;
    if (selected.includes(code)) return update({ familiar_methods: selected.filter((item) => item !== code) });
    if (code === "none") return update({ familiar_methods: ["none"] });
    update({ familiar_methods: [...selected.filter((item) => item !== "none"), code] });
  };
  const update = (changes: Partial<SetupV2>) => {
    edited.current = true;
    const next = { ...draft, ...changes };
    setDraft(next);
    setError(null);
    void session.mutate({ method: "PUT", path: "/api/v1/study/setup", body: body(next, false), coalesce: "setup", debounceMs: 500 }).catch(() => undefined);
  };
  const ready = draft.instructions_acknowledged && draft.external_inspection_optional_understood && draft.resource_access === "available";
  const otherMissing = v3 && draft.familiar_methods.includes("other_method") && !draft.familiar_other_method?.trim();
  const submit = async () => {
    if (otherMissing) {
      setOtherError("Name the other methods, or clear Other method.");
      otherInput.current?.focus();
      return;
    }
    if (!ready) {
      setError(
        draft.resource_access === "needs_help"
          ? "You said you have a problem getting the files. Try again, or save and come back later; the study keeps your progress."
          : "Please confirm the three points above to continue.",
      );
      return;
    }
    setBusy(true);
    try {
      await session.mutate({ method: "PUT", path: "/api/v1/study/setup", body: body(draft, true), transition: true });
    } catch {
      /* shown in the header */
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="study-page">
      <div className="study-intro">
        <h1>Before you start: the task and the ontology files</h1>
        <p className="lead">You do not need to install or use any particular program. Please read the task, check that you can get the two files, and confirm that you understand how outside inspection works in this study.</p>
      </div>

      <section className="study-card" aria-labelledby="setup-task-h">
        <h2 id="setup-task-h">1 · The task</h2>
        <div className="prose">
          <Paragraphs text={state.instructions} />
          <Paragraphs text={state.setup_instructions} />
        </div>
        <label className="check-row">
          <input type="checkbox" checked={draft.instructions_acknowledged} onChange={() => update({ instructions_acknowledged: !draft.instructions_acknowledged })} />
          <span>I have read the task instructions.</span>
        </label>
      </section>

      <section className="study-card" aria-labelledby="setup-files-h">
        <h2 id="setup-files-h">2 · The two ontology files</h2>
        <p className="muted">{SCOPE_NOTICE}</p>
        <ResourceList resources={state.ontology_resources} />
        <p className="meta">{OPEN_HELP}</p>
        <fieldset className="question">
          <legend className="question-label">Can you get the two files?</legend>
          <p className="meta">Downloading now is optional; this only checks that they are available to you if you want them.</p>
          <div className="option-grid option-grid-column">
            {(
              [
                ["available", "Yes, I can download or open them if I want to"],
                ["needs_help", "I have a problem getting them"],
              ] as [ResourceAccess, string][]
            ).map(([value, label]) => (
              <label key={value} className={draft.resource_access === value ? "option on" : "option"}>
                <input type="radio" name="resource-access" checked={draft.resource_access === value} onChange={() => update({ resource_access: value })} />
                <span>{label}</span>
              </label>
            ))}
          </div>
          {draft.resource_access === "needs_help" && (
            <div className="note note-warn" role="status">
              <span>
                Try the Download buttons again, or a different browser. If it still does not work, use Pause at the top and come back later with your private link: your progress is kept and you will not be moved to a different part of the
                study. You can also contact the study team.
              </span>
              <button type="button" className="btn btn-sm" onClick={() => update({ resource_access: "not_checked" })}>
                I will try again
              </button>
            </div>
          )}
        </fieldset>
      </section>

      <section className="study-card" aria-labelledby="setup-methods-h">
        <h2 id="setup-methods-h">3 · Inspecting the ontologies outside this site is optional</h2>
        <p>{v3 ? METHOD_FREEDOM_V3 : METHOD_FREEDOM}</p>
        <label className="check-row">
          <input type="checkbox" checked={draft.external_inspection_optional_understood} onChange={() => update({ external_inspection_optional_understood: !draft.external_inspection_optional_understood })} />
          <span>I understand that outside inspection is optional, that I can combine and change methods between cases, and that I will be asked about my methods after each case.</span>
        </label>
      </section>

      <section className="study-card" aria-labelledby="setup-familiar-h">
        <h2 id="setup-familiar-h">
          4 · Methods you are familiar with <span className="meta">Optional</span>
        </h2>
        <p className="muted">This only describes your experience. It does not commit you to using anything, it is not taken as what you used in a case, and leaving it empty is fine.</p>
        <fieldset className="question">
          <legend className="sr-only">{familiarQuestion?.label ?? "Methods you are familiar with"}</legend>
          <div className="option-grid option-grid-column">
            {familiarOptions.map(([code, label]) => (
              <label key={code} className={`option${draft.familiar_methods.includes(code) ? " on" : ""}${code === "none" ? " special" : ""}`}>
                <input type="checkbox" checked={draft.familiar_methods.includes(code)} onChange={() => (v3 ? toggleFamiliar(code) : update({ familiar_methods: draft.familiar_methods.includes(code) ? draft.familiar_methods.filter((item) => item !== code) : [...draft.familiar_methods, code] }))} />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </fieldset>
        {v3 && draft.familiar_methods.includes("other_method") && (
          <div className={otherError ? "field invalid" : "field"}>
            <label htmlFor="familiar-other-method">{otherQuestion?.label ?? "Which other methods are familiar to you?"}</label>
            <input
              id="familiar-other-method"
              ref={otherInput}
              className="input"
              maxLength={otherQuestion?.text_max_length ?? 160}
              value={draft.familiar_other_method ?? ""}
              aria-invalid={otherError ? true : undefined}
              aria-describedby={otherError ? "familiar-other-error" : "familiar-other-hint"}
              onChange={(event) => {
                setOtherError(null);
                update({ familiar_other_method: event.target.value });
              }}
            />
            {otherError ? (
              <p className="field-error" id="familiar-other-error">
                {otherError}
              </p>
            ) : (
              <p className="meta" id="familiar-other-hint">
                Short names only, separated by commas if there are several.
              </p>
            )}
          </div>
        )}
        {v3 && state.forms.collection_notice && <p className="meta">{state.forms.collection_notice}</p>}
      </section>

      <div className="study-actions">
        <button type="button" className="btn btn-primary btn-large" disabled={busy} onClick={submit} aria-describedby="setup-status">
          {busy ? "Saving…" : "Continue"}
        </button>
        <span id="setup-status" className={error ? "field-error" : "muted"} role={error ? "alert" : undefined}>
          {error ?? (ready ? "Ready to continue." : "Confirm the task, file access and the optional-inspection note to continue.")}
        </span>
      </div>
    </div>
  );
}

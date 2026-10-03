// DEVELOPMENT PREVIEW ONLY. An in-browser stand-in for the proposed exact-study/2.0
// participant routes (16 B2–B4), used by /preview/participate/ under `next dev` to
// demonstrate the corrected flows before the backend implements them. It is an executable
// statement of the frontend's contract assumptions (mutation envelope, revisions,
// idempotency, draft/commit separation, server-side grading), not a study service: no data
// leaves the browser, nothing is scored, and production builds never include it.

import { V1_EVENT_TYPES, V2_EVENT_TYPES, type AssessmentResponse, type AttemptReceipt, type ConsultationDraftV2, type ConsultationReceiptV2, type PublicAsset, type Question, type RankingResponse, type SetupV2, type Stage, type StudyCase, type StudyState, type TutorialProgress } from "../types";
import { ASSESSMENT_KEYS, gradeResponse, LESSONS, ontologyDocument, TUTORIAL_CANDIDATES, TUTORIAL_DOWNLOADS, TUTORIAL_RESOURCE, TUTORIAL_RESOURCE_ID, TUTORIAL_VERSION, tutorialPublic, ENTITY, ASSESSMENT } from "./tutorialContent";

const STORE = "exact.preview.v2.server";
const SESSION = "preview-session";
const FORM_VERSION = "exact-study-forms/2-preview";
const CASE_RESOURCE = "preview-case-explanations";

type Answers = Record<string, unknown>;
interface Server {
  revision: number;
  stage: Stage;
  pausedFrom: Stage | null;
  consent: { information_version: string; accepted: boolean; acknowledged_at: string } | null;
  setup: SetupV2 | null;
  questionnaires: Record<string, { form_version: string; answers: Answers; submitted: boolean; answer_states: Record<string, "answered" | "skipped" | "not_answered">; saved_at: string }>;
  progress: TutorialProgress;
  caseIndex: number;
  assigned: boolean;
  ranking: RankingResponse | null;
  draft: ConsultationDraftV2 | null;
  previous: ConsultationReceiptV2 | null;
  reports: ConsultationReceiptV2[];
  keys: Record<string, { status: number; body: unknown }>;
  offline: boolean;
}

const CASES: { case_id: string; condition: "explanation" | "ontology_baseline" }[] = [
  { case_id: "preview-case-1", condition: "explanation" },
  { case_id: "preview-case-2", condition: "ontology_baseline" },
];

const order = (q: Question): Question => ({ ...q, option_order: q.options ? Object.keys(q.options) : undefined, row_order: q.matrix ? Object.keys(q.matrix) : undefined });
const q = (id: string, label: string, options: Record<string, string> | null, extra: Partial<Question> = {}): Question => order({ id, label, options, multiple: false, required: true, show_if: null, matrix: null, ...extra });
// Option objects are deliberately built in reverse key order: the UI must follow option_order.
const reversed = (entries: [string, string][]) => Object.fromEntries([...entries].reverse());
const ordered = (entries: [string, string][], question: Question): Question => ({ ...question, options: reversed(entries), option_order: entries.map(([code]) => code) });

const YEARS: [string, string][] = [["none", "None"], ["less_than_1", "Less than 1 year"], ["1_2", "1–2 years"], ["3_4", "3–4 years"], ["5_9", "5–9 years"], ["10_plus", "10 or more years"], ["prefer_not_to_say", "Prefer not to say"]];
const HELPFUL: [string, string][] = [["not_helpful", "Not helpful"], ["slightly", "Slightly helpful"], ["moderately", "Moderately helpful"], ["very", "Very helpful"], ["cannot_judge", "Did not use / cannot judge"]];
const EFFORT: [string, string][] = [["very_low", "Very low"], ["low", "Low"], ["moderate", "Moderate"], ["high", "High"], ["very_high", "Very high"], ["cannot_judge", "Cannot judge"]];
const METHODS: [string, string][] = [
  ["protege", "Protégé"],
  ["other_editor", "Another ontology editor"],
  ["plain_files", "The ontology files directly (for example, in a text editor)"],
  ["other_resource", "Another ontology resource or viewer"],
  ["queries_scripts", "Queries or scripts"],
  ["reasoner", "A reasoner"],
  ["other_method", "Another method"],
];

function forms() {
  const components: [string, string][] = [
    ["original_context", "Original entity definitions and context"],
    ["entity_description", "Generated entity descriptions"],
    ["hierarchy", "Hierarchy browser"],
    ["evidence_table", "Comparison and evidence table"],
    ["evidence_graph", "Evidence graph"],
    ["pair_comparison", "Generated pair comparison"],
  ];
  return {
    version: FORM_VERSION,
    background: [
      ordered(YEARS, q("cs_experience_years", "How much study or professional experience do you have in computer science or computing?", null)),
      ordered(YEARS, q("health_bio_experience_years", "How much study or professional experience do you have in health, medicine or biosciences?", null)),
      q("domain_specialty", "What is your main health or bioscience field, if any? Please do not identify an institution or employer.", null, { required: false }),
    ],
    consultation: [
      ordered([["true", "Yes"], ["false", "No"]], q("consulted_external_ontologies", "For the case you just completed, did you inspect ontology information outside this study interface?", null)),
      ordered(METHODS, q("methods", "Select all methods you actually used. If you used several methods for different candidates, include all of them.", null, { multiple: true, show_if: { consulted_external_ontologies: true } })),
    ],
    final: [
      { ...ordered(HELPFUL, q("component_usefulness", "How helpful was each component for deciding your ranking?", null)), matrix: reversed(components), row_order: components.map(([code]) => code) },
      { ...ordered(EFFORT, q("mental_effort", "How much mental effort did each approach require?", null)), matrix: reversed([["explanation", "Explanation interface"], ["ontology_baseline", "Candidates and scores with separate ontology inspection"]]), row_order: ["explanation", "ontology_baseline"] },
      q("comments", "What was confusing, missing or difficult, and what would you change? Please avoid identifying details.", null, { required: false }),
    ],
    experience_note: "Years mean approximate combined study/work experience; count overlapping years once.",
  };
}

function fresh(): Server {
  return {
    revision: 0,
    stage: "welcome",
    pausedFrom: null,
    consent: null,
    setup: null,
    questionnaires: {},
    progress: { tutorial_version: TUTORIAL_VERSION, current_lesson_id: null, completed_requirements: [], practice: {}, assessment_drafts: {}, attempts: [], passed_items: [], outstanding: [], help_opened: 0, completed_at: null },
    caseIndex: 0,
    assigned: false,
    ranking: null,
    draft: null,
    previous: null,
    reports: [],
    keys: {},
    offline: false,
  };
}

function load(): Server {
  try {
    const raw = window.localStorage.getItem(STORE);
    return raw ? { ...fresh(), ...(JSON.parse(raw) as Server) } : fresh();
  } catch {
    return fresh();
  }
}

function persist(server: Server) {
  try {
    window.localStorage.setItem(STORE, JSON.stringify(server));
  } catch {
    /* preview only */
  }
}

export function resetPreview() {
  try {
    window.localStorage.removeItem(STORE);
  } catch {
    /* preview only */
  }
}

export function setPreviewOffline(offline: boolean) {
  const server = load();
  server.offline = offline;
  persist(server);
}

export function previewOffline(): boolean {
  return load().offline;
}

function outstanding(progress: TutorialProgress): string[] {
  const done = new Set(progress.completed_requirements);
  const lessons = LESSONS.filter((lesson) => !lesson.optional && lesson.requirements.some((requirement) => !done.has(requirement.requirement_id))).map((lesson) => lesson.lesson_id);
  const items = ASSESSMENT.filter((item) => !progress.passed_items.includes(item.question_id)).map((item) => item.question_id);
  return [...lessons, ...items];
}

let downloads: PublicAsset[] = [];

function presentation(server: Server) {
  return `preview-presentation-${server.caseIndex + 1}`;
}

function publicState(server: Server): StudyState {
  const inCase = server.stage === "case" || server.stage === "consultation" || (server.stage === "paused" && server.assigned);
  const current = server.assigned && server.caseIndex < CASES.length ? CASES[server.caseIndex] : null;
  return {
    artifact_type: "study_state",
    contract_version: "exact-study/2.0",
    study_revision: "preview-study/2",
    session_id: SESSION,
    revision: server.revision,
    stage: server.stage,
    assignment_id: server.assigned ? "preview-assignment" : null,
    current_case_id: inCase && current ? current.case_id : null,
    current_presentation_id: inCase && current ? presentation(server) : null,
    completed_cases: server.reports.length,
    assigned_case_count: CASES.length,
    ranking: server.ranking,
    consultation: null,
    consent: server.consent,
    setup: server.setup,
    questionnaires: server.questionnaires,
    information_version: "preview-information/1",
    information_text: "[Development preview. The study owner supplies the participant information.]",
    consent_text: "[Development preview. The study owner supplies the consent statement.]",
    instructions:
      "Rank the candidates you consider plausible equivalents of the source concept, with the best first. You may use fewer than five. If none appears equivalent, choose None of these; if you cannot judge, choose Insufficient information. The initial order and scores are suggestions, not answers.",
    setup_instructions: "[Development preview. The study owner supplies any setup notes, for example the supplied file versions and help contacts.]",
    tutorial_steps: [],
    practice_cases: [],
    forms: forms(),
    ontology_resources: downloads,
    synthetic: true,
    gap_recovery: "ask",
    telemetry: { event_types: [...V1_EVENT_TYPES, ...V2_EVENT_TYPES], scopes: ["case"] },
    tutorial: tutorialPublic(downloads),
    tutorial_progress: { ...server.progress, outstanding: outstanding(server.progress) },
    consultation_draft: server.stage === "consultation" ? server.draft : null,
    previous_consultation: server.previous,
    protocol_versions: { setup: "setup/2-preview", tutorial: TUTORIAL_VERSION, forms: FORM_VERSION },
  } as StudyState;
}

function studyCase(server: Server): StudyCase | null {
  const current = CASES[server.caseIndex];
  if (!current || !server.assigned) return null;
  return {
    artifact_type: "study_case",
    study_revision: "preview-study/2",
    case_id: current.case_id,
    presentation_id: presentation(server),
    condition: current.condition,
    source: ENTITY["a.crpc"],
    source_label: "closed crate of round pieces",
    candidates: TUTORIAL_CANDIDATES,
    ontology_resource_ids: downloads.map((asset) => asset.asset_id),
    package_version: "preview",
    explanation_refs: current.condition === "explanation" ? [CASE_RESOURCE] : [],
  };
}

class Reply {
  constructor(public status: number, public body: unknown, public type = "application/json") {}
}
const json = (body: unknown, status = 200) => new Reply(status, body);
const fail = (status: number, detail: string, revision?: number) => new Reply(status, { detail, ...(revision !== undefined ? { current_revision: revision } : {}) });

function mutate(server: Server, body: Record<string, unknown>, apply: () => Reply | null): Reply {
  const key = String(body.idempotency_key ?? "");
  if (key && server.keys[key]) return new Reply(server.keys[key].status, server.keys[key].body);
  if (body.expected_revision !== server.revision) return fail(409, "The study changed in another tab or device.", server.revision);
  const error = apply();
  if (error) return error;
  server.revision += 1;
  const reply = json(publicState(server));
  if (key) server.keys[key] = { status: reply.status, body: reply.body };
  return reply;
}

function route(method: string, path: string, body: Record<string, unknown>): Reply {
  const server = load();
  const done = (reply: Reply) => {
    persist(server);
    return reply;
  };
  const stage = server.stage;
  if (method === "GET" && path === "/api/v1/study/state") return json(publicState(server));
  if (method === "POST" && path === "/api/v1/study/session") return json(publicState(server));
  if (method === "POST" && (path === "/api/v1/study/events" || path === "/api/v1/study/timing")) {
    const events = (body.events as { event_id: string }[] | undefined) ?? [];
    return json(path.endsWith("events") ? { acknowledged_event_ids: events.map((event) => event.event_id), sequence_gaps: [] } : { acknowledged_segment_id: body.segment_id });
  }
  if (method === "GET" && path === "/api/v1/study/cases/current") {
    const current = studyCase(server);
    return current && (stage === "case" || stage === "consultation") ? json(current) : fail(404, "No current case");
  }
  const resource = /^\/api\/v1\/study\/resources\/([^/]+)$/.exec(path);
  if (method === "GET" && resource) {
    const id = decodeURIComponent(resource[1]);
    if (id === TUTORIAL_RESOURCE_ID) return json(TUTORIAL_RESOURCE);
    if (id === CASE_RESOURCE) return CASES[server.caseIndex]?.condition === "explanation" && stage === "case" ? json(TUTORIAL_RESOURCE) : fail(403, "Explanation resources are not available in this condition");
    const download = TUTORIAL_DOWNLOADS.find((asset) => asset.asset_id === id);
    if (download) return new Reply(200, ontologyDocument(download.ontology), "application/owl-functional");
    return fail(404, "Unknown resource");
  }
  const at = (expected: Stage | Stage[]) => ((Array.isArray(expected) ? expected : [expected]).includes(stage) ? null : fail(409, `Not available at the ${stage} step`, server.revision));
  if (method === "PUT" && path === "/api/v1/study/consent")
    return done(
      mutate(server, body, () => {
        server.consent = { information_version: String(body.information_version), accepted: Boolean(body.accepted), acknowledged_at: new Date().toISOString() };
        server.stage = body.accepted ? "setup" : "closed";
        return null;
      }),
    );
  if (method === "PUT" && path === "/api/v1/study/setup")
    return done(
      mutate(server, body, () => {
        const blocked = at("setup");
        if (blocked) return blocked;
        const next: SetupV2 = {
          setup_version: String(body.setup_version),
          instructions_acknowledged: Boolean(body.instructions_acknowledged),
          external_inspection_optional_understood: Boolean(body.external_inspection_optional_understood),
          resource_access: (body.resource_access as SetupV2["resource_access"]) ?? "not_checked",
          familiar_methods: (body.familiar_methods as string[]) ?? [],
          submitted: Boolean(body.submitted),
          saved_at: new Date().toISOString(),
        };
        if (next.submitted && !(next.instructions_acknowledged && next.external_inspection_optional_understood && next.resource_access === "available")) return fail(422, "Setup needs both acknowledgements and available resources");
        server.setup = next;
        if (next.submitted) server.stage = "background";
        return null;
      }),
    );
  const questionnaire = /^\/api\/v1\/study\/questionnaires\/(background|final)$/.exec(path);
  if (method === "PUT" && questionnaire)
    return done(
      mutate(server, body, () => {
        const form = questionnaire[1];
        const blocked = at(form as Stage);
        if (blocked) return blocked;
        const answers = (body.answers as Answers) ?? {};
        const questions = forms()[form as "background" | "final"];
        const states: Record<string, "answered" | "skipped" | "not_answered"> = {};
        for (const question of questions) {
          const value = answers[question.id];
          const empty = value === undefined || value === null || value === "" || (Array.isArray(value) && !value.length);
          if (body.submitted && question.required && empty) return fail(422, `Required answer: ${question.id}`);
          states[question.id] = empty ? "not_answered" : "answered";
        }
        server.questionnaires[form] = { form_version: FORM_VERSION, answers, submitted: Boolean(body.submitted), answer_states: states, saved_at: new Date().toISOString() };
        if (body.submitted && form === "background") server.stage = "tutorial";
        return null;
      }),
    );
  if (method === "PUT" && path === "/api/v1/study/tutorial/progress")
    return done(
      mutate(server, body, () => {
        if (body.tutorial_version !== TUTORIAL_VERSION) return fail(409, "Tutorial version mismatch", server.revision);
        const known = new Set(LESSONS.flatMap((lesson) => lesson.requirements.map((requirement) => requirement.requirement_id)));
        const completed = (body.completed_requirements as string[] | undefined) ?? [];
        if (completed.some((id) => !known.has(id))) return fail(422, "Unknown tutorial requirement");
        server.progress.completed_requirements = Array.from(new Set([...server.progress.completed_requirements, ...completed]));
        if ("current_lesson_id" in body) server.progress.current_lesson_id = (body.current_lesson_id as string | null) ?? server.progress.current_lesson_id;
        const practice = body.practice as { key: string; response_type: RankingResponse["response_type"]; ranked_candidate_ids: string[] } | undefined;
        if (practice) server.progress.practice[practice.key] = { response_type: practice.response_type, ranked_candidate_ids: practice.ranked_candidate_ids };
        const draft = body.assessment_draft as { question_id: string; response: AssessmentResponse } | undefined;
        if (draft) server.progress.assessment_drafts[draft.question_id] = draft.response;
        if (body.help_opened) server.progress.help_opened += 1;
        return null;
      }),
    );
  if (method === "POST" && path === "/api/v1/study/tutorial/assessment")
    return done(
      mutate(server, body, () => {
        const questionId = String(body.question_id);
        const key = ASSESSMENT_KEYS[questionId];
        if (!key) return fail(422, "Unknown question");
        const response = body.response as AssessmentResponse;
        const correct = gradeResponse(questionId, response);
        const attempt: AttemptReceipt = {
          attempt_id: String(body.attempt_id),
          question_id: questionId,
          response,
          correct,
          feedback: correct ? key.right : key.wrong,
          revisit_lesson_id: correct ? null : ASSESSMENT.find((item) => item.question_id === questionId)?.lesson_id ?? null,
          submitted_at: new Date().toISOString(),
        };
        server.progress.attempts.push(attempt);
        if (correct && !server.progress.passed_items.includes(questionId)) server.progress.passed_items.push(questionId);
        return null;
      }),
    );
  if (method === "POST" && path === "/api/v1/study/tutorial/complete")
    return done(
      mutate(server, body, () => {
        const blocked = at("tutorial");
        if (blocked) return blocked;
        if (outstanding(server.progress).length) return fail(422, "The tutorial is not complete yet");
        server.progress.completed_at = new Date().toISOString();
        server.assigned = true;
        server.caseIndex = 0;
        server.stage = "case";
        return null;
      }),
    );
  const caseRoute = /^\/api\/v1\/study\/cases\/([^/]+)\/(draft|submit|consultation|consultation\/draft)$/.exec(path);
  if (caseRoute)
    return done(
      mutate(server, body, () => {
        const [, caseId, action] = caseRoute;
        const current = CASES[server.caseIndex];
        if (!current || current.case_id !== decodeURIComponent(caseId)) return fail(409, "Not the current case", server.revision);
        if (action === "draft" || action === "submit") {
          const blocked = at("case");
          if (blocked) return blocked;
          if (body.presentation_id !== presentation(server)) return fail(409, "Not the current presentation", server.revision);
          const submitted = action === "submit";
          if (submitted && !body.response_type) return fail(422, "An empty answer cannot be submitted");
          server.ranking = {
            case_id: current.case_id,
            presentation_id: presentation(server),
            response_type: (body.response_type as RankingResponse["response_type"]) ?? null,
            ranked_candidate_ids: (body.ranked_candidate_ids as string[]) ?? [],
            revision: server.revision + 1,
            workflow_state: submitted ? "submitted" : "draft",
            saved_at: new Date().toISOString(),
            submitted_at: submitted ? new Date().toISOString() : null,
          };
          if (submitted) {
            server.stage = "consultation";
            server.draft = null;
          }
          return null;
        }
        const blocked = at("consultation");
        if (blocked) return blocked;
        if (body.presentation_id !== presentation(server)) return fail(409, "Not the current presentation", server.revision);
        const draft: ConsultationDraftV2 = {
          presentation_id: presentation(server),
          form_version: FORM_VERSION,
          consulted_external_ontologies: (body.consulted_external_ontologies as boolean | null) ?? null,
          methods: (body.methods as ConsultationDraftV2["methods"]) ?? [],
          other_editor: (body.other_editor as string | null) ?? null,
          other_resource: (body.other_resource as string | null) ?? null,
          other_method: (body.other_method as string | null) ?? null,
          resource_scope: (body.resource_scope as ConsultationDraftV2["resource_scope"]) ?? null,
          saved_at: new Date().toISOString(),
        };
        if (action === "consultation/draft") {
          server.draft = draft;
          return null;
        }
        if (draft.consulted_external_ontologies === null) return fail(422, "Yes or No is required");
        if (draft.consulted_external_ontologies !== draft.methods.length > 0) return fail(422, "Yes requires a method; No requires no methods");
        const receipt: ConsultationReceiptV2 = { ...draft, case_id: current.case_id, submitted_at: new Date().toISOString() };
        server.reports.push(receipt);
        server.previous = receipt;
        server.draft = null;
        server.ranking = null;
        server.caseIndex += 1;
        server.stage = server.caseIndex < CASES.length ? "case" : "final";
        return null;
      }),
    );
  if (method === "POST" && path === "/api/v1/study/pause")
    return done(
      mutate(server, body, () => {
        server.pausedFrom = server.stage;
        server.stage = "paused";
        return null;
      }),
    );
  if (method === "POST" && path === "/api/v1/study/resume")
    return done(
      mutate(server, body, () => {
        server.stage = server.pausedFrom ?? "welcome";
        server.pausedFrom = null;
        return null;
      }),
    );
  if (method === "POST" && path === "/api/v1/study/complete")
    return done(
      mutate(server, body, () => {
        const blocked = at("final");
        if (blocked) return blocked;
        server.stage = "completed";
        return null;
      }),
    );
  return fail(404, "Unknown preview route");
}

async function digest(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash), (value) => value.toString(16).padStart(2, "0")).join("");
}

let installs = 0;
let patched: Promise<() => void> | null = null;

/** Route same-origin /api/v1/study requests to the preview service; returns an undo function.
 * Reference-counted with one shared patch, so React's development double-mount cannot
 * restore the real fetch early or patch twice. */
export async function installFixtureServer(): Promise<() => void> {
  installs += 1;
  patched = patched ?? patch();
  await patched;
  let released = false;
  return () => {
    if (released) return;
    released = true;
    installs -= 1;
    if (installs === 0 && patched) {
      const current = patched;
      patched = null;
      void current.then((undo) => undo());
    }
  };
}

async function patch(): Promise<() => void> {
  downloads = await Promise.all(
    TUTORIAL_DOWNLOADS.map(async ({ ontology, ...asset }) => {
      const text = ontologyDocument(ontology);
      return { ...asset, sha256: await digest(text), size_bytes: new TextEncoder().encode(text).length } as PublicAsset;
    }),
  );
  const original = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, window.location.href);
    if (url.origin !== window.location.origin || !url.pathname.startsWith("/api/v1/study/")) return original(input, init);
    if (load().offline) throw new TypeError("Preview: simulated network failure");
    const method = (init?.method ?? "GET").toUpperCase();
    let body: Record<string, unknown> = {};
    try {
      body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {};
    } catch {
      body = {};
    }
    await new Promise((resolve) => setTimeout(resolve, 120));
    const reply = route(method, url.pathname, body);
    const text = reply.type === "application/json" ? JSON.stringify(reply.body) : String(reply.body);
    return new Response(text, { status: reply.status, headers: { "Content-Type": reply.type, ...(reply.type !== "application/json" ? { "Content-Disposition": "attachment" } : {}) } });
  };
  // Download links navigate rather than fetch; serve them from the preview as files.
  const onClick = async (event: MouseEvent) => {
    const anchor = (event.target as HTMLElement | null)?.closest?.("a[download]") as HTMLAnchorElement | null;
    if (!anchor) return;
    const url = new URL(anchor.href, window.location.href);
    if (url.origin !== window.location.origin || !url.pathname.startsWith("/api/v1/study/resources/")) return;
    event.preventDefault();
    const response = await window.fetch(url.pathname);
    if (!response.ok) return;
    const blob = new Blob([await response.text()], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${decodeURIComponent(url.pathname.split("/").pop() ?? "ontology")}.ofn`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };
  document.addEventListener("click", onClick, true);
  return () => {
    window.fetch = original;
    document.removeEventListener("click", onClick, true);
  };
}

"use client";

// Minimal researcher administration over the authenticated admin API. The researcher token
// is held in memory for this tab only and sent as a bearer header; it is never stored.
// This page never sends invitations to anyone: links are shown once for you to distribute.

import { useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { IconCopy, IconDownload, IconWarning } from "@/components/common/Icons";
import { TextSizeControl, ThemeControl } from "@/components/shell/Preferences";
import { ApiError } from "@/lib/api";
import { readStored, writeStored } from "@/lib/prefs";

const STAGE_NAMES: [string, string][] = [
  ["welcome", "Welcome"],
  ["setup", "Setup"],
  ["background", "Background"],
  ["practice", "Practice"],
  ["case", "Ranking a case"],
  ["consultation", "Consultation question"],
  ["final", "Final feedback"],
  ["paused", "Paused"],
  ["completed", "Completed"],
  ["closed", "Closed or declined"],
];

async function adminRequest<T>(token: string, method: "GET" | "POST", path: string, body?: unknown): Promise<{ data: T; blob?: Blob; filename?: string }> {
  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers: { Authorization: `Bearer ${token}`, ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      credentials: "omit",
    });
  } catch {
    throw new ApiError(0, "network_unreachable", "The study service could not be reached", true);
  }
  if (!response.ok) {
    let detail = "Request failed";
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : Array.isArray(payload.detail) ? payload.detail.map((item: { msg?: string }) => item.msg).join("; ") : detail;
    } catch {
      /* keep generic */
    }
    throw new ApiError(response.status, `http_${response.status}`, response.status === 401 ? "The researcher token was not accepted." : detail);
  }
  const type = response.headers.get("content-type") ?? "";
  if (type.includes("application/json")) return { data: (await response.json()) as T };
  const disposition = response.headers.get("content-disposition") ?? "";
  const match = /filename="([^"]+)"/.exec(disposition);
  return { data: undefined as T, blob: await response.blob(), filename: match?.[1] };
}

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function AdminApp() {
  const [token, setToken] = useState("");
  const [tokenInput, setTokenInput] = useState("");
  const [revision, setRevision] = useState(() => (typeof window === "undefined" ? "" : readStored<string>("exact.admin.revision", "")));
  const [progress, setProgress] = useState<Record<string, number> | null>(null);
  const [links, setLinks] = useState<{ session_id: string; url: string }[]>([]);
  const [count, setCount] = useState(10);
  const [test, setTest] = useState(true);
  const [sessionId, setSessionId] = useState("");
  const [reissued, setReissued] = useState<string | null>(null);
  const [format, setFormat] = useState<"json" | "csv">("json");
  const [includeTest, setIncludeTest] = useState(false);
  const [includeKeys, setIncludeKeys] = useState(false);
  const [message, setMessage] = useState<{ tone: "ok" | "bad"; text: string } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmClose, setConfirmClose] = useState(false);

  const run = async <T,>(label: string, action: () => Promise<T>): Promise<T | undefined> => {
    setBusy(label);
    setMessage(null);
    try {
      return await action();
    } catch (error) {
      setMessage({ tone: "bad", text: error instanceof ApiError ? error.message : "The action failed." });
      if (error instanceof ApiError && error.status === 401) setToken("");
      return undefined;
    } finally {
      setBusy(null);
    }
  };
  const rev = encodeURIComponent(revision.trim());

  if (!token) {
    return (
      <div className="admin-root">
        <header className="admin-header">
          <span className="study-title">Study administration</span>
          <span className="pill">Researcher only</span>
        </header>
        <main className="admin-page" id="main">
          <form
            className="study-card admin-signin"
            onSubmit={(event) => {
              event.preventDefault();
              if (tokenInput.trim().length >= 32) {
                setToken(tokenInput.trim());
                setTokenInput("");
              } else setMessage({ tone: "bad", text: "Researcher tokens are at least 32 characters." });
            }}
          >
            <h1 className="study-card-title">Sign in with the researcher token</h1>
            <p className="muted">The token stays in this tab&apos;s memory and is cleared when you close or reload it. Participants never use this page.</p>
            <div className="field">
              <label htmlFor="token">Researcher token</label>
              <input id="token" type="password" autoComplete="off" className="input" value={tokenInput} onChange={(event) => setTokenInput(event.target.value)} />
            </div>
            {message && <p className="note note-bad">{message.text}</p>}
            <button type="submit" className="btn btn-primary">
              Continue
            </button>
          </form>
        </main>
      </div>
    );
  }

  return (
    <div className="admin-root">
      <header className="admin-header">
        <span className="study-title">Study administration</span>
        <span className="pill">Researcher only</span>
        <span className="study-header-spacer" />
        <TextSizeControl compact />
        <ThemeControl />
        <button type="button" className="btn btn-sm" onClick={() => setToken("")}>
          Sign out
        </button>
      </header>
      <main className="admin-page" id="main">
        {message && (
          <p className={message.tone === "ok" ? "note note-ok" : "note note-bad"} role={message.tone === "bad" ? "alert" : "status"}>
            {message.text}
          </p>
        )}
        <section className="study-card admin-row">
          <div className="field admin-revision">
            <label htmlFor="revision">Study revision</label>
            <input
              id="revision"
              className="input mono"
              value={revision}
              onChange={(event) => {
                setRevision(event.target.value);
                writeStored("exact.admin.revision", event.target.value);
              }}
              placeholder="as published"
            />
          </div>
          <label className="btn file-label">
            Publish a study revision…
            <input
              type="file"
              accept="application/json,.json"
              className="sr-only"
              onChange={async (event) => {
                const file = event.target.files?.[0];
                event.target.value = "";
                if (!file) return;
                let publication: { definition?: { study_revision?: string } };
                try {
                  publication = JSON.parse(await file.text());
                } catch {
                  setMessage({ tone: "bad", text: "The file is not valid JSON." });
                  return;
                }
                const result = await run("publish", () => adminRequest<Record<string, unknown>>(token, "POST", "/api/v1/admin/studies", publication));
                if (result) {
                  const published = publication.definition?.study_revision ?? "";
                  setRevision(published);
                  writeStored("exact.admin.revision", published);
                  setMessage({ tone: "ok", text: `Published and frozen: ${published}. Every case, resource and key was validated.` });
                }
              }}
            />
          </label>
          <button type="button" className="btn btn-danger" disabled={!revision.trim() || busy !== null} onClick={() => setConfirmClose(true)}>
            Close study
          </button>
        </section>

        <div className="admin-grid">
          <section className="study-card" aria-labelledby="progress-h">
            <div className="section-head">
              <h2 id="progress-h">Progress</h2>
              <span className="meta">Sessions by current step</span>
              <span className="study-header-spacer" />
              <button
                type="button"
                className="btn btn-sm"
                disabled={!revision.trim() || busy !== null}
                onClick={async () => {
                  const result = await run("progress", () => adminRequest<{ counts: Record<string, number> }>(token, "GET", `/api/v1/admin/studies/${rev}/progress`));
                  if (result) setProgress(result.data.counts);
                }}
              >
                {progress ? "Refresh" : "Load progress"}
              </button>
            </div>
            {progress ? (
              <table className="admin-table">
                <thead>
                  <tr>
                    <th scope="col">Step</th>
                    <th scope="col">Participants</th>
                    <th scope="col">Test</th>
                  </tr>
                </thead>
                <tbody>
                  {STAGE_NAMES.map(([code, name]) => (
                    <tr key={code}>
                      <th scope="row">{name}</th>
                      <td>{progress[`participant:${code}`] ?? 0}</td>
                      <td>{progress[`test:${code}`] ?? 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">Counts only. No answers, secrets or free text appear here.</p>
            )}
          </section>

          <div className="admin-column">
            <section className="study-card" aria-labelledby="invite-h">
              <h2 id="invite-h">Invitation links</h2>
              <div className="admin-inline">
                <div className="field">
                  <label htmlFor="count">How many</label>
                  <input id="count" type="number" min={1} max={1000} className="input admin-number" value={count} onChange={(event) => setCount(Math.max(1, Math.min(1000, Number(event.target.value) || 1)))} />
                </div>
                <label className="check-row">
                  <input type="checkbox" checked={test} onChange={(event) => setTest(event.target.checked)} />
                  <span>Test sessions (preview, excluded from analysis)</span>
                </label>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!revision.trim() || busy !== null}
                  onClick={async () => {
                    const result = await run("invite", () => adminRequest<{ invitations: { session_id: string; invitation: string }[] }>(token, "POST", `/api/v1/admin/studies/${rev}/invitations`, { count, test }));
                    if (result) setLinks(result.data.invitations.map((item) => ({ session_id: item.session_id, url: `${window.location.origin}${item.invitation}` })));
                  }}
                >
                  Create links
                </button>
              </div>
              <p className="note note-warn">
                <IconWarning /> Links are shown once. Copy or download them now and send them yourself; this page never emails anyone and keeps no names.
              </p>
              {links.length > 0 && (
                <>
                  <div className="admin-inline">
                    <button type="button" className="btn btn-sm" onClick={() => navigator.clipboard.writeText(links.map((link) => link.url).join("\n"))}>
                      <IconCopy /> Copy all links
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => download(new Blob([["session_id,invitation_link", ...links.map((link) => `${link.session_id},${link.url}`)].join("\n")], { type: "text/csv" }), `invitations-${Date.now()}.csv`)}
                    >
                      <IconDownload /> Download CSV
                    </button>
                    <a className="btn btn-sm" href={links[0].url} target="_blank" rel="noreferrer noopener">
                      Open first link to preview
                    </a>
                  </div>
                  <ol className="link-list">
                    {links.map((link) => (
                      <li key={link.session_id}>
                        <span className="iri">{link.session_id}</span>
                        <span className="iri">{link.url}</span>
                      </li>
                    ))}
                  </ol>
                </>
              )}
            </section>

            <section className="study-card" aria-labelledby="session-h">
              <h2 id="session-h">Replace or revoke one link</h2>
              <div className="field">
                <label htmlFor="session-id">Session ID</label>
                <input id="session-id" className="input mono" value={sessionId} onChange={(event) => setSessionId(event.target.value)} placeholder="From the issuance list" />
              </div>
              <div className="admin-inline">
                <button
                  type="button"
                  className="btn"
                  disabled={!sessionId.trim() || busy !== null}
                  onClick={async () => {
                    const result = await run("reissue", () => adminRequest<{ invitation?: string }>(token, "POST", `/api/v1/admin/invitations/${encodeURIComponent(sessionId.trim())}/reissue`));
                    if (result?.data.invitation) setReissued(`${window.location.origin}${result.data.invitation}`);
                  }}
                >
                  Issue a replacement link
                </button>
                <button
                  type="button"
                  className="btn btn-danger"
                  disabled={!sessionId.trim() || busy !== null}
                  onClick={async () => {
                    const result = await run("revoke", () => adminRequest(token, "POST", `/api/v1/admin/invitations/${encodeURIComponent(sessionId.trim())}/revoke`));
                    if (result) setMessage({ tone: "ok", text: "Link revoked. The old link and any browser signed in with it no longer work." });
                  }}
                >
                  Revoke
                </button>
              </div>
              {reissued && (
                <p className="note note-info">
                  <span>
                    Replacement link (shown once): <span className="iri">{reissued}</span>
                  </span>
                </p>
              )}
              <p className="meta">Both keep the participant&apos;s answers. The old link and any browser signed in with it stop working.</p>
            </section>
          </div>
        </div>

        <section className="study-card" aria-labelledby="export-h">
          <h2 id="export-h">Export analysis data</h2>
          <div className="admin-inline">
            <div className="segmented" role="radiogroup" aria-label="Format">
              {(["json", "csv"] as const).map((item) => (
                <button key={item} type="button" role="radio" aria-checked={format === item} className={format === item ? "segmented-on" : ""} onClick={() => setFormat(item)}>
                  {item === "json" ? "JSON" : "CSV archive"}
                </button>
              ))}
            </div>
            <label className="check-row">
              <input type="checkbox" checked={includeTest} onChange={(event) => setIncludeTest(event.target.checked)} />
              <span>Include test sessions</span>
            </label>
            <label className="check-row">
              <input type="checkbox" checked={includeKeys} onChange={(event) => setIncludeKeys(event.target.checked)} />
              <span>Include adjudicated answer keys and scoring</span>
            </label>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!revision.trim() || busy !== null}
              onClick={async () => {
                const query = new URLSearchParams({ include_test: String(includeTest), include_keys: String(includeKeys), format }).toString();
                const result = await run("export", () => adminRequest<Record<string, unknown>>(token, "POST", `/api/v1/admin/studies/${rev}/exports?${query}`));
                if (!result) return;
                if (result.blob) download(result.blob, result.filename ?? `export-${Date.now()}.zip`);
                else {
                  const manifest = (result.data.manifest ?? {}) as { export_id?: string };
                  download(new Blob([JSON.stringify(result.data, null, 2)], { type: "application/json" }), `${manifest.export_id ?? `export-${Date.now()}`}.json`);
                }
                setMessage({ tone: "ok", text: "Export created and downloaded. It is frozen with its manifest on the server." });
              }}
            >
              Create export
            </button>
          </div>
          <p className="muted">Exports never contain invitation links, session cookies or free-text comments. Answer keys are included only when you ask for them.</p>
        </section>
      </main>
      {confirmClose && (
        <Dialog title="Close this study?" onClose={() => setConfirmClose(false)}>
          <p>Closing stops all participation for {revision}. Existing answers are kept. This cannot be undone from this page.</p>
          <div className="study-actions">
            <button
              type="button"
              className="btn btn-danger"
              onClick={async () => {
                setConfirmClose(false);
                const result = await run("close", () => adminRequest(token, "POST", `/api/v1/admin/studies/${rev}/close`));
                if (result) setMessage({ tone: "ok", text: `${revision} is closed.` });
              }}
            >
              Close the study
            </button>
            <button type="button" className="btn" onClick={() => setConfirmClose(false)}>
              Keep it open
            </button>
          </div>
        </Dialog>
      )}
    </div>
  );
}

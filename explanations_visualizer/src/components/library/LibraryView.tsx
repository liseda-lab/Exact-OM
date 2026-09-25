"use client";

// Local bundle library and import. Uploads go to the service on this computer; importing
// validates inert data only and never runs matching or text generation. A failed import
// leaves the currently open bundle untouched.

import { useCallback, useEffect, useRef, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconCheck, IconSpinner, IconUpload, IconWarning } from "@/components/common/Icons";
import { useExplore } from "@/components/explore/ExploreContext";
import { ApiError, getJson, request, sendJson } from "@/lib/api";
import { shortHash } from "@/lib/iri";
import type { BundleListItem, ImportJob } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { formatBytes, readBundlePreview, type BundlePreview } from "@/lib/zipManifest";

type Phase = "idle" | "uploading" | "validating" | "done" | "failed" | "cancelled";

const IMPORT_ERRORS: Record<string, string> = {
  invalid_archive: "The file is not a ZIP archive. Export a bundle with `exact-inspect export`.",
  unsafe_archive: "The archive contains duplicate, linked, encrypted or special entries, so it was refused.",
  corrupt_archive: "The archive is corrupt or incomplete. Try exporting or copying it again.",
  payload_too_large: "The bundle is larger than this service's import limit.",
  unbound_artifact: "The archive contains files that its manifest does not list, so it cannot be trusted as a bundle.",
  invalid_identity: "The bundle's identity does not match its manifest.",
  corrupt_artifact: "A file in the bundle does not match its recorded checksum.",
  duplicate_artifact: "The manifest lists the same file twice.",
  unsafe_path: "The manifest refers to a path outside the bundle.",
  import_cancelled: "The import was cancelled.",
  import_conflict: "This import was already started in another tab.",
  library_unavailable: "This service was started without a local bundle library. Restart it with --library-dir.",
  origin_denied: "The request came from another site and was refused.",
};

function uploadWithProgress(url: string, file: File, onProgress: (sent: number) => void, register: (xhr: XMLHttpRequest) => void): Promise<ImportJob> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    register(xhr);
    xhr.open("POST", url);
    xhr.setRequestHeader("Content-Type", "application/zip");
    xhr.upload.onprogress = (event) => onProgress(event.loaded);
    xhr.onload = () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = {};
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as unknown as ImportJob);
      else reject(new ApiError(xhr.status, String(body.code ?? `http_${xhr.status}`), String(body.message ?? "Import failed"), Boolean(body.retryable)));
    };
    xhr.onerror = () => reject(new ApiError(0, "network_unreachable", "The local service could not be reached", true));
    xhr.onabort = () => reject(new ApiError(0, "import_cancelled", "The import was cancelled"));
    xhr.send(file);
  });
}

export function LibraryView() {
  const state = useExplore();
  const bundles = useAsync<BundleListItem[]>("bundles", (signal) => getJson<BundleListItem[]>("/api/v1/bundles", undefined, signal));
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<BundlePreview | null | "unreadable">(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [sent, setSent] = useState(0);
  const [job, setJob] = useState<ImportJob | null>(null);
  const [error, setError] = useState<{ file: string; message: string } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [selecting, setSelecting] = useState<string | null>(null);
  const [selectError, setSelectError] = useState<unknown>(null);
  const xhr = useRef<XMLHttpRequest | null>(null);
  const poll = useRef<number | null>(null);

  const stopPolling = () => {
    if (poll.current) window.clearInterval(poll.current);
    poll.current = null;
  };
  useEffect(() => stopPolling, []);

  const choose = useCallback(async (next: File) => {
    setFile(next);
    setError(null);
    setPhase("idle");
    setJob(null);
    setSent(0);
    setPreview(null);
    try {
      setPreview((await readBundlePreview(next)) ?? "unreadable");
    } catch {
      setPreview("unreadable");
    }
  }, []);

  const start = async () => {
    if (!file) return;
    setError(null);
    setPhase("uploading");
    setSent(0);
    let created: ImportJob;
    try {
      created = await sendJson<ImportJob>("POST", "/api/v1/bundles/import-jobs");
    } catch (err) {
      setPhase("failed");
      setError({ file: file.name, message: err instanceof ApiError ? IMPORT_ERRORS[err.code] ?? err.message : "The import could not start." });
      return;
    }
    setJob(created);
    poll.current = window.setInterval(async () => {
      try {
        const current = await getJson<ImportJob>(`/api/v1/bundles/import-jobs/${created.job_id}`);
        setJob(current);
        if (current.status === "validating") setPhase("validating");
      } catch {
        /* the upload request reports the final outcome */
      }
    }, 1000);
    try {
      const result = await uploadWithProgress(
        `/api/v1/bundles/import?job_id=${created.job_id}`,
        file,
        (loaded) => {
          setSent(loaded);
          if (loaded >= file.size) setPhase("validating");
        },
        (request) => {
          xhr.current = request;
        },
      );
      stopPolling();
      setJob(result);
      setPhase("done");
      bundles.reload();
    } catch (err) {
      stopPolling();
      const cancelled = err instanceof ApiError && err.code === "import_cancelled";
      setPhase(cancelled ? "cancelled" : "failed");
      setError(cancelled ? null : { file: file.name, message: err instanceof ApiError ? IMPORT_ERRORS[err.code] ?? err.message : "The import failed." });
    } finally {
      xhr.current = null;
    }
  };

  const cancel = async () => {
    if (job) {
      try {
        await request(`/api/v1/bundles/import-jobs/${job.job_id}`, { method: "DELETE" });
      } catch {
        /* abort below still stops the upload */
      }
    }
    xhr.current?.abort();
  };

  const open = async (packageId: string) => {
    setSelecting(packageId);
    setSelectError(null);
    try {
      await sendJson("POST", `/api/v1/bundles/${encodeURIComponent(packageId)}/select`);
      window.location.assign("/");
    } catch (err) {
      setSelectError(err);
      setSelecting(null);
    }
  };

  const busy = phase === "uploading" || phase === "validating";
  const percent = file && file.size ? Math.min(100, Math.round((sent / file.size) * 100)) : 0;
  const current = state.health?.package_id ?? null;

  return (
    <div className="library-page" id="main" tabIndex={-1}>
      <div className="library-import">
        <div className="page-intro">
          <h1 className="page-title">Import an inspection bundle</h1>
          <p className="muted">The file goes to the Exact service running on this computer. Nothing is uploaded to the internet, and importing never runs matching or text generation.</p>
        </div>
        <div
          className={dragging ? "dropzone dragging" : "dropzone"}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const dropped = event.dataTransfer.files?.[0];
            if (dropped && !busy) choose(dropped);
          }}
        >
          <IconUpload className="dropzone-icon" />
          <span className="dropzone-title">Drop a bundle .zip here</span>
          <span className="meta">
            Created by <code>exact-inspect export</code>
          </span>
          <label className="btn btn-primary file-label">
            Choose file
            <input
              type="file"
              accept=".zip,application/zip"
              className="sr-only"
              disabled={busy}
              onChange={(event) => {
                const chosen = event.target.files?.[0];
                if (chosen) choose(chosen);
                event.target.value = "";
              }}
            />
          </label>
        </div>

        {file && (
          <section className="card import-card" aria-labelledby="import-h">
            <div className="import-head">
              <h2 id="import-h">{file.name}</h2>
              <span className="meta">{formatBytes(file.size)}</span>
            </div>
            {preview === "unreadable" && <p className="note">No bundle manifest could be read from this file before upload. The service will still validate it.</p>}
            {preview && preview !== "unreadable" && (
              <dl className="kv kv-grid">
                <div>
                  <dt>Bundle</dt>
                  <dd className="iri">{preview.package_id ? shortHash(preview.package_id, 16) : "not declared"}</dd>
                </div>
                <div>
                  <dt>Contract</dt>
                  <dd>{preview.contract_version ?? "not declared"}</dd>
                </div>
                <div>
                  <dt>Contents</dt>
                  <dd>
                    {preview.ontologies} {preview.ontologies === 1 ? "ontology" : "ontologies"} · {preview.runs} {preview.runs === 1 ? "run" : "runs"} · {preview.explanations} prepared texts
                  </dd>
                </div>
                <div>
                  <dt>Expanded size</dt>
                  <dd>
                    {formatBytes(preview.artifactBytes)} in {preview.artifacts} files
                  </dd>
                </div>
                {preview.audience && (
                  <div>
                    <dt>Audience</dt>
                    <dd>{preview.audience === "development_demo" ? "Approved for development demos" : "Local use"}</dd>
                  </div>
                )}
              </dl>
            )}
            {phase !== "idle" && (
              <>
                <div className="progress" role="progressbar" aria-label="Import progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={phase === "validating" || phase === "done" ? 100 : percent}>
                  <div className="progress-bar" data-width={phase === "validating" || phase === "done" ? 100 : percent} ref={(node) => node?.style.setProperty("width", `${phase === "validating" || phase === "done" ? 100 : percent}%`)} />
                </div>
                <ol className="import-steps">
                  <li className={phase === "uploading" ? "active" : sent >= file.size || phase === "done" || phase === "validating" ? "done" : ""}>
                    {phase === "uploading" ? <IconSpinner className="icon spin" /> : <IconCheck />}
                    {phase === "uploading" ? `Sending to the local service · ${percent}%` : "Received the whole file"}
                  </li>
                  <li className={phase === "validating" ? "active" : phase === "done" ? "done" : ""}>
                    {phase === "validating" ? <IconSpinner className="icon spin" /> : phase === "done" ? <IconCheck /> : <span className="step-dot" />}
                    Checking every file against its checksum
                  </li>
                  <li className={phase === "done" ? "done" : ""}>
                    {phase === "done" ? <IconCheck /> : <span className="step-dot" />}
                    Added to the library
                  </li>
                </ol>
              </>
            )}
            <div className="import-actions">
              {(phase === "idle" || phase === "failed" || phase === "cancelled") && (
                <button type="button" className="btn btn-primary" onClick={start}>
                  {phase === "idle" ? "Import this bundle" : "Try again"}
                </button>
              )}
              {busy && (
                <button type="button" className="btn" onClick={cancel}>
                  Cancel import
                </button>
              )}
              {phase === "done" && job?.package_id && (
                <button type="button" className="btn btn-primary" onClick={() => open(job.package_id!)} disabled={selecting !== null}>
                  Open this bundle
                </button>
              )}
              <span className="meta">{busy ? "The bundle you are using stays open until this one is ready." : phase === "cancelled" ? "Import cancelled. Nothing was added." : ""}</span>
            </div>
          </section>
        )}

        {error && (
          <section className="card import-error" role="alert">
            <h2>
              <IconWarning /> {error.file} was not imported
            </h2>
            <p>{error.message}</p>
            <p className="meta">Your current bundle is unchanged. An older run directory can still be opened with exact-inspect serve --run-dir.</p>
          </section>
        )}
      </div>

      <section className="library-list" aria-labelledby="library-h">
        <div className="section-head">
          <h2 id="library-h">Library</h2>
          <span className="meta">{bundles.data ? `${bundles.data.length} ${bundles.data.length === 1 ? "bundle" : "bundles"} on this computer` : ""}</span>
        </div>
        {bundles.error ? <ErrorNote error={bundles.error} onRetry={bundles.reload} what="Library" /> : null}
        {selectError ? <ErrorNote error={selectError} what="Opening bundle" /> : null}
        {!bundles.data && !bundles.error && <Skeleton lines={3} />}
        {bundles.data && bundles.data.length === 0 && <p className="note">No bundle has been imported yet.</p>}
        {current && bundles.data && !bundles.data.some((item) => item.package_id === current) && (
          <article className="card bundle-card current">
            <span className="pill pill-strong">Open now</span>
            <h3>Bundle started with the service</h3>
            <span className="iri">{current}</span>
            <p className="meta">This bundle was passed on the command line and is not a library copy.</p>
          </article>
        )}
        {bundles.data?.map((item) => {
          const isCurrent = item.package_id === current;
          return (
            <article key={item.package_id} className={isCurrent ? "card bundle-card current" : "card bundle-card"}>
              <div className="bundle-card-head">
                {isCurrent && <span className="pill pill-strong">Open now</span>}
                <h3>{isCurrent ? state.runs[0]?.run_id ?? "Open bundle" : "Bundle"}</h3>
              </div>
              <span className="iri">{item.package_id}</span>
              <ul className="capability-list">
                {Object.entries(item.capabilities).map(([name, status]) => (
                  <li key={name} className={status === "available" ? "status status-ok" : "status"}>
                    {name.replace(/_/g, " ")} · {status.replace(/_/g, " ")}
                  </li>
                ))}
              </ul>
              {isCurrent && state.phase === "ready" && (
                <p className="meta">
                  {Object.values(state.ontologies)
                    .map((meta) => `${meta.completeness.entity_count?.toLocaleString() ?? "?"} entities (${meta.completeness.scope === "root" ? "root only" : "import closure"})`)
                    .join(" · ")}
                </p>
              )}
              {!isCurrent && (
                <button type="button" className="btn" onClick={() => open(item.package_id)} disabled={selecting !== null}>
                  {selecting === item.package_id ? "Opening…" : "Open this bundle"}
                </button>
              )}
            </article>
          );
        })}
      </section>
    </div>
  );
}

"use client";

/** Keep a failed render recoverable without clearing the cookie or saved draft. */
export default function StudyError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return <main className="study-page" id="main">
    <section className="study-card" role="alert">
      <h1>The study page could not be displayed</h1>
      <p>Try opening the page again. Your saved answers will be restored.</p>
      <div className="study-actions">
        <button className="btn" type="button" onClick={reset}>Try again</button>
        <button className="btn btn-primary" type="button" onClick={() => window.location.reload()}>Reload study</button>
      </div>
      <p>If this keeps happening, tell the study team which step you had reached.</p>
    </section>
  </main>;
}

"use client";

import { IconWarning } from "@/components/common/Icons";
import { ApiError, describeError } from "@/lib/api";

export function ErrorNote({ error, onRetry, what }: { error: unknown; onRetry?: () => void; what?: string }) {
  const retryable = !(error instanceof ApiError) || error.retryable || error.status === 0 || error.code === "stale_cursor";
  return (
    <div className="note note-bad" role="alert">
      <IconWarning />
      <div className="error-body">
        <span>
          {what ? <strong>{what}: </strong> : null}
          {describeError(error)}
        </span>
        {onRetry && retryable && (
          <button type="button" className="btn btn-sm" onClick={onRetry}>
            {error instanceof ApiError && error.code === "stale_cursor" ? "Reload list" : "Try again"}
          </button>
        )}
      </div>
    </div>
  );
}

export function Skeleton({ lines = 3, title = true }: { lines?: number; title?: boolean }) {
  return (
    <div className="skeleton-block" aria-hidden="true">
      {title && <span className="skeleton skeleton-title" />}
      {Array.from({ length: lines }, (_, index) => (
        <span key={index} className={`skeleton skeleton-line skeleton-line-${index % 3}`} />
      ))}
    </div>
  );
}

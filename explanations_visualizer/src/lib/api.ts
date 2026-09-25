// Same-origin JSON client. Errors keep the backend's stable code and retryability so the
// interface can say exactly what failed instead of showing an empty panel.

export class ApiError extends Error {
  status: number;
  code: string;
  retryable: boolean;
  currentRevision: number | null;

  constructor(status: number, code: string, message: string, retryable = false, currentRevision: number | null = null) {
    super(message);
    this.status = status;
    this.code = code;
    this.retryable = retryable;
    this.currentRevision = currentRevision;
  }
}

type Param = string | number | boolean | null | undefined | string[];

export function buildUrl(path: string, params?: Record<string, Param>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) value.forEach((item) => search.append(key, item));
    else search.append(key, String(value));
  }
  const query = search.toString();
  return query ? `${path}?${query}` : path;
}

async function parseError(response: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  const record = (body ?? {}) as Record<string, unknown>;
  if (typeof record.code === "string") {
    return new ApiError(
      response.status,
      record.code,
      typeof record.message === "string" ? record.message : "Request failed",
      Boolean(record.retryable),
    );
  }
  // The study service reports {detail, current_revision}.
  const detail = record.detail;
  const message = typeof detail === "string" ? detail : "Request failed";
  const revision = typeof record.current_revision === "number" ? record.current_revision : null;
  return new ApiError(response.status, `http_${response.status}`, message, response.status >= 500 || response.status === 429, revision);
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { credentials: "same-origin", ...init });
  } catch (error) {
    if ((error as Error)?.name === "AbortError") throw error;
    throw new ApiError(0, "network_unreachable", "The service could not be reached", true);
  }
  if (!response.ok) throw await parseError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function getJson<T>(path: string, params?: Record<string, Param>, signal?: AbortSignal): Promise<T> {
  return request<T>(buildUrl(path, params), { signal, headers: { Accept: "application/json" } });
}

export function sendJson<T>(method: "POST" | "PUT" | "DELETE", path: string, body?: unknown, headers: Record<string, string> = {}): Promise<T> {
  return request<T>(path, {
    method,
    headers: { "Content-Type": "application/json", Accept: "application/json", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export function isAbort(error: unknown): boolean {
  return (error as Error)?.name === "AbortError";
}

/** Plain-language sentence for a failed request, never a raw stack or path. */
export function describeError(error: unknown): string {
  if (!(error instanceof ApiError)) return "Something went wrong while loading this part.";
  switch (error.code) {
    case "network_unreachable":
      return "The service is not responding. What is already on screen stays; nothing new has loaded.";
    case "stale_cursor":
      return "This list changed because a different bundle or query is now active. Reload the list.";
    case "package_unavailable":
      return "No bundle is open. Import or select one in the library.";
    case "explanation_unverified":
      return "A generated text exists but has not passed grounding review, so it is not shown.";
    case "response_too_large":
    case "axiom_detail_too_large":
      return "This item is too large to show inline. Its original stays available in the ontology file.";
    case "not_found":
      return "This item is not available in the open bundle.";
    case "artifact_unavailable":
      return "A prepared file could not be read. Try again; if it persists, re-import the bundle.";
    default:
      return error.message || "The request failed.";
  }
}

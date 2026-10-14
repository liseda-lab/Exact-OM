// HR1 (spec 24 §3): application styling at the study's reload and stage boundaries, with bounded,
// sanitized evidence of how the application stylesheet was delivered and kept.
//
// `expectStyled(stage)` is the regression check: the stylesheet's own token (--paper) must be
// computed and the inline icons must be sized by it, within a bounded wait. A link element or
// font readiness alone is not accepted as styling. On failure it captures the viewport at once.
//
// The recorder (always attached by the tests that use it; test code only, never shipped) keeps:
// - per document: stylesheet <link> insertion/removal, href path, rel, media, disabled,
//   load/error and rule availability; CSP violations; page errors. Paths only: never a query,
//   fragment (invitation), DOM text, form answer, token, cookie or storage value.
// - per request (Chrome DevTools Protocol): CSS initiation, cache source, status, MIME, declared,
//   compressed (encoded) and decoded sizes, completion or failure/abort, and the body's SHA-256,
//   compared with the exact export file when EXACT_E2E_EXPORT_DIR names it.
// EXACT_E2E_STYLE_DIAGNOSTICS=<dir> writes the sanitized record and failure screenshots there;
// without it nothing is written unless a check fails (then only the error message carries a
// summary). Evidence from development-review sessions only; never participant data.

import { createHash } from "node:crypto";
import { closeSync, mkdirSync, openSync, readSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { expect, type CDPSession, type Page, type TestInfo } from "@playwright/test";

const OUTPUT = process.env.EXACT_E2E_STYLE_DIAGNOSTICS ?? null;
const EXPORT_DIR = process.env.EXACT_E2E_EXPORT_DIR ?? null;
const MAX_EVENTS = 600;
// Diagnostic work only: these limits never retry, cancel or modify a browser request.
const MAX_CSS_BYTES = 1_048_576;
const MAX_PENDING_HASHES = 4;
const HASH_TIMEOUT_MS = 2_000;
/** An inline icon is 1–1.5rem; an unstyled one fills its container (HR1 showed ~900 px). */
export const MAX_STYLED_ICON_PX = 64;

export interface StyleProbe {
  stage: string;
  document: number | null;
  navigation_type: string | null;
  token: string;
  body_font: string;
  max_icon_px: number;
  icons: number;
  style_sheets: number;
  links: { path: string; rel: string; media: string; disabled: boolean; rules: number | null }[];
  fonts: string;
  styled: boolean;
}

interface NetworkRecord {
  id: string;
  path: string;
  type: string;
  document: string;
  started_ms: number;
  from_cache: "memory" | "disk" | "prefetch" | "service_worker" | "network" | "unknown";
  status: number | null;
  mime: string | null;
  declared_length: string | null;
  content_encoding: string | null;
  server_read_id: string | null;
  server_asset_sha256: string | null;
  encoded_bytes: number | null;
  decoded_bytes: number;
  outcome: "pending" | "finished" | "failed";
  error: string | null;
  canceled: boolean | null;
  blocked: string | null;
  body_sha256: string | null;
  body_hash_status: "not_requested" | "pending" | "available" | "size_limit" | "budget_limit" | "timeout" | "unavailable";
  export_hash_status: "not_requested" | "matched" | "mismatch" | "size_limit" | "unavailable";
  matches_export: boolean | null;
}

/** Path only, for same-origin application resources; everything else is reduced to its kind. */
function safePath(url: string, origin: string): string {
  try {
    const parsed = new URL(url);
    if (parsed.origin !== origin) return parsed.protocol === "data:" ? "data:" : "external";
    return parsed.pathname;
  } catch {
    return "unparsed";
  }
}

// Runs in the page before any document script; observation only.
function installRecorder(maxEvents: number) {
  const state = { document: Date.now(), events: [] as Record<string, unknown>[] };
  const push = (event: Record<string, unknown>) => {
    if (state.events.length < maxEvents) state.events.push({ at: Math.round(performance.now()), ...event });
  };
  const path = (href: string | null) => {
    try {
      const url = new URL(href ?? "", location.href);
      return url.origin === location.origin ? url.pathname : "external";
    } catch {
      return "unparsed";
    }
  };
  const rules = (link: HTMLLinkElement) => {
    try {
      return link.sheet ? link.sheet.cssRules.length : null;
    } catch {
      return -1;
    }
  };
  const describe = (link: HTMLLinkElement) => ({ path: path(link.getAttribute("href")), rel: link.rel, media: link.media, disabled: link.disabled, rules: rules(link) });
  const isStyle = (node: Node): node is HTMLLinkElement => node instanceof HTMLLinkElement && /stylesheet/i.test(node.rel);
  (window as unknown as { __exactStyleDiag: typeof state }).__exactStyleDiag = state;
  push({ type: "document", page: location.pathname, navigation: (performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined)?.type ?? null });
  new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === "attributes" && isStyle(record.target)) push({ type: "link-attribute", name: record.attributeName, ...describe(record.target) });
      for (const node of Array.from(record.addedNodes)) if (isStyle(node)) push({ type: "link-added", parent: node.parentNode?.nodeName ?? null, ...describe(node) });
      for (const node of Array.from(record.removedNodes)) if (isStyle(node)) push({ type: "link-removed", ...describe(node) });
      for (const node of Array.from(record.addedNodes)) if (node instanceof HTMLStyleElement) push({ type: "style-element-added" });
    }
  }).observe(document, { subtree: true, childList: true, attributes: true, attributeFilter: ["href", "rel", "media", "disabled"] });
  for (const type of ["load", "error"] as const)
    document.addEventListener(type, (event) => {
      if (isStyle(event.target as Node)) push({ type: `link-${type}`, ...describe(event.target as HTMLLinkElement) });
    }, true);
  document.addEventListener("securitypolicyviolation", (event) =>
    push({ type: "csp-violation", directive: event.effectiveDirective, blocked: event.blockedURI.startsWith("http") ? path(event.blockedURI) : event.blockedURI.slice(0, 16), source: path(event.sourceFile), line: event.lineNumber }),
  );
  window.addEventListener("error", (event) => push({ type: "page-error", error_type: /^(Error|TypeError|ReferenceError|SyntaxError|RangeError|AbortError|NetworkError|SecurityError)$/.test(event.error?.name ?? "") ? event.error.name : "unknown", source: path(event.filename) }));
  window.addEventListener("unhandledrejection", (event) => push({ type: "unhandled-rejection", error_type: /^(Error|TypeError|ReferenceError|SyntaxError|RangeError|AbortError|NetworkError|SecurityError)$/.test((event.reason as Error)?.name ?? "") ? (event.reason as Error).name : "unknown" }));
  window.addEventListener("pagehide", () => push({ type: "pagehide" }));
}

function probeInPage({ stage, maxIcon }: { stage: string; maxIcon: number }): StyleProbe {
  const state = (window as unknown as { __exactStyleDiag?: { document: number } }).__exactStyleDiag;
  const root = getComputedStyle(document.documentElement);
  const icons = Array.from(document.querySelectorAll("svg.icon, svg[class]")).filter((icon) => {
    const box = icon.getBoundingClientRect();
    return box.width > 0 && box.height > 0;
  });
  const widest = icons.reduce((max, icon) => Math.max(max, Math.round(icon.getBoundingClientRect().width)), 0);
  const links = Array.from(document.querySelectorAll<HTMLLinkElement>('link[rel~="stylesheet"]')).map((link) => {
    let rules: number | null = null;
    try {
      rules = link.sheet ? link.sheet.cssRules.length : null;
    } catch {
      rules = -1;
    }
    let path = "unparsed";
    try {
      path = new URL(link.href).pathname;
    } catch {
      /* keep unparsed */
    }
    return { path, rel: link.rel, media: link.media, disabled: link.disabled, rules };
  });
  const token = root.getPropertyValue("--paper").trim();
  return {
    stage,
    document: state?.document ?? null,
    navigation_type: (performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined)?.type ?? null,
    token,
    body_font: getComputedStyle(document.body).fontFamily.slice(0, 120),
    max_icon_px: widest,
    icons: icons.length,
    style_sheets: document.styleSheets.length,
    links,
    fonts: document.fonts.status,
    styled: Boolean(token) && widest <= maxIcon && links.some((link) => (link.rules ?? 0) > 0),
  };
}

export class StyleDiagnostics {
  readonly probes: StyleProbe[] = [];
  readonly network: NetworkRecord[] = [];
  readonly lifecycle: Record<string, unknown>[] = [];
  readonly documents: { document: number; events: Record<string, unknown>[] }[] = [];
  private readonly byId = new Map<string, NetworkRecord>();
  private readonly pendingHashes = new Set<Promise<void>>();
  private finishing = false;
  private readonly started = Date.now();
  private cdp: CDPSession | null = null;
  private failures = 0;

  constructor(private readonly page: Page, private readonly name: string, private readonly identity: Record<string, string | null>) {}

  /** Attach before the first navigation so the first document is observed too. */
  static async attach(page: Page, name: string, identity: Record<string, string | null> = {}): Promise<StyleDiagnostics> {
    const diagnostics = new StyleDiagnostics(page, name, identity);
    await page.addInitScript(installRecorder, MAX_EVENTS);
    await diagnostics.watchNetwork();
    page.on("framenavigated", (frame) => {
      if (frame === page.mainFrame()) diagnostics.note({ type: "navigated", page: safePath(frame.url(), new URL(frame.url()).origin) });
    });
    page.on("load", () => diagnostics.note({ type: "load" }));
    page.on("pageerror", (error) => diagnostics.note({ type: "pageerror", error_type: /^(Error|TypeError|ReferenceError|SyntaxError|RangeError|AbortError|NetworkError|SecurityError)$/.test(error.name) ? error.name : "unknown" }));
    page.on("requestfailed", (request) => {
      if (request.resourceType() === "stylesheet" || request.resourceType() === "document")
        diagnostics.note({ type: "requestfailed", resource: request.resourceType(), path: safePath(request.url(), new URL(request.url()).origin), error: /^net::ERR_[A-Z_]+$/.test(request.failure()?.errorText ?? "") ? request.failure()!.errorText : "unknown" });
    });
    return diagnostics;
  }

  /** A test-supplied, non-identifying observation (for example the first case's condition). */
  record(event: Record<string, unknown>) {
    this.note(event);
  }

  private note(event: Record<string, unknown>) {
    if (this.lifecycle.length < MAX_EVENTS) this.lifecycle.push({ at_ms: Date.now() - this.started, ...event });
  }

  private async watchNetwork() {
    const cdp = await this.page.context().newCDPSession(this.page);
    this.cdp = cdp;
    await cdp.send("Network.enable", {
      maxResourceBufferSize: MAX_CSS_BYTES,
      maxTotalBufferSize: MAX_PENDING_HASHES * MAX_CSS_BYTES,
    });
    const relevant = (type: string | undefined, path: string) => type === "Stylesheet" || type === "Document" || path.startsWith("/_next/static/css/");
    cdp.on("Network.requestWillBeSent", (event) => {
      const origin = new URL(event.documentURL || event.request.url).origin;
      const path = safePath(event.request.url, origin);
      if (!relevant(event.type, path)) return;
      const record: NetworkRecord = {
        id: event.requestId,
        path,
        type: event.type ?? "unknown",
        document: safePath(event.documentURL, origin),
        started_ms: Date.now() - this.started,
        from_cache: "unknown",
        status: null,
        mime: null,
        declared_length: null,
        content_encoding: null,
        server_read_id: null,
        server_asset_sha256: null,
        encoded_bytes: null,
        decoded_bytes: 0,
        outcome: "pending",
        error: null,
        canceled: null,
        blocked: null,
        body_sha256: null,
        body_hash_status: "not_requested",
        export_hash_status: "not_requested",
        matches_export: null,
      };
      if (this.network.length < MAX_EVENTS) { this.network.push(record); this.byId.set(event.requestId, record); }
    });
    cdp.on("Network.requestServedFromCache", (event) => {
      const record = this.byId.get(event.requestId);
      if (record) record.from_cache = "memory";
    });
    cdp.on("Network.responseReceived", (event) => {
      const record = this.byId.get(event.requestId);
      if (!record) return;
      const headers = Object.fromEntries(Object.entries(event.response.headers).map(([key, value]) => [key.toLowerCase(), String(value)]));
      record.status = event.response.status;
      record.mime = event.response.mimeType;
      record.declared_length = headers["content-length"] ?? null;
      record.content_encoding = headers["content-encoding"] ?? null;
      record.server_read_id = /^[a-f0-9]{32}$/.test(headers["x-exact-read-id"] ?? "") ? headers["x-exact-read-id"] : null;
      record.server_asset_sha256 = /^[a-f0-9]{64}$/.test(headers["x-exact-asset-sha256"] ?? "") ? headers["x-exact-asset-sha256"] : null;
      if (event.response.fromServiceWorker) record.from_cache = "service_worker";
      else if (event.response.fromPrefetchCache) record.from_cache = "prefetch";
      else if (event.response.fromDiskCache) record.from_cache = "disk";
      else if (record.from_cache === "unknown") record.from_cache = "network";
    });
    cdp.on("Network.dataReceived", (event) => {
      const record = this.byId.get(event.requestId);
      if (record) record.decoded_bytes += event.dataLength;
    });
    cdp.on("Network.loadingFinished", (event) => {
      const record = this.byId.get(event.requestId);
      if (!record) return;
      record.outcome = "finished";
      record.encoded_bytes = event.encodedDataLength;
      if (record.path.startsWith("/_next/static/css/")) this.queueHash(record);
    });
    cdp.on("Network.loadingFailed", (event) => {
      const record = this.byId.get(event.requestId);
      if (!record) return;
      record.outcome = "failed";
      record.error = /^net::ERR_[A-Z_]+$/.test(event.errorText) ? event.errorText : "unknown";
      record.canceled = event.canceled ?? null;
      record.blocked = event.blockedReason ?? null;
    });
  }

  private queueHash(record: NetworkRecord) {
    if (this.finishing || this.pendingHashes.size >= MAX_PENDING_HASHES) {
      record.body_hash_status = this.finishing ? "unavailable" : "budget_limit";
      return;
    }
    record.body_hash_status = "pending";
    const task: Promise<void> = this.hashBody(record).finally(() => this.pendingHashes.delete(task));
    this.pendingHashes.add(task);
  }

  private async hashBody(record: NetworkRecord) {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const declared = Number(record.declared_length);
      if (record.decoded_bytes > MAX_CSS_BYTES || (Number.isFinite(declared) && declared > MAX_CSS_BYTES)) {
        record.body_hash_status = "size_limit";
        return;
      }
      const body = await Promise.race([
        this.cdp!.send("Network.getResponseBody", { requestId: record.id }),
        new Promise<null>((resolve) => { timer = setTimeout(() => resolve(null), HASH_TIMEOUT_MS); }),
      ]);
      if (body === null) {
        record.body_hash_status = "timeout";
        return;
      }
      // The CDP buffer is capped above. Check the encoded representation before another
      // allocation, then the actual decoded size; compressed transfer size is not body size.
      const decodedSize = body.base64Encoded
        ? Math.ceil(body.body.length / 4) * 3 - (body.body.endsWith("==") ? 2 : body.body.endsWith("=") ? 1 : 0)
        : Buffer.byteLength(body.body, "utf8");
      if (decodedSize > MAX_CSS_BYTES) {
        record.body_hash_status = "size_limit";
        return;
      }
      const bytes = body.base64Encoded ? Buffer.from(body.body, "base64") : Buffer.from(body.body, "utf8");
      if (bytes.length > MAX_CSS_BYTES) {
        record.body_hash_status = "size_limit";
        return;
      }
      record.body_sha256 = createHash("sha256").update(bytes).digest("hex");
      record.body_hash_status = "available";
      if (EXPORT_DIR) {
        try {
          // Read at most the limit plus one sentinel byte even if an export changes mid-read.
          const expected = Buffer.alloc(MAX_CSS_BYTES + 1);
          const fd = openSync(join(EXPORT_DIR, record.path), "r");
          let size = 0;
          try {
            while (size < expected.length) {
              const read = readSync(fd, expected, size, expected.length - size, size);
              if (read === 0) break;
              size += read;
            }
          } finally {
            closeSync(fd);
          }
          if (size > MAX_CSS_BYTES) {
            record.export_hash_status = "size_limit";
            return;
          }
          record.matches_export = createHash("sha256").update(expected.subarray(0, size)).digest("hex") === record.body_sha256;
          record.export_hash_status = record.matches_export ? "matched" : "mismatch";
        } catch {
          record.export_hash_status = "unavailable";
        }
      }
    } catch {
      record.body_hash_status = "unavailable";
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }

  /** Bounded, condition-based: styled within the wait, or captured and failed with the evidence. */
  async expectStyled(stage: string, timeout = 10_000) {
    const seen: { last: StyleProbe | null } = { last: null };
    try {
      await expect
        .poll(async () => {
          seen.last = await this.probe(stage);
          return seen.last.styled;
        }, { timeout, intervals: [100, 250, 500, 1000] })
        .toBe(true);
    } catch (error) {
      this.failures += 1;
      if (seen.last) this.probes.push(seen.last);
      await this.captureDocuments();
      if (OUTPUT) {
        mkdirSync(OUTPUT, { recursive: true });
        await this.page.screenshot({ path: join(OUTPUT, `${this.name}-unstyled-${stage}.png`), fullPage: false }).catch(() => undefined);
      }
      this.write();
      const css = this.network.filter((record) => record.path.startsWith("/_next/static/css/")).slice(-4);
      throw new Error(`Application styling missing at "${stage}": ${JSON.stringify(seen.last)}; recent CSS transfers ${JSON.stringify(css)}; ${(error as Error).message.split("\n")[0]}`);
    }
    if (seen.last) this.probes.push(seen.last);
  }

  /** Computed application styling now; never throws. */
  async probe(stage: string): Promise<StyleProbe> {
    const result = await this.page.evaluate(probeInPage, { stage, maxIcon: MAX_STYLED_ICON_PX }).catch(() => null);
    return result ?? { stage, document: null, navigation_type: null, token: "", body_font: "", max_icon_px: -1, icons: 0, style_sheets: 0, links: [], fonts: "unknown", styled: false };
  }

  /** The current document's in-page record (retained per document across reloads). */
  async captureDocuments() {
    const current = await this.page.evaluate(() => (window as unknown as { __exactStyleDiag?: { document: number; events: Record<string, unknown>[] } }).__exactStyleDiag ?? null).catch(() => null);
    if (!current) return;
    const known = this.documents.find((item) => item.document === current.document);
    if (known) known.events = current.events;
    else this.documents.push(current);
  }

  /** Call before a reload or navigation so the outgoing document's record is kept. */
  async beforeNavigation(reason: string) {
    this.note({ type: "before-navigation", reason });
    await this.captureDocuments();
  }

  summary() {
    const css = this.network.filter((record) => record.path.startsWith("/_next/static/css/"));
    return {
      schema: "exact-style-diagnostics/1",
      test: this.name,
      identity: this.identity,
      checks: this.probes.length,
      failed_checks: this.failures,
      css_requests: css.length,
      css_failed: css.filter((record) => record.outcome === "failed").length,
      css_failed_not_canceled: css.filter((record) => record.outcome === "failed" && !record.canceled).length,
      css_pending: css.filter((record) => record.outcome === "pending").length,
      css_from: css.reduce<Record<string, number>>((counts, record) => ({ ...counts, [record.from_cache]: (counts[record.from_cache] ?? 0) + 1 }), {}),
      css_export_mismatch: css.filter((record) => record.matches_export === false).length,
      css_completed_hash_unknown: css.filter((record) => record.outcome === "finished" && record.body_hash_status !== "available").length,
      css_export_unknown: EXPORT_DIR ? css.filter((record) => record.outcome === "finished" && record.matches_export === null).length : null,
    };
  }

  write(info?: TestInfo) {
    if (!OUTPUT) return;
    mkdirSync(OUTPUT, { recursive: true });
    const record = { ...this.summary(), status: info?.status ?? null, probes: this.probes, network: this.network, lifecycle: this.lifecycle, documents: this.documents };
    writeFileSync(join(OUTPUT, `${this.name}.json`), JSON.stringify(record, null, 2) + "\n");
  }

  async finish(info?: TestInfo) {
    // Each pending operation has its own two-second deadline. Settle them before saving
    // evidence/detaching; a late body must not silently change a receipt after it is written.
    this.finishing = true;
    await Promise.allSettled([...this.pendingHashes]);
    await this.captureDocuments();
    this.write(info);
    await this.cdp?.detach().catch(() => undefined);
  }
}


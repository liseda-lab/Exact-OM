"use client";

// The frozen ontology files for the study, with the same entry points in both conditions.
// Downloading is never required and never recorded as use; the information notice explains
// that other releases or viewers may show different information without prescribing a tool.

import { useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { IconCopy, IconDownload } from "@/components/common/Icons";
import { shortHash } from "@/lib/iri";
import { formatBytes } from "@/lib/zipManifest";
import type { PublicAsset } from "@/study/types";

const FORMATS: Record<string, string> = {
  "application/owl-functional": "OWL Functional Syntax",
  "application/rdf+xml": "RDF/XML",
  "application/owl+xml": "OWL/XML",
  "text/turtle": "Turtle",
  "text/plain": "Plain text",
};

export const SCOPE_NOTICE =
  "These are the exact ontology versions and the information this study supplies. Another release, an online viewer or another source may show different or additional information, including mappings this study withholds. You may use any inspection method; if you consult a different source, you can say so after the case.";

export const OPEN_HELP =
  "How to open them: ontology editors (Protégé is one) and many ontology viewers open these files directly, and any text editor shows them as plain text. Opening them is optional; nothing on your computer is checked.";

function roleText(asset: PublicAsset): string {
  if (asset.role === "source") return "Source ontology";
  if (asset.role === "target") return "Target ontology";
  if (asset.role === "both") return "Source and target";
  return "Role not recorded in this study version";
}

export function ResourceList({
  resources,
  compact = false,
  baseUrl = "/api/v1/study/resources",
  onDownload,
}: {
  resources: PublicAsset[];
  compact?: boolean;
  baseUrl?: string;
  onDownload?: (assetId: string) => void;
}) {
  const [copied, setCopied] = useState<string | null>(null);
  if (!resources.length) return <p className="note">No ontology file is attached to this study.</p>;
  return (
    <ul className={compact ? "downloads compact" : "downloads"}>
      {resources.map((asset) => (
        <li key={asset.asset_id} className="download-card">
          <span className="download-name">{asset.title || asset.asset_id}</span>
          <span className="meta">
            {roleText(asset)}
            {asset.version_label ? ` · version ${asset.version_label}` : ""} · {FORMATS[asset.media_type] ?? asset.media_type} · {formatBytes(asset.size_bytes)}
          </span>
          {!compact && (
            <span className="download-hash">
              <span className="iri">SHA-256 {shortHash(asset.sha256, 16)}</span>
              <button
                type="button"
                className="btn btn-quiet btn-sm"
                aria-label={`Copy the full SHA-256 of ${asset.title || asset.asset_id}`}
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
          {asset.information_notice && <span className="meta">{asset.information_notice}</span>}
          <a className="btn" href={`${baseUrl}/${encodeURIComponent(asset.asset_id)}`} download onClick={() => onDownload?.(asset.asset_id)}>
            <IconDownload /> Download
          </a>
        </li>
      ))}
    </ul>
  );
}

/** A persistent case-toolbar entry point to the same files in both conditions. */
export function ResourceAccessButton({
  resources,
  onOpen,
  onDownload,
  baseUrl,
}: {
  resources: PublicAsset[];
  onOpen?: () => void;
  onDownload?: (assetId: string) => void;
  baseUrl?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        className="btn btn-sm"
        aria-haspopup="dialog"
        onClick={() => {
          setOpen(true);
          onOpen?.();
        }}
      >
        <IconDownload /> Ontology files ({resources.length})
      </button>
      {open && (
        <Dialog title="Ontology files for this study" onClose={() => setOpen(false)} wide>
          <p className="muted">{SCOPE_NOTICE}</p>
          <ResourceList resources={resources} onDownload={onDownload} baseUrl={baseUrl} />
          <p className="meta">{OPEN_HELP}</p>
          <p className="meta">Downloading is optional and is not recorded as using a tool. Time you spend inspecting before you submit counts as part of the case.</p>
        </Dialog>
      )}
    </>
  );
}

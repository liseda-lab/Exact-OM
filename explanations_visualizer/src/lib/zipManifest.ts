// Read only package.json from a local ZIP before upload, so the import screen can show the
// bundle's declared identity and coverage. The service still validates everything.

export interface BundlePreview {
  package_id?: string;
  contract_version?: string;
  schema_version?: string;
  audience?: string;
  ontologies: number;
  runs: number;
  explanations: number;
  artifacts: number;
  artifactBytes: number;
  capabilities: Record<string, string>;
  portable?: boolean;
}

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function readBundlePreview(file: File): Promise<BundlePreview | null> {
  const tailSize = Math.min(file.size, 66 * 1024);
  const tail = new DataView(await file.slice(file.size - tailSize).arrayBuffer());
  let eocd = -1;
  for (let offset = tail.byteLength - 22; offset >= 0; offset -= 1) {
    if (tail.getUint32(offset, true) === 0x06054b50) {
      eocd = offset;
      break;
    }
  }
  if (eocd < 0) return null;
  const entries = tail.getUint16(eocd + 10, true);
  const cdSize = tail.getUint32(eocd + 12, true);
  const cdOffset = tail.getUint32(eocd + 16, true);
  if (cdOffset === 0xffffffff || cdSize > 16 * 1024 * 1024) return null;
  const cd = new DataView(await file.slice(cdOffset, cdOffset + cdSize).arrayBuffer());
  const decoder = new TextDecoder();
  let pointer = 0;
  for (let index = 0; index < entries && pointer + 46 <= cd.byteLength; index += 1) {
    if (cd.getUint32(pointer, true) !== 0x02014b50) return null;
    const method = cd.getUint16(pointer + 10, true);
    const compressed = cd.getUint32(pointer + 20, true);
    const nameLength = cd.getUint16(pointer + 28, true);
    const extraLength = cd.getUint16(pointer + 30, true);
    const commentLength = cd.getUint16(pointer + 32, true);
    const localOffset = cd.getUint32(pointer + 42, true);
    const name = decoder.decode(new Uint8Array(cd.buffer, cd.byteOffset + pointer + 46, nameLength));
    pointer += 46 + nameLength + extraLength + commentLength;
    if (name !== "package.json") continue;
    if (compressed > 64 * 1024 * 1024) return null;
    const header = new DataView(await file.slice(localOffset, localOffset + 30).arrayBuffer());
    if (header.getUint32(0, true) !== 0x04034b50) return null;
    const start = localOffset + 30 + header.getUint16(26, true) + header.getUint16(28, true);
    const raw = new Uint8Array(await file.slice(start, start + compressed).arrayBuffer());
    const bytes = method === 0 ? raw : method === 8 ? await inflate(raw) : null;
    if (!bytes) return null;
    const manifest = JSON.parse(decoder.decode(bytes)) as Record<string, unknown>;
    const artifacts = Array.isArray(manifest.artifacts) ? (manifest.artifacts as { size?: number }[]) : [];
    const count = (value: unknown) => (value && typeof value === "object" ? Object.keys(value).length : 0);
    return {
      package_id: typeof manifest.package_id === "string" ? manifest.package_id : undefined,
      contract_version: typeof manifest.contract_version === "string" ? manifest.contract_version : undefined,
      schema_version: typeof manifest.schema_version === "string" ? manifest.schema_version : undefined,
      audience: typeof manifest.audience === "string" ? manifest.audience : undefined,
      ontologies: count(manifest.ontologies),
      runs: count(manifest.runs),
      explanations: count(manifest.explanations),
      artifacts: artifacts.length,
      artifactBytes: artifacts.reduce((sum, item) => sum + (typeof item.size === "number" ? item.size : 0), 0),
      capabilities: (manifest.capabilities as Record<string, string>) ?? {},
      portable: typeof manifest.portable === "boolean" ? manifest.portable : undefined,
    };
  }
  return null;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
}

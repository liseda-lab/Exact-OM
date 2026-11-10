"use client";

import { useEffect, useRef, useState } from "react";

type Invitation = { session_id: string; generation: number; test: boolean; stage: string };
type Page = { items: Invitation[]; next_cursor: string | null; study_closed: boolean };

/** Secrets cannot be recovered. Select an existing session to replace or revoke its link. */
export function ActiveInvitations({ token, revision, refresh, value, onSelect, disabled }: {
  token: string; revision: string; refresh: number; value: string;
  onSelect: (id: string) => void; disabled: boolean;
}) {
  const [data, setData] = useState<Page | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [retry, setRetry] = useState(0);
  const generation = useRef(0);
  async function load(cursor: string | null, current: number) {
    setLoading(true); setError("");
    try {
      const response = await fetch(`/api/v1/admin/studies/${encodeURIComponent(revision)}/invitations?limit=100${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`, {
        headers: { Authorization: `Bearer ${token}` }, credentials: "omit", cache: "no-store",
      });
      if (!response.ok) throw new Error(response.status === 401 ? "Sign out and sign in again to load the links." : "The active links could not be loaded. Please try again.");
      const page: Page = await response.json();
      if (generation.current === current) setData(old => ({ ...page, items: cursor ? [...(old?.items ?? []), ...page.items] : page.items }));
    } catch (cause) {
      if (generation.current === current) setError(cause instanceof Error ? cause.message : "The links could not be loaded.");
    } finally { if (generation.current === current) setLoading(false); }
  }
  useEffect(() => {
    const current = ++generation.current;
    setData(null);
    if (revision && token) void load(null, current);
    return () => { generation.current++; };
    // Every request is scoped to this auth/revision generation, including load-more requests.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, revision, refresh, retry]);
  return <div className="field">
    <label htmlFor="active-invitation">Active links</label>
    <select id="active-invitation" className="input" value={data?.items.some(item => item.session_id === value) ? value : ""}
      disabled={disabled || !data?.items.length} onChange={e => onSelect(e.target.value)}>
      <option value="">Choose a link</option>
      {data?.items.map(item => <option key={item.session_id} value={item.session_id}>
        {item.test ? "Test" : "Participant"} · {item.session_id} · {item.stage.replace(/_/g, " ")} · link {item.generation}
      </option>)}
    </select>
    {!revision && <p className="meta">Select a study revision first.</p>}
    {loading && <p role="status">Loading active links…</p>}
    {error && <p role="alert">{error}</p>}
    {data && !data.items.length && <p role="status">{data.study_closed ? "This study is closed. It has no usable links." : "No active links in this study. Create links above to get started."}</p>}
    <div className="admin-inline">
      <button className="btn btn-sm" type="button" disabled={!revision || loading || disabled} onClick={() => { onSelect(""); setRetry(n => n + 1); }}>Refresh links</button>
      {data?.next_cursor && <button className="btn btn-sm" type="button" disabled={loading || disabled} onClick={() => void load(data.next_cursor, generation.current)}>Load more links</button>}
    </div>
    <p className="meta">Select a session by its ID and current step. The original private URL cannot be retrieved.</p>
  </div>;
}

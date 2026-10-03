"use client";

// Accessible combobox over the active workspace's entity search (exact IRI or label/synonym
// prefix). A workspace that can only search part of an ontology says so in the results.

import { useEffect, useId, useRef, useState } from "react";

import { IconSearch } from "@/components/common/Icons";
import { describeError } from "@/lib/api";
import { curie, KIND_NAMES } from "@/lib/iri";
import type { SearchItem } from "@/lib/types";
import { useWorkspace } from "@/lib/workspace/WorkspaceContext";

export function EntitySearch({
  ontology,
  label,
  placeholder,
  onChoose,
}: {
  ontology: string;
  label: string;
  placeholder: string;
  onChoose: (item: SearchItem) => void;
}) {
  const workspace = useWorkspace();
  // Search restarts only for a different term, ontology or workspace scope, never because
  // the same workspace was re-created by an unrelated state update.
  const workspaceRef = useRef(workspace);
  workspaceRef.current = workspace;
  const id = useId();
  const [term, setTerm] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [results, setResults] = useState<SearchItem[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [cursor, setCursor] = useState<string | null>(null);
  const [scopeReason, setScopeReason] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    const query = term.trim();
    setResults([]);
    setTotal(null);
    setCursor(null);
    setError(null);
    setScopeReason(null);
    setLoading(Boolean(query));
    if (!query) return () => current.abort();
    const timer = window.setTimeout(async () => {
      try {
        const page = await workspaceRef.current.search(ontology, query, null, current.signal);
        if (current.signal.aborted) return;
        setResults(page.items);
        setTotal(page.total_count);
        setCursor(page.next_cursor);
        setScopeReason(page.status === "partial" ? page.reason : null);
        setError(null);
        setActive(0);
        setOpen(true);
      } catch (err) {
        if (!current.signal.aborted && (err as Error)?.name !== "AbortError") setError(describeError(err));
      } finally {
        if (!current.signal.aborted) setLoading(false);
      }
    }, 250);
    return () => {
      window.clearTimeout(timer);
      current.abort();
    };
  }, [term, ontology, workspace.key]);

  const loadMore = async () => {
    const current = controller.current;
    if (!cursor || !current || current.signal.aborted || loading) return;
    setLoading(true);
    try {
      const page = await workspaceRef.current.search(ontology, term.trim(), cursor, current.signal);
      if (current.signal.aborted) return;
      setResults((list) => [...list, ...page.items]);
      setCursor(page.next_cursor);
    } catch (err) {
      if (!current.signal.aborted) setError(describeError(err));
    } finally {
      if (!current.signal.aborted) setLoading(false);
    }
  };

  const choose = (item: SearchItem) => {
    onChoose(item);
    setOpen(false);
    setTerm("");
  };

  const onKey = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setOpen(true);
      setActive((index) => Math.min(results.length - 1, index + 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((index) => Math.max(0, index - 1));
    } else if (event.key === "Enter" && open && results[active]) {
      event.preventDefault();
      choose(results[active]);
    } else if (event.key === "Escape") {
      setOpen(false);
    }
  };

  const listId = `${id}-list`;
  return (
    <div className="combo">
      <label htmlFor={`${id}-input`} className="sr-only">
        {label}
      </label>
      <div className="search-field">
        <IconSearch />
        <input
          id={`${id}-input`}
          type="search"
          role="combobox"
          aria-expanded={open && results.length > 0}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={open && results[active] ? `${id}-opt-${active}` : undefined}
          className="search-input"
          placeholder={placeholder}
          value={term}
          onChange={(event) => setTerm(event.target.value)}
          onKeyDown={onKey}
          onFocus={() => results.length && setOpen(true)}
          onBlur={() => window.setTimeout(() => setOpen(false), 150)}
          autoComplete="off"
          spellCheck={false}
        />
        {loading && <span className="meta">Searching…</span>}
      </div>
      {error && <p className="note note-bad">{error}</p>}
      {open && term.trim() && (
        <ul id={listId} role="listbox" className="combo-list" aria-label={`${label} results`}>
          {results.map((item, index) => (
            <li
              key={`${item.entity.kind}|${item.entity.iri}`}
              id={`${id}-opt-${index}`}
              role="option"
              aria-selected={index === active}
              className={index === active ? "combo-option active" : "combo-option"}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => choose(item)}
            >
              <span className="combo-label">{item.preferred_label.value ?? <span className="muted">No label in scope</span>}</span>
              <span className="iri">
                {curie(item.entity.iri)} · {KIND_NAMES[item.entity.kind] ?? item.entity.kind}
              </span>
            </li>
          ))}
          {!results.length && !loading && <li className="combo-empty">No label, synonym or IRI starts with “{term.trim()}”{scopeReason ? " in the searchable part of this view" : ""}.</li>}
          {(results.length > 0 || scopeReason) && (
            <li className="combo-footer">
              <span className="meta">
                {results.length} of {total ?? "?"} matches{scopeReason ? ` · ${scopeReason}` : ""}
              </span>
              {cursor && (
                <button type="button" className="btn btn-sm" disabled={loading} onMouseDown={(event) => event.preventDefault()} onClick={loadMore}>
                  Load more
                </button>
              )}
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

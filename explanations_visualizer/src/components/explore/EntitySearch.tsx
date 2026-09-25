"use client";

// Accessible combobox over GET /api/v1/entities (exact IRI or label/synonym prefix).

import { useEffect, useId, useRef, useState } from "react";

import { IconSearch } from "@/components/common/Icons";
import { describeError, getJson } from "@/lib/api";
import { curie, KIND_NAMES } from "@/lib/iri";
import type { Page, SearchItem } from "@/lib/types";

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
  const id = useId();
  const [term, setTerm] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [results, setResults] = useState<SearchItem[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [cursor, setCursor] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    const query = term.trim();
    if (!query) {
      setResults([]);
      setTotal(null);
      setCursor(null);
      setError(null);
      return;
    }
    const timer = window.setTimeout(async () => {
      controller.current?.abort();
      const current = new AbortController();
      controller.current = current;
      setLoading(true);
      try {
        const page = await getJson<Page<SearchItem>>("/api/v1/entities", { ontology_version_id: ontology, term: query, limit: 20 }, current.signal);
        setResults(page.items);
        setTotal(page.total_count);
        setCursor(page.next_cursor);
        setError(null);
        setActive(0);
        setOpen(true);
      } catch (err) {
        if ((err as Error)?.name !== "AbortError") setError(describeError(err));
      } finally {
        if (!current.signal.aborted) setLoading(false);
      }
    }, 250);
    return () => window.clearTimeout(timer);
  }, [term, ontology]);

  const loadMore = async () => {
    if (!cursor) return;
    try {
      const page = await getJson<Page<SearchItem>>("/api/v1/entities", { ontology_version_id: ontology, term: term.trim(), limit: 20, cursor });
      setResults((list) => [...list, ...page.items]);
      setCursor(page.next_cursor);
    } catch (err) {
      setError(describeError(err));
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
          {!results.length && !loading && <li className="combo-empty">No label, synonym or IRI starts with “{term.trim()}”.</li>}
          {results.length > 0 && (
            <li className="combo-footer">
              <span className="meta">
                {results.length} of {total ?? "?"} matches
              </span>
              {cursor && (
                <button type="button" className="btn btn-sm" onMouseDown={(event) => event.preventDefault()} onClick={loadMore}>
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

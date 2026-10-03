"use client";

// One accessible tab primitive for every product: roving focus with Arrow keys, Home and
// End, manual activation (Enter, Space or click), and labelled tab/panel relationships.
// Moving focus never loads a panel; only activation does.

import { useEffect, useId, useRef, useState } from "react";

export interface TabItem<K extends string> {
  key: K;
  label: string;
  note?: string;
}

export function Tabs<K extends string>({
  label,
  tabs,
  active,
  onChange,
  allowNone = false,
  emptyNote,
  className,
  children,
}: {
  label: string;
  tabs: TabItem<K>[];
  active: K | null;
  onChange: (key: K | null) => void;
  /** Clicking the open tab closes it; nothing is shown until a tab is chosen. */
  allowNone?: boolean;
  emptyNote?: React.ReactNode;
  className?: string;
  children: (active: K) => React.ReactNode;
}) {
  const base = useId();
  const [focused, setFocused] = useState<K>(active ?? tabs[0]?.key);
  const buttons = useRef(new Map<K, HTMLButtonElement>());
  useEffect(() => {
    if (active) setFocused(active);
  }, [active]);
  const current = tabs.some((tab) => tab.key === focused) ? focused : tabs[0]?.key;
  const tabId = (key: K) => `${base}-tab-${key}`;
  const panelId = `${base}-panel`;

  const move = (event: React.KeyboardEvent, index: number) => {
    const next =
      event.key === "ArrowRight" ? (index + 1) % tabs.length
      : event.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length
      : event.key === "Home" ? 0
      : event.key === "End" ? tabs.length - 1
      : null;
    if (next === null) return;
    event.preventDefault();
    setFocused(tabs[next].key);
    buttons.current.get(tabs[next].key)?.focus();
  };

  return (
    <div className={className}>
      <div className="tab-list" role="tablist" aria-label={label}>
        {tabs.map((tab, index) => (
          <button
            key={tab.key}
            ref={(node) => {
              if (node) buttons.current.set(tab.key, node);
              else buttons.current.delete(tab.key);
            }}
            type="button"
            role="tab"
            id={tabId(tab.key)}
            aria-selected={active === tab.key}
            aria-controls={active === tab.key ? panelId : undefined}
            tabIndex={current === tab.key ? 0 : -1}
            className={active === tab.key ? "tab on" : "tab"}
            title={tab.note}
            onFocus={() => setFocused(tab.key)}
            onKeyDown={(event) => move(event, index)}
            onClick={() => onChange(allowNone && active === tab.key ? null : tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {active ? (
        <div id={panelId} role="tabpanel" aria-labelledby={tabId(active)} tabIndex={0} className="details-panel">
          {children(active)}
        </div>
      ) : (
        <div className="details-panel">{emptyNote}</div>
      )}
    </div>
  );
}

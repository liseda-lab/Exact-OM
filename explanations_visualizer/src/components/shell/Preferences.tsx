"use client";

import { TEXT_SCALES, useTextScale, useTheme } from "@/lib/prefs";

export function TextSizeControl({ compact = false }: { compact?: boolean }) {
  const [scale, setScale] = useTextScale();
  const index = TEXT_SCALES.indexOf(scale as (typeof TEXT_SCALES)[number]);
  const step = (delta: number) => {
    const next = TEXT_SCALES[Math.min(TEXT_SCALES.length - 1, Math.max(0, (index < 0 ? 0 : index) + delta))];
    setScale(next);
  };
  return (
    <div className="segmented" role="group" aria-label="Text size">
      <button type="button" aria-label="Decrease text size" onClick={() => step(-1)} disabled={index <= 0}>
        A−
      </button>
      {!compact && (
        <span className="segmented-value" aria-live="polite">
          {Math.round(scale * 100)}%
        </span>
      )}
      <button type="button" aria-label="Increase text size" className="segmented-large" onClick={() => step(1)} disabled={index >= TEXT_SCALES.length - 1}>
        A+
      </button>
    </div>
  );
}

export function ThemeControl() {
  const [theme, setTheme] = useTheme();
  return (
    <label className="theme-select">
      <span className="sr-only">Colour theme</span>
      <select className="select select-compact" value={theme} onChange={(event) => setTheme(event.target.value as "system" | "light" | "dark")}>
        <option value="system">Auto theme</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>
  );
}

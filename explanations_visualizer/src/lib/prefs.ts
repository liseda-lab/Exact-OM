"use client";

// Per-viewer display preferences. Storage can be unavailable (private mode, blocked site
// data); every access is guarded and the interface works with defaults.

import { useCallback, useEffect, useState } from "react";

export const TEXT_SCALES = [1, 1.125, 1.25, 1.5, 1.75, 2] as const;
export type Theme = "system" | "light" | "dark";

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* preferences are a convenience only */
  }
}

function applyScale(scale: number) {
  document.documentElement.style.setProperty("--text-scale", String(scale));
}

function applyTheme(theme: Theme) {
  if (theme === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
}

export function useTextScale(): [number, (next: number) => void] {
  const [scale, setScale] = useState(1);
  useEffect(() => {
    const stored = Number(read("exact.textScale"));
    const initial = TEXT_SCALES.includes(stored as (typeof TEXT_SCALES)[number]) ? stored : 1;
    setScale(initial);
    applyScale(initial);
  }, []);
  const update = useCallback((next: number) => {
    setScale(next);
    applyScale(next);
    write("exact.textScale", String(next));
  }, []);
  return [scale, update];
}

export function useTheme(): [Theme, (next: Theme) => void] {
  const [theme, setTheme] = useState<Theme>("system");
  useEffect(() => {
    const stored = read("exact.theme");
    const initial: Theme = stored === "light" || stored === "dark" ? stored : "system";
    setTheme(initial);
    applyTheme(initial);
  }, []);
  const update = useCallback((next: Theme) => {
    setTheme(next);
    applyTheme(next);
    write("exact.theme", next);
  }, []);
  return [theme, update];
}

export function readStored<T>(key: string, fallback: T): T {
  const raw = read(key);
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function writeStored(key: string, value: unknown) {
  write(key, JSON.stringify(value));
}

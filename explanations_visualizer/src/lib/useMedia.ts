"use client";

import { useSyncExternalStore } from "react";

function subscribe(listener: () => void) {
  window.addEventListener("resize", listener);
  // The text-size control changes the root font size through an inline custom property.
  const observer = new MutationObserver(listener);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["style"] });
  return () => {
    window.removeEventListener("resize", listener);
    observer.disconnect();
  };
}

/**
 * True when the viewport is at most `rem` root-em wide. Like the CSS container queries it
 * follows the app's text-size control, so enlarged text gets the narrow layout. False
 * during static prerender so markup stays stable.
 */
export function useNarrow(rem = 56.25): boolean {
  return useSyncExternalStore(
    subscribe,
    () => {
      const root = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      return document.documentElement.clientWidth / root <= rem;
    },
    () => false,
  );
}

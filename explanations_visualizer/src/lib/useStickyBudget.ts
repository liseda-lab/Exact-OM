"use client";

// Sticky bars must never cover the content they serve (WCAG 2.4.11, 1.4.10). A bar publishes
// its height in a CSS variable so scrolled-to and focused controls clear it; when enlarged text
// or a short viewport makes it taller than its share of the viewport, it stops being sticky
// (`data-unstuck` on the nearest `[data-sticky]` element) and publishes no height.

import { useEffect, type RefObject } from "react";

/** `active` re-measures when the bar mounts or unmounts (for example a layout-dependent bar). */
export function useStickyBudget(ref: RefObject<HTMLElement | null>, variable: string, share: number, active = true) {
  useEffect(() => {
    const element = ref.current;
    if (!active || !element) return;
    const root = document.documentElement;
    const sticky = element.closest<HTMLElement>("[data-sticky]") ?? element;
    const update = () => {
      const height = Math.ceil(element.getBoundingClientRect().height);
      const fits = height <= window.innerHeight * share;
      sticky.toggleAttribute("data-unstuck", !fits);
      root.style.setProperty(variable, `${fits ? height : 0}px`);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    window.addEventListener("resize", update);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
      sticky.removeAttribute("data-unstuck");
      root.style.removeProperty(variable);
    };
  }, [ref, variable, share, active]);
}

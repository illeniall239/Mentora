"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { navClass } from "./NavLink";

// On an exercise page, opens that exercise's sketchpad (Workspace listens for this event);
// anywhere else, goes to the general sketchpad page.
export const OPEN_SKETCHPAD = "poopy:open-sketchpad";

export function SketchpadLink() {
  const path = usePathname();
  const active = path === "/sketchpad";

  if (path.startsWith("/exercises/")) {
    return (
      <button type="button" onClick={() => window.dispatchEvent(new Event(OPEN_SKETCHPAD))} className={navClass(false)} title="Sketch this exercise">
        Sketchpad
      </button>
    );
  }
  return (
    <Link href="/sketchpad" aria-current={active ? "page" : undefined} className={navClass(active)}>
      Sketchpad
    </Link>
  );
}

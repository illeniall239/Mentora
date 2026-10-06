import Link from "next/link";
import { Settings } from "./Icons";
import { NavLink } from "./NavLink";
import { SketchpadLink } from "./SketchpadLink";

export function Topbar() {
  return (
    <header className="flex shrink-0 items-center gap-6 border-b border-line px-4 py-[18px] sm:px-8 lg:gap-10">
      <Link href="/" aria-label="Mentora home" className="shrink-0 font-logo text-[23px] font-semibold leading-none tracking-[-0.03em]">
        mentora<span className="font-extrabold text-accent">.</span>
      </Link>
      <nav className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto" aria-label="Main">
        <NavLink href="/">Home</NavLink>
        <NavLink href="/recap">Recap</NavLink>
        <NavLink href="/interview">Interview practice</NavLink>
        <NavLink href="/mistakes">Mistake log</NavLink>
        <SketchpadLink />
      </nav>
      <Link href="/settings" aria-label="Settings" title="Settings" className="grid size-9 shrink-0 place-items-center rounded-full text-muted transition-colors hover:bg-subtle hover:text-ink">
        <Settings className="size-[18px]" />
      </Link>
    </header>
  );
}

"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export const navClass = (active: boolean) =>
  `shrink-0 whitespace-nowrap rounded-full px-3.5 py-2 text-[13.5px] transition-colors ${active ? "bg-subtle font-bold text-ink" : "font-medium text-muted hover:text-ink"}`;

export function NavLink({ href, children, match }: { href: string; children: React.ReactNode; match?: string }) {
  const path = usePathname();
  const active = href === "/" ? path === "/" : path.startsWith(match ?? href);
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={navClass(active)}
    >
      {children}
    </Link>
  );
}

export function TopicLink({ href, children, className }: { href: string; children: React.ReactNode; className: string }) {
  const path = usePathname();
  const active = path === href;
  return (
    <Link href={href} aria-current={active ? "page" : undefined} className={`${className} ${active ? "bg-ground" : "hover:bg-ground"}`}>
      {children}
    </Link>
  );
}

import { connection } from "next/server";
import { getMessages } from "@/lib/db.ts";
import { threadFor } from "@/lib/tutor.ts";
import { todaySummary } from "@/app/actions.ts";
import { Chat } from "@/components/Chat";
import { Markdown } from "@/components/Markdown";

export default async function RecapPage() {
  await connection();
  const summary = await todaySummary();
  const messages = getMessages(threadFor("recap", "")).map((m) => ({ role: m.role, content: m.content, provider: m.provider }));
  return (
    <main className="grid gap-6 p-6 lg:h-full lg:grid-cols-[minmax(0,1fr)_340px] xl:px-10">
      <section className="flex min-h-[70vh] flex-col gap-4 lg:min-h-0">
        <header className="grid gap-1">
          <p className="eyebrow">End of Session</p>
          <h1 className="font-display text-3xl font-bold">Recap</h1>
        </header>
        <div className="panel min-h-0 flex-1 p-5">
          <Chat kind="recap" id="today" initial={messages} autoStart placeholder="In 3 sentences: what did you learn, and what was hardest?" />
        </div>
      </section>
      <aside className="panel grid content-start gap-2 self-start p-5">
        <p className="eyebrow">What you did today</p>
        <Markdown text={summary} />
        <p className="border-t border-line pt-2 text-xs text-muted">The Tutor uses this list to check your recap.</p>
      </aside>
    </main>
  );
}

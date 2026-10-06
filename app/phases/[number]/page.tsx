import Link from "next/link";
import { notFound } from "next/navigation";
import { connection } from "next/server";
import { loadCurriculum } from "@/lib/curriculum.ts";
import { allExerciseStates, allTopicStates } from "@/lib/db.ts";
import { exerciseStatus, isTopicLearned, localToday } from "@/lib/progress.ts";
import { Circle, CircleCheck, CircleDot } from "@/components/Icons";

const statusLabel = { new: "", in_progress: "In progress", waiting_retry: "Comes back later", retry_due: "Retry due", done: "Done" };

// One phase of the Curriculum Map: its topics, each with its exercises.
export default async function PhasePage(props: PageProps<"/phases/[number]">) {
  await connection();
  const n = Number((await props.params).number);
  const { phases, topics, exercises } = loadCurriculum();
  const phase = phases.find((p) => p.number === n);
  if (!phase) notFound();

  const today = localToday();
  const topicStates = allTopicStates();
  const exerciseStates = allExerciseStates();
  const learned = (t: (typeof topics)[number]) => isTopicLearned(t, topicStates.get(t.id), exerciseStates, today);
  const currentTopicId = topics.find((t) => !learned(t))?.id;
  const inPhase = topics.filter((t) => t.phase === n);
  const done = inPhase.filter(learned).length;

  return (
    <main className="grid content-start gap-6 p-4 sm:p-8">
      <header className="grid gap-1.5">
        <p className="text-[13px] font-semibold text-muted">
          <Link href="/" className="hover:text-ink">Curriculum</Link> · Phase {phase.number}
        </p>
        <h1 className="text-[32px] font-extrabold leading-tight tracking-[-1px] text-balance">{phase.title}</h1>
        <div className="mt-2 flex max-w-md items-center gap-3">
          <span className="h-1 flex-1 overflow-hidden rounded-sm bg-subtle" aria-hidden>
            <span className="block h-full bg-accent" style={{ width: `${(done / inPhase.length) * 100}%` }} />
          </span>
          <span className="text-[12.5px] font-semibold text-muted">{done}/{inPhase.length} topics learned</span>
        </div>
      </header>

      <ol className="grid gap-3.5 md:grid-cols-2 2xl:grid-cols-3">
        {inPhase.map((t) => {
          const isLearned = learned(t);
          const isNow = t.id === currentTopicId;
          return (
            <li key={t.id} className={`grid content-start gap-3 rounded-[14px] border bg-panel p-[18px] ${isNow ? "border-accent-line" : "border-line"}`}>
              <Link href={`/topics/${t.id}`} aria-current={isNow ? "step" : undefined} className="group grid gap-2">
                <span className="flex items-center justify-between gap-2">
                  <span className={`eyebrow ${isNow ? "text-accent" : ""}`}>Topic {t.id}</span>
                  {isLearned ? (
                    <span className="chip bg-accent-soft text-accent">Learned</span>
                  ) : isNow ? (
                    <span className="chip bg-accent-soft text-accent">You are here</span>
                  ) : topicStates.get(t.id)?.teach_done_at ? (
                    <span className="chip bg-warn-soft text-warn">In progress</span>
                  ) : null}
                </span>
                <span className={`text-[15px] font-semibold leading-5 group-hover:text-accent ${isLearned ? "text-muted" : ""}`}>{t.title}</span>
              </Link>
              {t.exerciseIds.length > 0 ? (
                <div className="grid rounded-xl bg-subtle p-1">
                  {t.exerciseIds.map((id) => {
                    const status = exerciseStatus(exerciseStates.get(id), today);
                    return (
                      <Link key={id} href={`/exercises/${id}`} className="flex items-center gap-2.5 rounded-[9px] px-3 py-2.5 text-[13.5px] transition-colors hover:bg-panel">
                        {status === "done" ? <CircleCheck className="size-4 text-accent" /> : status === "new" ? <Circle className="size-4 text-line" /> : <CircleDot className="size-4" />}
                        <span className={`flex-1 font-medium ${status === "done" ? "text-muted" : ""}`}>{exercises.get(id)!.title}</span>
                        <span className="text-xs font-semibold text-muted">{statusLabel[status]}</span>
                      </Link>
                    );
                  })}
                </div>
              ) : (
                <p className="rounded-xl bg-subtle px-3 py-2.5 text-[13px] text-muted">Project practice</p>
              )}
            </li>
          );
        })}
      </ol>
    </main>
  );
}

import Link from "next/link";
import { connection } from "next/server";
import { loadCurriculum } from "@/lib/curriculum.ts";
import { allExerciseStates, allReviews, allTopicStates, getMessages } from "@/lib/db.ts";
import { exerciseStatus, isTopicLearned, localToday, planToday, type PlanItem } from "@/lib/progress.ts";
import { CurriculumCards } from "@/components/CurriculumCards";
import { ArrowRight, Circle, CircleCheck, CircleDot, RotateCcw } from "@/components/Icons";

type Step = { label: string; meta: string; done: boolean };

export default async function Home() {
  await connection();
  const today = localToday();
  const { phases, topics, exercises } = loadCurriculum();
  const topicStates = allTopicStates();
  const exerciseStates = allExerciseStates();
  const reviewsAll = allReviews();
  const plan = planToday(topics, topicStates, exerciseStates, reviewsAll, today);
  const topicOf = (id: string) => topics.find((t) => t.id === id)!;

  const reviews = plan.filter((p): p is Extract<PlanItem, { kind: "review" }> => p.kind === "review");
  const retries = plan.filter((p): p is Extract<PlanItem, { kind: "retry" }> => p.kind === "retry");
  const teach = plan.find((p): p is Extract<PlanItem, { kind: "teach" }> => p.kind === "teach");
  const practiceItem = plan.find((p): p is Extract<PlanItem, { kind: "practice" }> => p.kind === "practice");
  const exerciseItems = plan.filter((p): p is Extract<PlanItem, { kind: "exercise" }> => p.kind === "exercise");
  const currentTopic = teach ? topicOf(teach.topicId) : practiceItem ? topicOf(practiceItem.topicId) : exerciseItems[0] ? topicOf(exercises.get(exerciseItems[0].exerciseId)!.topicId) : null;
  const nextExerciseId = exerciseItems[0]?.exerciseId;

  const start = !currentTopic
    ? null
    : teach
      ? { href: `/topics/${currentTopic.id}`, label: getMessages(`teach:${currentTopic.id}`).some((m) => m.role === "learner") ? "Continue the lesson" : "Start the lesson" }
      : practiceItem
        ? { href: `/topics/${currentTopic.id}`, label: "Get it reviewed" }
        : { href: `/exercises/${nextExerciseId}`, label: exerciseItems[0].status === "in_progress" ? "Continue exercise" : "Start exercise" };

  // Today's plan: the fixed daily loop, with the first unfinished step marked as current.
  const steps: Step[] = [];
  const reviewedToday = reviewsAll.filter((r) => r.last_done === today).length;
  if (reviews.length || reviewedToday) steps.push({ label: "Spaced Reviews", meta: reviews.length ? `${reviews.length} due` : "Done", done: !reviews.length });
  if (retries.length) steps.push({ label: "Come-back Exercises", meta: `${retries.length} due`, done: false });
  if (currentTopic) {
    steps.push({ label: `Topic ${currentTopic.id}`, meta: teach ? "Lesson" : practiceItem ? "Project review" : "Lesson done", done: !teach && !practiceItem });
    if (currentTopic.exerciseIds.length) {
      const doneCount = currentTopic.exerciseIds.filter((id) => exerciseStatus(exerciseStates.get(id), today) === "done").length;
      steps.push({ label: `${currentTopic.exerciseIds.length} exercises`, meta: `${doneCount}/${currentTopic.exerciseIds.length} done`, done: doneCount === currentTopic.exerciseIds.length });
    }
  }
  steps.push({ label: "Recap", meta: "2 min", done: false });
  const currentStep = steps.findIndex((s) => !s.done);

  return (
    <main className="grid gap-6 p-4 sm:p-8 xl:grid-cols-[minmax(0,1fr)_340px]">
      {/* Up next: the main work of the Session. Same row as the rail, so both stretch to one height. */}
      <section className="flex min-w-0 flex-col gap-[22px] rounded-[18px] border border-accent-line bg-panel p-5 sm:p-7" aria-labelledby="next">
        <header className="grid gap-1.5">
          <p className="text-[13px] font-semibold text-muted">{new Date(`${today}T12:00:00`).toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" })}</p>
          <h1 className="text-[32px] font-extrabold tracking-[-1px]">Today&apos;s session</h1>
        </header>

        {!currentTopic || !start ? (
          <>
            <span id="next" className="chip justify-self-start bg-accent-soft px-2.5 py-1 font-extrabold uppercase tracking-[0.09em] text-accent">Up next</span>
            <p className="text-muted">Everything available today is done. Take the recap.</p>
          </>
        ) : (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <span id="next" className="chip bg-accent-soft px-2.5 py-1 font-extrabold uppercase tracking-[0.09em] text-accent">Up next</span>
                <span className="text-[12.5px] font-semibold text-muted">Topic {currentTopic.id} · Phase {currentTopic.phase}</span>
              </div>
              <Link href={start.href} className="btn btn-primary gap-[7px]">
                {start.label}
                <ArrowRight className="size-[15px]" />
              </Link>
            </div>

            <div className="grid gap-2.5">
              <h2 className="text-[26px] font-bold leading-tight tracking-[-0.7px] text-balance">{currentTopic.title}</h2>
              <p className="max-w-[75ch] text-[14.5px] leading-[23px] text-muted first-letter:uppercase">{currentTopic.learnedWhen}</p>
            </div>

            <div className="mt-auto grid rounded-xl bg-subtle p-1.5">
              {teach ? (
                <>
                  <p className="px-3.5 pb-1 pt-2.5 text-[12.5px] font-semibold text-muted">What this lesson covers</p>
                  {currentTopic.teach.split(/;\s*/).map((c) => (
                    <p key={c} className="flex gap-[13px] px-3.5 py-2.5 text-sm font-medium">
                      <span className="text-accent" aria-hidden>›</span>
                      {c.replace(/\.$/, "")}
                    </p>
                  ))}
                </>
              ) : practiceItem ? (
                <div className="grid gap-1 px-3.5 py-3">
                  <p className="text-[12.5px] font-semibold text-muted">Practice in your own project</p>
                  <p className="text-sm font-medium">{currentTopic.practice}</p>
                </div>
              ) : (
                currentTopic.exerciseIds.map((id, i) => {
                  const status = exerciseStatus(exerciseStates.get(id), today);
                  const isNext = id === nextExerciseId;
                  const meta = isNext ? "Start now" : status === "done" ? "Done" : status === "waiting_retry" ? "Comes back later" : status === "in_progress" ? "In progress" : `Exercise ${i + 1}`;
                  return (
                    <Link key={id} href={`/exercises/${id}`} className={`flex items-center gap-[13px] rounded-[9px] px-3.5 py-[13px] transition-colors ${isNext ? "bg-panel" : "hover:bg-panel/60"}`}>
                      {status === "done" ? <CircleCheck className="size-[17px] text-accent" /> : isNext ? <CircleDot className="size-[17px]" /> : <Circle className="size-[17px] text-line" />}
                      <span className={`flex-1 text-sm ${isNext ? "font-bold" : status === "done" ? "font-medium text-muted" : "font-medium"}`}>{exercises.get(id)!.title}</span>
                      <span className={`text-[12.5px] font-semibold ${isNext ? "text-accent" : "text-muted"}`}>{meta}</span>
                    </Link>
                  );
                })
              )}
            </div>
          </>
        )}
      </section>

      <aside className="flex flex-col gap-4">
        <section className="panel grid gap-3.5 p-[22px]" aria-labelledby="plan">
          <p id="plan" className="eyebrow">Today&apos;s plan</p>
          <ol className="grid gap-3.5">
            {steps.map((s, i) => (
              <li key={s.label} className="flex items-center gap-[11px]">
                {s.done ? <CircleCheck className="size-4 text-accent" /> : i === currentStep ? <CircleDot className="size-4" /> : <Circle className="size-4 text-line" />}
                <span className={`flex-1 text-[13.5px] ${s.done ? "font-medium text-muted" : i === currentStep ? "font-bold" : "font-medium"}`}>{s.label}</span>
                <span className="text-xs font-semibold text-muted">{s.meta}</span>
              </li>
            ))}
          </ol>
        </section>

        <section className="panel grid gap-2.5 p-[22px]" aria-labelledby="reviews">
          <p id="reviews" className="eyebrow flex items-center gap-2"><RotateCcw className="size-3.5" />Spaced Reviews</p>
          {reviews.length === 0 ? (
            <p className="text-[13.5px] leading-[21px] text-muted">
              Nothing due today.{reviewsAll.length ? ` Next one: ${reviewsAll[0].due_date}.` : " Reviews start once you've learned your first topic."}
            </p>
          ) : (
            reviews.map((r) => <Row key={r.topicId} href={`/review/${r.topicId}`} title={`${r.topicId} · ${topicOf(r.topicId).title}`} chip="Due" />)
          )}
        </section>

        {retries.length > 0 && (
          <section className="panel grid gap-2.5 p-[22px]" aria-labelledby="retries">
            <p id="retries" className="eyebrow">Come-back Exercises</p>
            <p className="text-[13.5px] leading-[21px] text-muted">You solved these with help. Solve them again from scratch, no hints.</p>
            {retries.map((r) => <Row key={r.exerciseId} href={`/exercises/${r.exerciseId}`} title={exercises.get(r.exerciseId)!.title} chip="Retry due" />)}
          </section>
        )}

        <section className="flex flex-1 flex-col gap-[13px] rounded-2xl bg-accent p-[22px]" aria-labelledby="recap">
          <p id="recap" className="text-[10.5px] font-extrabold uppercase tracking-[0.09em] text-ground/70">Recap</p>
          <p className="text-[13.5px] leading-[21px] text-ground">Two minutes at the end: say what you learned and what was hardest.</p>
          <Link href="/recap" className="mt-auto self-start rounded-[9px] bg-ground px-[15px] py-[9px] text-[13px] font-bold text-ink transition-opacity hover:opacity-90">Open the recap</Link>
        </section>
      </aside>

      <CurriculumCards
        phases={phases}
        topics={topics}
        topicStates={topicStates}
        exerciseStates={exerciseStates}
        currentTopicId={topics.find((t) => !isTopicLearned(t, topicStates.get(t.id), exerciseStates, today))?.id ?? null}
        today={today}
      />
    </main>
  );
}

function Row({ href, title, chip }: { href: string; title: string; chip: string }) {
  return (
    <Link href={href} className="flex items-center justify-between gap-4 rounded-[10px] bg-subtle px-3 py-2.5 text-[13.5px] transition-colors hover:bg-accent-soft">
      <span className="font-semibold">{title}</span>
      <span className="chip shrink-0 bg-accent-soft text-accent">{chip}</span>
    </Link>
  );
}

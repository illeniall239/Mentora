import Link from "next/link";
import type { Phase, Topic } from "@/lib/curriculum.ts";
import type { ExerciseState, TopicState } from "@/lib/progress.ts";
import { isTopicLearned } from "@/lib/progress.ts";

type Props = {
  phases: Phase[];
  topics: Topic[];
  topicStates: Map<string, TopicState>;
  exerciseStates: Map<string, ExerciseState>;
  currentTopicId: string | null;
  today: string;
};

// The whole Curriculum Map: one card per phase, each opening its own page.
export function CurriculumCards({ phases, topics, topicStates, exerciseStates, currentTopicId, today }: Props) {
  const learned = (t: Topic) => isTopicLearned(t, topicStates.get(t.id), exerciseStates, today);
  const currentPhase = topics.find((t) => t.id === currentTopicId)?.phase ?? phases[0]?.number;

  return (
    <section className="grid gap-3.5" aria-labelledby="curriculum">
      <div className="flex items-center justify-between gap-4">
        <h2 id="curriculum" className="text-lg font-bold tracking-[-0.3px]">Curriculum</h2>
        <span className="text-[12.5px] font-medium text-muted">{phases.length} phases · {topics.length} topics</span>
      </div>

      <div className="grid gap-3.5 md:grid-cols-2">
        {phases.map((phase) => {
          const inPhase = topics.filter((t) => t.phase === phase.number);
          const done = inPhase.filter(learned).length;
          const isCurrent = phase.number === currentPhase;
          return (
            <Link
              key={phase.number}
              href={`/phases/${phase.number}`}
              className={`grid content-start gap-3.5 rounded-[14px] border bg-panel p-[18px] transition-colors hover:border-accent ${isCurrent ? "border-accent-line" : "border-line"}`}
            >
              <span className="flex items-center justify-between gap-2">
                <span className={`eyebrow ${isCurrent ? "text-accent" : ""}`}>Phase {phase.number}</span>
                {isCurrent ? (
                  <span className="chip bg-accent-soft text-accent">You are here</span>
                ) : (
                  <span className="text-[11.5px] font-semibold text-muted">{done}/{inPhase.length}</span>
                )}
              </span>
              <span className="text-[15px] font-semibold leading-5">{phase.title}</span>
              <span className="flex items-center gap-3">
                <span className="h-1 flex-1 overflow-hidden rounded-sm bg-subtle" aria-hidden>
                  <span className="block h-full bg-accent" style={{ width: `${(done / inPhase.length) * 100}%` }} />
                </span>
                <span className="text-[11.5px] font-semibold text-muted">{done}/{inPhase.length} learned</span>
              </span>
            </Link>
          );
        })}
      </div>
    </section>
  );
}

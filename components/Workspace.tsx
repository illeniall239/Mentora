"use client";
import { useEffect, useRef, useState, useTransition } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import Editor, { type OnMount } from "@monaco-editor/react";
import dynamic from "next/dynamic";
import { Chat, type ChatMessageView } from "./Chat";
import { Markdown } from "./Markdown";
import { BreakdownView } from "./BreakdownView";
import { OPEN_SKETCHPAD } from "./SketchpadLink";
import { markPlanDone, runExercise, saveCode, setLanguage, showNextHint, startRetry } from "@/app/actions";
import { PLAN_READY, PLAN_STAGES } from "@/lib/tutor";
import type { ExerciseStatus } from "@/lib/progress";
import type { Language } from "@/lib/languages";

type Props = {
  exercise: { id: string; title: string; difficulty: string; body: string; hintCount: number; starter: string; starters: Record<Language, string> };
  topic: { id: string; title: string };
  language: Language;
  languageOptions: { id: Language; label: string; note: string; editorFile: string; monaco: string }[];
  nextExerciseId: string | null;
  initial: {
    code: string;
    sketch: string | null;
    planDone: boolean;
    planText: string | null;
    planChat: ChatMessageView[];
    hints: string[];
    status: ExerciseStatus;
    retryDue: string | null;
    workedExample: string | null;
    exerciseChat: ChatMessageView[];
    breakdown: string | null;
  };
};

const SketchPad = dynamic(() => import("./SketchPad"), { ssr: false, loading: () => <p className="p-6 text-sm text-muted">Loading the sketchpad…</p> });

// The Exercise is done in this order; each stage is its own screen.
type Stage = "problem" | "plan" | "code" | "breakdown";
const STAGES: { id: Stage; label: string }[] = [
  { id: "problem", label: "Problem" },
  { id: "plan", label: "Plan" },
  { id: "code", label: "Code" },
  { id: "breakdown", label: "Breakdown" },
];
type SideTab = "tutor" | "plan" | "problem" | "hints";

export function Workspace({ exercise, topic, nextExerciseId, initial, language: initialLanguage, languageOptions }: Props) {
  const router = useRouter();
  const [code, setCode] = useState(initial.code);
  const [language, setLang] = useState<Language>(initialLanguage);
  const [status, setStatus] = useState(initial.status);
  const [hints, setHints] = useState(initial.hints);
  const [worked, setWorked] = useState(initial.workedExample);
  const [workedLoading, setWorkedLoading] = useState(false);
  const [run, setRun] = useState<{ passed: boolean; output: string } | null>(null);
  const [running, setRunning] = useState(false);
  const [planDone, setPlanDone] = useState(initial.planDone);
  const [planReady, setPlanReady] = useState(initial.planChat.some((m) => m.role === "tutor" && m.content.includes(PLAN_READY)));
  const stageOf = (text: string) => Number(text.match(/Stage (\d) of 6/)?.[1] ?? 0);
  const [planStage, setPlanStage] = useState(Math.max(1, ...initial.planChat.filter((m) => m.role === "tutor").map((m) => stageOf(m.content))));
  const [planText, setPlanText] = useState(initial.planText);
  const [topicLearned, setTopicLearned] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryStarted, setRetryStarted] = useState(false);
  const [sideTab, setSideTab] = useState<SideTab>("tutor");
  const [sketchOpen, setSketchOpen] = useState(false);
  const [sketchMounted, setSketchMounted] = useState(false);
  const openSketch = (open: boolean) => { setSketchOpen(open); if (open) setSketchMounted(true); };
  const [stage, setStage] = useState<Stage>(
    initial.status === "done" || initial.status === "waiting_retry" ? "breakdown"
      : initial.planDone ? "code"
      : initial.planChat.length > 0 ? "plan"
      : "problem",
  );
  // The top bar's Sketchpad link toggles this exercise's sketchpad.
  useEffect(() => {
    const toggle = () => { setSketchMounted(true); setSketchOpen((o) => !o); };
    window.addEventListener(OPEN_SKETCHPAD, toggle);
    return () => window.removeEventListener(OPEN_SKETCHPAD, toggle);
  }, []);
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const locked = status === "done" || status === "waiting_retry";
  const editable = !locked && planDone;
  const solved = status === "done" || status === "waiting_retry";
  const reachable: Record<Stage, boolean> = { problem: true, plan: true, code: planDone || locked, breakdown: solved };

  // Autosave the Learner's code a second after they stop typing.
  useEffect(() => {
    if (code === initial.code) return;
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => void saveCode(exercise.id, code), 1000);
    return () => clearTimeout(saveTimer.current);
  }, [code, exercise.id, initial.code]);

  const onMount: OnMount = (_editor, monaco) => {
    const ts = monaco.languages.typescript;
    ts.typescriptDefaults.setCompilerOptions({ ...ts.typescriptDefaults.getCompilerOptions(), strict: true, target: ts.ScriptTarget.ES2022, lib: ["es2022", "dom"] });
  };

  async function onRun() {
    setRunning(true);
    setError(null);
    try {
      const result = await runExercise(exercise.id, code);
      setRun(result);
      setStatus(result.status);
      if (result.passed) {
        setTopicLearned(result.topicLearned);
        setStage("breakdown");
      }
      router.refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  }

  async function onHint() {
    setHints(await showNextHint(exercise.id));
  }

  async function onWorkedExample() {
    setWorkedLoading(true);
    setError(null);
    let text = "";
    try {
      const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind: "worked", id: exercise.id }) });
      const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += value;
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const ev = JSON.parse(buf.slice(0, nl));
          buf = buf.slice(nl + 1);
          if (ev.error) throw new Error(ev.error);
          if (ev.text) setWorked((text += ev.text));
        }
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setWorkedLoading(false);
    }
  }

  async function onLanguage(next: Language) {
    setLang(next);
    await setLanguage(next);
    // An untouched starter follows the language; code you've written is left alone.
    if (Object.values(exercise.starters).some((s) => s.trim() === code.trim())) {
      setCode(exercise.starters[next]);
      await saveCode(exercise.id, exercise.starters[next]);
    }
    router.refresh();
  }

  async function onStartCoding() {
    try {
      setPlanText(await markPlanDone(exercise.id));
    } catch (e) {
      setError((e as Error).message);
      return;
    }
    setPlanDone(true);
    setStage("code");
    setSideTab("plan");
    router.refresh();
  }

  async function onStartRetry() {
    await startRetry(exercise.id);
    setCode(exercise.starters[language]);
    setHints([]);
    setWorked(null);
    setRun(null);
    setStatus("retry_due");
    setRetryStarted(true);
    setPlanDone(false);
    setPlanReady(false);
    setPlanStage(1);
    setPlanText(null);
    setStage("problem");
    router.refresh();
  }

  const allHintsUsed = hints.length >= exercise.hintCount;
  const lang = languageOptions.find((l) => l.id === language) ?? languageOptions[0];
  const retryPending = status === "retry_due" && !retryStarted && planDone;

  return (
    <main className="flex flex-col lg:h-full">
      {/* Header: where you are in the Exercise */}
      <header className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 border-b border-line bg-panel px-6 py-3">
        <div className="grid min-w-0 gap-0.5">
          <p className="eyebrow truncate">
            <Link href={`/topics/${topic.id}`} className="hover:underline">Topic {topic.id} · {topic.title}</Link>
            {exercise.difficulty && <span className="ml-2 text-muted">Difficulty {exercise.difficulty}</span>}
          </p>
          <h1 className="font-display text-2xl font-bold">{exercise.title}</h1>
        </div>
        <nav aria-label="Exercise stages">
          <ol className="flex items-center gap-1">
            {STAGES.map((s, i) => {
              const active = stage === s.id;
              return (
                <li key={s.id} className="flex items-center gap-1">
                  {i > 0 && <span className="h-px w-5 bg-line" aria-hidden />}
                  <button
                    onClick={() => { setStage(s.id); openSketch(false); }}
                    disabled={!reachable[s.id]}
                    aria-current={active && !sketchOpen ? "step" : undefined}
                    className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${active && !sketchOpen ? "bg-accent text-panel" : "text-muted hover:bg-ground hover:text-ink"}`}
                  >
                    <span className={`grid size-5 place-items-center rounded-full font-mono text-[11px] ${active ? "bg-panel text-accent" : "bg-line"}`}>{i + 1}</span>
                    {s.label}
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>
        {sketchOpen && (
          <button onClick={() => openSketch(false)} className="btn border-accent bg-accent-soft text-accent">
            Back to {STAGES.find((s) => s.id === stage)?.label}
          </button>
        )}
      </header>

      {error && <p className="mx-6 mt-3 rounded bg-bad-soft px-3 py-2 text-sm" role="alert">{error}</p>}

      {/* Sketchpad: independent of the stages; the problem stays in view */}
      {sketchMounted && (
        <section hidden={!sketchOpen} className="min-h-0 flex-1 p-4 lg:px-6">
          <div className="grid h-full gap-4 lg:grid-cols-[minmax(260px,1fr)_minmax(0,3fr)]">
            <aside className="panel min-h-0 overflow-y-auto p-5">
              <p className="eyebrow mb-3">The problem</p>
              <Markdown text={exercise.body} />
            </aside>
            <SketchPad exerciseId={exercise.id} initialScene={initial.sketch} />
          </div>
        </section>
      )}

      {/* 1 · Problem: read it properly first */}
      <section hidden={sketchOpen || stage !== "problem"} className="min-h-0 flex-1 overflow-y-auto px-6 py-8">
        <div className="mx-auto grid max-w-3xl gap-6">
          <StatusLine status={status} retryDue={initial.retryDue} />
          <div className="panel p-8">
            <Markdown text={exercise.body} className="text-base" />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-muted">Read it until you could explain it to someone else. Don&apos;t think about code yet.</p>
            <button onClick={() => setStage(planDone ? "code" : "plan")} className="btn btn-primary px-5 py-3 text-base">
              {planDone ? "Back to coding" : "I've read it — plan it"}
            </button>
          </div>
        </div>
      </section>

      {/* 2 · Plan: the problem stays in view beside the planning session */}
      <section hidden={sketchOpen || stage !== "plan"} className="min-h-0 flex-1 p-4 lg:px-6">
        <div className="grid h-full gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <aside className="panel min-h-0 overflow-y-auto p-6">
            <p className="eyebrow mb-3">The problem</p>
            <Markdown text={exercise.body} />
          </aside>
          <div className="panel flex min-h-[70vh] flex-col gap-3 p-5 lg:min-h-0">
            <div className="grid gap-2">
              <p className="font-display text-lg font-semibold">Planning session</p>
              <ol className="flex flex-wrap gap-1.5" aria-label="Planning stages">
                {PLAN_STAGES.map((name, i) => (
                  <li key={name} className={`chip ${planDone || planReady || i + 1 < planStage ? "bg-accent text-panel" : i + 1 === planStage ? "border border-accent bg-panel text-accent" : "bg-line text-muted"}`} aria-current={i + 1 === planStage ? "step" : undefined}>
                    {i + 1}. {name}
                  </li>
                ))}
              </ol>
            </div>
            <div className="min-h-0 flex-1">
              <Chat
                kind="plan"
                id={exercise.id}
                initial={initial.planChat}
                autoStart={!planDone && stage === "plan"}
                placeholder="Answer the Tutor for this stage… (Enter to send, Shift+Enter for a new line)"
                onTutorReply={(t) => { if (stageOf(t)) setPlanStage(stageOf(t)); if (t.includes(PLAN_READY)) setPlanReady(true); }}
              />
            </div>
            <div className="flex items-center justify-between gap-3 border-t border-line pt-3">
              <p className="text-sm text-muted">
                {planDone ? "Plan approved." : planReady ? "The Tutor approved your plan." : `Stage ${planStage} of 6 · ${PLAN_STAGES[planStage - 1]}. The editor unlocks when your final plan is approved.`}
              </p>
              {planDone ? (
                <button onClick={() => setStage("code")} className="btn btn-primary">Go to the editor</button>
              ) : (
                <button onClick={onStartCoding} disabled={!planReady} className="btn btn-primary">Start coding</button>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* 4 · Breakdown: the full write-up, with the Tutor beside it for follow-up questions */}
      <section hidden={sketchOpen || stage !== "breakdown"} className="min-h-0 flex-1 p-4 lg:px-6">
        <div className="grid h-full gap-4 lg:grid-cols-[minmax(0,1fr)_420px]">
          <div className="flex min-h-0 flex-col gap-3">
            {topicLearned && <p className="rounded-md bg-accent-soft px-4 py-2 text-[15px] font-semibold" role="status">Topic {topic.id} is now learned. Its first spaced review is tomorrow.</p>}
            {status === "waiting_retry" && <p className="rounded-md bg-warn-soft px-4 py-2 text-sm">You used help on this one, so it comes back on {initial.retryDue ?? "a later day"} to solve again from scratch.</p>}
            <BreakdownView exerciseId={exercise.id} initial={initial.breakdown} active={stage === "breakdown" && solved} />
            <div className="flex flex-wrap gap-2">
              {nextExerciseId && <Link href={`/exercises/${nextExerciseId}`} className="btn btn-primary">Next exercise →</Link>}
              <Link href="/" className="btn">Back to Home</Link>
              <button onClick={() => setStage("code")} className="btn">See your code</button>
            </div>
          </div>
          <aside className="panel flex min-h-[60vh] flex-col p-4 lg:min-h-0">
            <p className="mb-2 font-display text-lg font-semibold">Questions about it?</p>
            <div className="min-h-0 flex-1">
              <Chat kind="exercise" id={exercise.id} initial={initial.exerciseChat} placeholder="Ask about the breakdown, the reference solution, or your code…" getContext={() => ({ code, lastRun: run })} />
            </div>
          </aside>
        </div>
      </section>

      {/* 3 · Code: big editor, Tutor in the sidebar */}
      <section hidden={sketchOpen || stage !== "code"} className="min-h-0 flex-1 p-4 lg:px-6">
        <div className="grid h-full gap-4 lg:grid-cols-[minmax(0,1fr)_420px]">
          <div className="flex min-h-0 flex-col gap-3">
            <div className="flex flex-wrap items-center gap-2">
              {retryPending ? (
                <button onClick={onStartRetry} className="btn btn-primary">Start the retry from scratch</button>
              ) : (
                <button onClick={onRun} disabled={running || !editable} className="btn btn-primary">{running ? "Running tests…" : "Run tests"}</button>
              )}
              {status === "done" && nextExerciseId && <Link href={`/exercises/${nextExerciseId}`} className="btn">Next exercise →</Link>}
              {status === "done" && <Link href="/" className="btn">Back to Home</Link>}
              <span className="text-sm text-muted"><StatusLine status={status} retryDue={initial.retryDue} inline /></span>
              <label className="ml-auto flex items-center gap-2 text-sm text-muted">
                Language
                <select id="language" value={language} onChange={(e) => void onLanguage(e.target.value as Language)} disabled={locked} className="field py-1.5 text-sm">
                  {languageOptions.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                </select>
              </label>
            </div>
            <div className="panel min-h-[480px] flex-1 overflow-hidden py-2">
              <Editor
                language={lang.monaco}
                path={`file:///${exercise.id}/${lang.editorFile}`}
                value={code}
                onChange={(v) => setCode(v ?? "")}
                onMount={onMount}
                options={{ minimap: { enabled: false }, fontSize: 15, fontFamily: "var(--font-code), monospace", scrollBeyondLastLine: false, readOnly: !editable, tabSize: 2 }}
                theme="light"
              />
            </div>
            <Markdown text={lang.note} className="text-xs text-muted" />
            {run && (
              <div className={`panel max-h-[32vh] shrink-0 overflow-auto p-3 ${run.passed ? "border-accent" : "border-bad"}`}>
                <p className={`mb-2 font-semibold ${run.passed ? "text-accent" : "text-bad"}`}>{run.passed ? "All tests pass." : "Some tests fail."}</p>
                <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed">{run.output}</pre>
              </div>
            )}
          </div>

          <aside className="panel flex min-h-[70vh] flex-col lg:min-h-0">
            <>
                <div className="flex gap-1 border-b border-line p-2" role="tablist">
                  {([["tutor", "Tutor"], ["plan", "Your plan"], ["problem", "Problem"], ["hints", `Hints ${hints.length}/${exercise.hintCount}`]] as const).map(([id, label]) => (
                    <button key={id} role="tab" aria-selected={sideTab === id} onClick={() => setSideTab(id)} className={`rounded-md px-3 py-1.5 text-sm font-semibold ${sideTab === id ? "bg-accent-soft text-accent" : "text-muted hover:text-ink"}`}>
                      {label}
                    </button>
                  ))}
                </div>
                <div hidden={sideTab !== "tutor"} className="min-h-0 flex-1 p-4">
                  <Chat
                    kind="exercise"
                    id={exercise.id}
                    initial={initial.exerciseChat}
                    placeholder="Ask about your approach, an error, or a concept…"
                    emptyText="Stuck? Ask. The Tutor sees your code, your plan and the last test run, and answers with questions, not code."
                    getContext={() => ({ code, lastRun: run })}
                  />
                </div>
                <div hidden={sideTab !== "plan"} className="min-h-0 flex-1 overflow-y-auto p-5">
                  {planText ? <Markdown text={planText} /> : <p className="text-sm text-muted">No saved plan for this exercise.</p>}
                  <p className="mt-4 border-t border-line pt-3 text-xs text-muted">Code one step at a time. If a step turns out wrong, change it here in your head and tell the Tutor why.</p>
                </div>
                <div hidden={sideTab !== "problem"} className="min-h-0 flex-1 overflow-y-auto p-5">
                  <Markdown text={exercise.body} />
                </div>
                <div hidden={sideTab !== "hints"} className="min-h-0 flex-1 overflow-y-auto p-5">
                  <div className="grid gap-3">
                    <p className="text-sm text-muted">Using any hint means this exercise comes back in 3 days to be solved again without help.</p>
                    {hints.map((h, i) => (
                      <div key={i} className="rounded-md bg-warn-soft px-3 py-2 text-[15px]">
                        <span className="mr-2 font-mono text-xs font-semibold text-warn">Hint {i + 1}</span>
                        <Markdown text={h} className="inline" />
                      </div>
                    ))}
                    {editable && !allHintsUsed && <button onClick={onHint} className="btn justify-self-start">Show hint {hints.length + 1} of {exercise.hintCount}</button>}
                    {editable && allHintsUsed && !worked && (
                      <button onClick={onWorkedExample} disabled={workedLoading} className="btn justify-self-start">
                        {workedLoading ? "Writing a Worked Example…" : "Still stuck: show a Worked Example"}
                      </button>
                    )}
                    {worked && (
                      <div className="rounded-md border border-line p-4">
                        <p className="eyebrow mb-2">Worked Example: a different problem, same technique</p>
                        <Markdown text={worked} />
                      </div>
                    )}
                  </div>
                </div>
              </>
          </aside>
        </div>
      </section>
    </main>
  );
}

function StatusLine({ status, retryDue, inline = false }: { status: ExerciseStatus; retryDue: string | null; inline?: boolean }) {
  const text: Record<ExerciseStatus, string> = {
    new: "Not started.",
    in_progress: "In progress.",
    waiting_retry: `Solved with help. It comes back on ${retryDue} to be solved again without help.`,
    retry_due: "Come-back day: solve it again from scratch, with no hints.",
    done: "Done: passed without help and explained.",
  };
  return inline ? <>{text[status]}</> : <p className="text-sm text-muted">{text[status]}</p>;
}

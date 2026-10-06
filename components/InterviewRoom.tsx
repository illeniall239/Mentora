"use client";
import { useEffect, useRef, useState, useTransition } from "react";
import Link from "next/link";
import Editor from "@monaco-editor/react";
import { Chat, type ChatMessageView } from "./Chat";
import { finishInterview, saveInterviewDraft } from "@/app/actions";
import type { Feedback } from "@/lib/interview";

type Props = {
  id: number;
  label: string;
  minutes: number;
  editor: boolean;
  startedAt: string;
  language: { monaco: string; label: string; file: string };
  initialCode: string;
  initialMessages: ChatMessageView[];
  initialFeedback: Feedback | null;
};

function useElapsed(startedAt: string, stopped: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (stopped) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [stopped]);
  return Math.max(0, Math.floor((now - new Date(startedAt).getTime()) / 1000));
}

export function InterviewRoom({ id, label, minutes, editor, startedAt, language, initialCode, initialMessages, initialFeedback }: Props) {
  const [code, setCode] = useState(initialCode);
  const [feedback, setFeedback] = useState(initialFeedback);
  const [error, setError] = useState<string | null>(null);
  const [ending, startEnding] = useTransition();
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const ended = !!feedback;
  const elapsed = useElapsed(startedAt, ended);
  const left = minutes * 60 - elapsed;
  const clock = (s: number) => `${Math.floor(Math.abs(s) / 60)}:${String(Math.abs(s) % 60).padStart(2, "0")}`;

  useEffect(() => {
    if (!editor || ended || code === initialCode) return;
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => void saveInterviewDraft(id, code), 1000);
    return () => clearTimeout(saveTimer.current);
  }, [code, editor, ended, id, initialCode]);

  function end() {
    if (!confirm("End the interview and get feedback?")) return;
    startEnding(async () => {
      try {
        setFeedback(await finishInterview(id));
      } catch (e) {
        setError((e as Error).message);
      }
    });
  }

  const chat = (
    <Chat
      kind="interview"
      id={String(id)}
      initial={initialMessages}
      autoStart={!ended}
      placeholder={ended ? "This interview has ended." : "Talk to the interviewer… (the mic button lets you answer out loud)"}
      getContext={editor ? () => ({ code }) : undefined}
    />
  );

  return (
    <main className="flex flex-col lg:h-full">
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-line bg-panel px-6 py-3">
        <div className="grid gap-0.5">
          <p className="eyebrow"><Link href="/interview" className="hover:underline">Interview practice</Link></p>
          <h1 className="font-display text-2xl font-bold">{label}</h1>
        </div>
        <div className="flex items-center gap-4">
          {!ended && (
            <span className={`font-mono text-2xl tabular-nums ${left < 0 ? "text-bad" : left < 300 ? "text-warn" : ""}`} aria-label="Time left">
              {left < 0 ? `+${clock(left)}` : clock(left)}
            </span>
          )}
          {!ended && <button onClick={end} disabled={ending} className="btn btn-primary">{ending ? "Writing feedback…" : "End interview"}</button>}
        </div>
      </header>
      {error && <p className="mx-6 mt-3 rounded bg-bad-soft px-3 py-2 text-sm" role="alert">{error}</p>}

      <div className={`grid min-h-0 flex-1 gap-4 p-4 lg:px-6 ${editor || feedback ? "lg:grid-cols-[minmax(0,1fr)_440px]" : ""}`}>
        {editor ? (
          <div className="flex min-h-0 flex-col gap-2">
            <p className="text-sm text-muted">{language.label} · the interviewer sees this code whenever you send a message</p>
            <div className="panel min-h-[480px] flex-1 overflow-hidden py-2">
              <Editor
                language={language.monaco}
                path={`file:///interview-${id}/${language.file}`}
                value={code}
                onChange={(v) => setCode(v ?? "")}
                options={{ minimap: { enabled: false }, fontSize: 15, fontFamily: "var(--font-code), monospace", scrollBeyondLastLine: false, readOnly: ended, tabSize: 2 }}
                theme="light"
              />
            </div>
          </div>
        ) : (
          <div className="panel flex min-h-[70vh] flex-col p-5 lg:min-h-0">{chat}</div>
        )}

        {(editor || feedback) && (
          <aside className="panel flex min-h-[70vh] flex-col p-5 lg:min-h-0">
            {feedback ? <FeedbackView feedback={feedback} /> : chat}
          </aside>
        )}
      </div>
    </main>
  );
}

function FeedbackView({ feedback }: { feedback: Feedback }) {
  return (
    <div className="grid content-start gap-4 overflow-y-auto">
      <div className="grid gap-1">
        <p className="eyebrow">Feedback</p>
        <p className="font-display text-2xl font-bold capitalize">{feedback.rating}</p>
      </div>
      <p className="text-[15px]">{feedback.summary}</p>
      <List title="Strengths" items={feedback.strengths} />
      <List title="Practise next" items={feedback.improvements} />
      {feedback.mistakes.length > 0 && (
        <p className="text-sm text-muted">
          {feedback.mistakes.length} mistake{feedback.mistakes.length === 1 ? "" : "s"} added to your <Link href="/mistakes" className="text-accent underline">mistake log</Link>.
        </p>
      )}
      <Link href="/interview" className="btn justify-self-start">Back to interviews</Link>
    </div>
  );
}

function List({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="grid gap-1">
      <p className="text-sm font-semibold">{title}</p>
      <ul className="grid list-disc gap-1 pl-5 text-[15px]">{items.map((s, i) => <li key={i}>{s}</li>)}</ul>
    </div>
  );
}

"use client";
import { useEffect, useRef, useState } from "react";
import { Markdown } from "./Markdown";
import { MicButton } from "./MicButton";
import type { ChatKind, RunInfo } from "@/lib/tutor";

// One-click requests for an explanation instead of a question (see "Explaining a concept" in lib/tutor.ts).
const QUICK_ASKS: [string, string][] = [
  ["Explain simply", "Explain the idea we're on simply, in plain words."],
  ["Like I'm 10", "Explain it like I'm 10 years old: an everyday analogy first, no jargon until it clicks."],
  ["Give an analogy", "Give me a different analogy for this, and say where it breaks down."],
  ["Show an example", "Walk me through a tiny concrete example with real values, step by step."],
];

export type ChatMessageView = { role: "learner" | "tutor"; content: string; provider?: string | null };

type Props = {
  kind: ChatKind;
  id: string;
  initial: ChatMessageView[];
  autoStart?: boolean; // ask the Tutor to open the conversation when it's empty
  placeholder?: string;
  getContext?: () => { code?: string; lastRun?: RunInfo };
  onTutorReply?: (text: string) => void;
  emptyText?: string;
};

export function Chat({ kind, id, initial, autoStart = false, placeholder = "Reply to the Tutor…", getContext, onTutorReply, emptyText }: Props) {
  const [messages, setMessages] = useState<ChatMessageView[]>(initial);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const started = useRef(false);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // The box grows with the draft so a long question stays visible start to finish; the CSS max-height caps it.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + el.offsetHeight - el.clientHeight}px`; // + the borders, since height is border-box
  }, [draft]);

  async function send(message?: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    if (message) setMessages((m) => [...m, { role: "learner", content: message }]);
    setMessages((m) => [...m, { role: "tutor", content: "" }]);
    let text = "";
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, id, message, ...getContext?.() }),
      });
      if (!res.ok || !res.body) throw new Error(await res.text());
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
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
          if (ev.notice) setNotice(ev.notice);
          if (ev.text || ev.provider) {
            if (ev.text) text += ev.text;
            setMessages((m) => {
              const last = m[m.length - 1];
              return [...m.slice(0, -1), { ...last, content: text, provider: ev.provider ?? last.provider }];
            });
          }
        }
      }
      onTutorReply?.(text);
    } catch (e) {
      setError((e as Error).message);
      setMessages((m) => (m[m.length - 1]?.role === "tutor" && !m[m.length - 1].content ? m.slice(0, -1) : m));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (autoStart && !started.current && initial.length === 0) {
      started.current = true;
      void send();
    }
  }, [autoStart, kind, id]);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const message = draft.trim();
    if (!message || busy) return;
    setDraft("");
    void send(message);
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1" aria-live="polite">
        {messages.length === 0 && !busy && emptyText && <p className="text-sm text-muted">{emptyText}</p>}
        {messages.map((m, i) => (
          <div key={i} className={m.role === "learner" ? "ml-8 rounded-lg bg-accent-soft px-3 py-2" : ""}>
            <div className="mb-1 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted">
              {m.role === "learner" ? "You" : "Tutor"}
              {m.provider && <span className="rounded bg-line px-1.5 py-px font-mono normal-case tracking-normal">{m.provider}</span>}
            </div>
            {!m.content ? (
              <p className="animate-pulse text-sm text-muted">Thinking…</p>
            ) : m.role === "learner" ? (
              // The Learner's own words, exactly as typed: their line breaks are part of the question.
              <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{m.content}</p>
            ) : (
              <Markdown text={m.content} />
            )}
          </div>
        ))}
        <div ref={bottom} />
      </div>
      {notice && <p className="mt-2 rounded bg-warn-soft px-3 py-2 text-sm">{notice}</p>}
      {error && <p className="mt-2 rounded bg-bad-soft px-3 py-2 text-sm" role="alert">{error}</p>}
      {kind !== "interview" && kind !== "breakdown" && kind !== "worked" && (
        <div className="mt-3 flex flex-wrap gap-1.5" aria-label="Ask the Tutor to explain">
          {QUICK_ASKS.map(([label, text]) => (
            <button key={label} type="button" disabled={busy} onClick={() => void send(text)} className="rounded-full border border-line px-3 py-1 text-xs font-semibold text-muted transition-colors hover:border-accent hover:text-ink disabled:opacity-50">
              {label}
            </button>
          ))}
        </div>
      )}
      <form onSubmit={submit} className="mt-2 flex gap-2">
        <label htmlFor={`chat-${kind}-${id}`} className="sr-only">Message</label>
        <textarea
          id={`chat-${kind}-${id}`}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) submit(e); }}
          placeholder={placeholder}
          ref={box}
          rows={2}
          className="field max-h-[50vh] min-h-[4.5rem] flex-1 resize-none overflow-y-auto"
        />
        <MicButton onText={(t) => setDraft((d) => (d.trim() ? `${d.trimEnd()} ${t}` : t))} />
        <button type="submit" disabled={busy || !draft.trim()} className="btn btn-primary self-end">Send</button>
      </form>
    </div>
  );
}

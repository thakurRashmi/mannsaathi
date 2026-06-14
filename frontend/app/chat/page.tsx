"use client";

import { useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
  // When set, the assistant message is a crisis escalation — render with
  // the prominent helpline treatment instead of a normal bubble.
  isCrisis?: boolean;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hi, I'm here to listen. What's on your mind today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function send() {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const next: Message[] = [...messages, { role: "user", content: trimmed }];
    setMessages(next);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, history: messages }),
      });
      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.reply ?? "(no reply)",
          isCrisis: Boolean(data.is_crisis),
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            "Something went wrong reaching the server. Take a breath — try again in a moment.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen flex flex-col bg-warm-50">
      <header className="px-6 py-4 border-b border-warm-100 bg-white">
        <h1 className="text-xl font-semibold text-sage-700">MannSaathi</h1>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-6 max-w-2xl w-full mx-auto space-y-4">
        {messages.map((m, i) => {
          if (m.role === "user") {
            return (
              <div key={i} className="flex justify-end">
                <div className="rounded-2xl px-4 py-3 max-w-[80%] text-sm leading-relaxed bg-sage-500 text-white">
                  {m.content}
                </div>
              </div>
            );
          }

          // Assistant message — either crisis card or normal bubble.
          if (m.isCrisis) {
            return <CrisisCard key={i} content={m.content} />;
          }

          return (
            <div key={i} className="flex justify-start">
              <div className="rounded-2xl px-4 py-3 max-w-[80%] text-sm leading-relaxed bg-white text-gray-800 shadow-sm">
                {m.content}
              </div>
            </div>
          );
        })}
        {loading && (
          <div className="text-sm text-gray-400 italic">listening…</div>
        )}
      </div>

      <footer className="border-t border-warm-100 bg-white px-4 py-3">
        <div className="max-w-2xl mx-auto flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Type what's on your mind…"
            className="flex-1 rounded-full border border-warm-100 px-4 py-2 text-sm focus:outline-none focus:border-sage-500"
            disabled={loading}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className="bg-sage-500 hover:bg-sage-700 disabled:bg-gray-300 text-white px-5 py-2 rounded-full text-sm font-medium transition-colors"
          >
            Send
          </button>
        </div>
        <p className="text-[10px] text-gray-400 text-center mt-2 max-w-2xl mx-auto">
          MannSaathi is an AI companion, not a therapist. In crisis, call iCall:
          9152987821.
        </p>
      </footer>
    </main>
  );
}

/**
 * CrisisCard renders the safety escalation. Visually distinct from chat
 * bubbles so the user immediately understands this is different — warm
 * background, larger, helpline numbers as tap-to-call links.
 */
function CrisisCard({ content }: { content: string }) {
  // The backend sends the canned text with newlines + emoji markers.
  // We split into lines and detect helpline rows ("📞 Name: number").
  const lines = content.split("\n").filter((l) => l.trim().length > 0);

  return (
    <div className="flex justify-start">
      <div className="w-full max-w-xl rounded-2xl border-2 border-rose-200 bg-rose-50 px-5 py-4 shadow-sm">
        <div className="text-sm leading-relaxed text-gray-800 space-y-2">
          {lines.map((line, i) => {
            // Helpline lines look like "📞 iCall (free): 9152987821"
            const phoneMatch = line.match(/(\+?\d[\d\-\s]{6,})$/);
            if (line.startsWith("📞") && phoneMatch) {
              const phone = phoneMatch[1].replace(/[\s\-]/g, "");
              const labelPart = line.slice(0, line.lastIndexOf(phoneMatch[1])).trim();
              return (
                <div key={i} className="flex items-center justify-between gap-3 py-1">
                  <span className="text-gray-700">{labelPart}</span>
                  <a
                    href={`tel:${phone}`}
                    className="inline-block rounded-full bg-rose-500 hover:bg-rose-600 transition-colors text-white text-xs font-medium px-3 py-1.5 whitespace-nowrap"
                  >
                    Call {phoneMatch[1].trim()}
                  </a>
                </div>
              );
            }
            return <p key={i}>{line}</p>;
          })}
        </div>
      </div>
    </div>
  );
}

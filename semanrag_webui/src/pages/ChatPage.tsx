import { useState, useRef, useCallback, type KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import * as Tooltip from "@radix-ui/react-tooltip";
import * as Select from "@radix-ui/react-select";
import * as Dialog from "@radix-ui/react-dialog";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import {
  Send, RefreshCw, ThumbsUp, ThumbsDown, Download, Columns2,
  ChevronDown, Check, Star,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { useChatStore, type ChatMessage } from "@/stores/chat";
import { queryApi, feedbackApi, type GroundedClaim, type CompareResponse } from "@/lib/semanrag";

const MODES = [
  { value: "local", label: "Local", tip: "Entity-centric retrieval from local subgraph" },
  { value: "global", label: "Global", tip: "Relationship-centric retrieval across full graph" },
  { value: "hybrid", label: "Hybrid", tip: "Fused local + global retrieval" },
  { value: "naive", label: "Naive", tip: "Direct vector search on document chunks" },
  { value: "mix", label: "Mix", tip: "KG retrieval + naive chunk retrieval" },
  { value: "community", label: "Community", tip: "Query-matched community summaries" },
  { value: "bypass", label: "Bypass", tip: "Direct LLM call without RAG context" },
] as const;

function GroundedCheckBadge({ claim }: { claim: GroundedClaim }) {
  const colors = { supported: "success", partial: "warning", unsupported: "destructive" } as const;
  return (
    <Tooltip.Provider>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <div><Badge variant={colors[claim.status]} className="cursor-help text-[10px]">{claim.status}</Badge></div>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content className="z-50 max-w-xs rounded-sm bg-card border p-3 text-xs shadow-md" sideOffset={5}>
            <p className="font-medium mb-1">{claim.claim}</p>
            {claim.evidence && <p className="text-muted-foreground">{claim.evidence}</p>}
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}

function InlineReference({ reference }: { reference: Record<string, unknown> }) {
  return (
    <Tooltip.Provider>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <button className="inline text-xs text-eds-blue underline cursor-help mx-0.5">
            [{String(reference.source_id ?? reference.id ?? "ref")}]
          </button>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content className="z-50 max-w-sm rounded-sm bg-card border p-3 text-xs shadow-md" sideOffset={5}>
            <p className="font-medium">{String(reference.source_id ?? "")}</p>
            <p className="text-muted-foreground mt-1 line-clamp-4">{String(reference.content ?? reference.description ?? "")}</p>
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}

function QuerySettingsPanel() {
  const { topK, chunkTopK, enableRerank, snapshotAt, userGroups, setTopK, setChunkTopK, setEnableRerank, setSnapshotAt, setUserGroups } = useChatStore();

  return (
    <div className="explanation-panel mb-4">
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <Label className="text-xs font-semibold uppercase text-muted-foreground">top_k</Label>
          <Input type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} className="w-20 h-8" min={1} max={100} />
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-xs font-semibold uppercase text-muted-foreground">chunk_top_k</Label>
          <Input type="number" value={chunkTopK} onChange={(e) => setChunkTopK(Number(e.target.value))} className="w-20 h-8" min={1} max={50} />
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-xs font-semibold uppercase text-muted-foreground">Reranker</Label>
          <SwitchPrimitive.Root
            checked={enableRerank}
            onCheckedChange={setEnableRerank}
            className="w-9 h-5 rounded-full bg-eds-gray-300 data-[state=checked]:bg-eds-blue transition-colors"
          >
            <SwitchPrimitive.Thumb className="block h-4 w-4 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-4 translate-x-0.5" />
          </SwitchPrimitive.Root>
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-xs font-semibold uppercase text-muted-foreground">snapshot_at</Label>
          <Input type="datetime-local" value={snapshotAt ?? ""} onChange={(e) => setSnapshotAt(e.target.value || null)} className="h-8" />
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-xs font-semibold uppercase text-muted-foreground">ACL groups</Label>
          <Input
            value={userGroups.join(",")}
            onChange={(e) => setUserGroups(e.target.value ? e.target.value.split(",") : [])}
            placeholder="group1,group2"
            className="w-36 h-8"
          />
        </div>
      </div>
    </div>
  );
}

function FeedbackDialog({ queryId, onClose }: { queryId: string; onClose: () => void }) {
  const [relevance, setRelevance] = useState(3);
  const [accuracy, setAccuracy] = useState(3);
  const [faithfulness, setFaithfulness] = useState(3);
  const [comment, setComment] = useState("");

  const submit = async (thumbs: "up" | "down") => {
    await feedbackApi.submit({ query_id: queryId, thumbs, rating: { relevance, accuracy, faithfulness }, comment: comment || undefined });
    onClose();
  };

  return (
    <div className="space-y-4 p-5">
      <h3 className="font-bold text-base">Rate this answer</h3>
      {(["relevance", "accuracy", "faithfulness"] as const).map((key) => {
        const val = { relevance, accuracy, faithfulness }[key];
        const setter = { relevance: setRelevance, accuracy: setAccuracy, faithfulness: setFaithfulness }[key];
        return (
          <div key={key} className="flex items-center gap-3">
            <Label className="w-28 capitalize text-sm">{key}</Label>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((n) => (
                <button key={n} onClick={() => setter(n)} className={`p-0.5 ${n <= val ? "text-eds-orange" : "text-eds-gray-300"}`}>
                  <Star className="h-4 w-4 fill-current" />
                </button>
              ))}
            </div>
          </div>
        );
      })}
      <Textarea placeholder="Optional comment…" value={comment} onChange={(e) => setComment(e.target.value)} rows={2} />
      <div className="flex gap-2 justify-end">
        <Button variant="outline" size="sm" onClick={() => submit("down")}><ThumbsDown className="h-3 w-3 mr-1" /> Poor</Button>
        <Button size="sm" onClick={() => submit("up")}><ThumbsUp className="h-3 w-3 mr-1" /> Good</Button>
      </div>
    </div>
  );
}

function MessageBubble({ msg, onRegenerate, onFeedback }: { msg: ChatMessage; onRegenerate?: () => void; onFeedback?: () => void }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-[80%] rounded-sm px-4 py-3 ${isUser ? "bg-eds-blue text-white" : "bg-muted"}`}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight, rehypeKatex]} className="prose prose-sm dark:prose-invert max-w-none">
          {msg.content}
        </ReactMarkdown>

        {msg.references && msg.references.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2 pt-2 border-t border-white/20">
            {msg.references.map((r, i) => <InlineReference key={i} reference={r} />)}
          </div>
        )}

        {msg.groundedCheck && msg.groundedCheck.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {msg.groundedCheck.map((c, i) => <GroundedCheckBadge key={i} claim={c} />)}
          </div>
        )}

        {!isUser && (
          <div className="flex items-center gap-1 mt-2 pt-2 border-t border-border/50">
            {msg.latencyMs != null && <span className="text-[10px] text-muted-foreground mr-2">{msg.latencyMs.toFixed(0)}ms</span>}
            {msg.mode && <span className="status-badge info text-[10px]">{msg.mode}</span>}
            <div className="ml-auto flex gap-1">
              {onFeedback && (
                <>
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onFeedback}><ThumbsUp className="h-3 w-3" /></Button>
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onFeedback}><ThumbsDown className="h-3 w-3" /></Button>
                </>
              )}
              {onRegenerate && (
                <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onRegenerate}><RefreshCw className="h-3 w-3" /></Button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ComparePanel({ data, onClose }: { data: CompareResponse; onClose: () => void }) {
  return (
    <Dialog.Root open onOpenChange={onClose}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 z-50" />
        <Dialog.Content className="fixed inset-6 z-50 rounded-sm border bg-card shadow-lg overflow-auto">
          <div className="p-5">
            <Dialog.Title className="text-base font-bold mb-4">Compare Answers</Dialog.Title>
            <div className="grid grid-cols-2 gap-4">
              {(["result_a", "result_b"] as const).map((key) => {
                const r = data[key];
                return (
                  <div key={key} className="border rounded-sm p-4 border-t-[3px] border-t-eds-blue">
                    <h3 className="font-semibold mb-2 text-sm">{key === "result_a" ? "Variant A" : "Variant B"}</h3>
                    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight, rehypeKatex]} className="prose prose-sm dark:prose-invert max-w-none">
                      {r.answer}
                    </ReactMarkdown>
                    <div className="mt-3 text-xs text-muted-foreground">{r.latency_ms.toFixed(0)}ms · {r.references.length} refs</div>
                  </div>
                );
              })}
            </div>
            <div className="mt-4 flex justify-end">
              <Button variant="outline" onClick={onClose}>Close</Button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function exportChat(messages: ChatMessage[], format: "md" | "json") {
  let content: string;
  let mime: string;
  if (format === "json") {
    content = JSON.stringify(messages, null, 2);
    mime = "application/json";
  } else {
    content = messages.map((m) => `## ${m.role === "user" ? "User" : "Assistant"}\n\n${m.content}\n`).join("\n---\n\n");
    mime = "text/markdown";
  }
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `chat-export.${format}`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ChatPage() {
  const store = useChatStore();
  const { threads, activeThreadId, mode, topK, enableRerank, snapshotAt, userGroups } = store;
  const thread = threads.find((t) => t.id === activeThreadId);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [feedbackMsgId, setFeedbackMsgId] = useState<string | null>(null);
  const [compareData, setCompareData] = useState<CompareResponse | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => bottomRef.current?.scrollIntoView({ behavior: "smooth" });

  const ensureThread = useCallback(() => {
    if (activeThreadId && threads.find((t) => t.id === activeThreadId)) return activeThreadId;
    return store.createThread();
  }, [activeThreadId, threads, store]);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    const tid = ensureThread();

    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: text, timestamp: Date.now() };
    store.addMessage(tid, userMsg);

    const assistantId = crypto.randomUUID();
    const assistantMsg: ChatMessage = { id: assistantId, role: "assistant", content: "", mode, timestamp: Date.now() };
    store.addMessage(tid, assistantMsg);

    const history = threads.find((t) => t.id === tid)?.messages
      .filter((m) => m.id !== assistantId)
      .map((m) => ({ role: m.role, content: m.content })) ?? [];

    setStreaming(true);
    try {
      let fullContent = "";
      for await (const evt of queryApi.stream({
        query: text,
        mode,
        top_k: topK,
        conversation_history: history,
        enable_rerank: enableRerank,
        snapshot_at: snapshotAt,
        user_groups: userGroups,
      })) {
        if (evt.chunk) {
          fullContent += evt.chunk;
          store.updateMessage(tid, assistantId, { content: fullContent });
          scrollToBottom();
        }
        if (evt.done) {
          store.updateMessage(tid, assistantId, {
            content: fullContent,
            references: evt.references,
            latencyMs: evt.latency_ms,
          });
        }
      }

      try {
        const dataResult = await queryApi.queryData({
          query: text, mode, top_k: topK, enable_rerank: enableRerank,
          snapshot_at: snapshotAt, user_groups: userGroups,
        });
        store.updateMessage(tid, assistantId, { groundedCheck: dataResult.grounded_check });
      } catch {
        // Grounded check is optional
      }
    } catch (err) {
      store.updateMessage(tid, assistantId, { content: `Error: ${err instanceof Error ? err.message : "Unknown error"}` });
    } finally {
      setStreaming(false);
      scrollToBottom();
    }
  }, [input, streaming, mode, topK, enableRerank, snapshotAt, userGroups, ensureThread, store, threads]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      sendMessage();
    }
  };

  const regenerate = useCallback(async (msgId: string) => {
    if (!thread || streaming) return;
    const idx = thread.messages.findIndex((m) => m.id === msgId);
    if (idx < 1) return;
    const userMsg = thread.messages[idx - 1];
    if (!userMsg || userMsg.role !== "user") return;
    setInput(userMsg.content);
  }, [thread, streaming]);

  const handleCompare = useCallback(async () => {
    if (!thread || thread.messages.length === 0) return;
    const lastUserMsg = [...thread.messages].reverse().find((m) => m.role === "user");
    if (!lastUserMsg) return;
    try {
      const result = await queryApi.compare({
        query: lastUserMsg.content,
        variant_a: { mode: "local", top_k: topK, enable_rerank: enableRerank, response_type: "Multiple Paragraphs" },
        variant_b: { mode: "global", top_k: topK, enable_rerank: enableRerank, response_type: "Multiple Paragraphs" },
        user_groups: userGroups,
      });
      setCompareData(result);
    } catch {
      // Compare is optional
    }
  }, [thread, topK, enableRerank, userGroups]);

  return (
    <div className="flex h-full gap-4">
      {/* Thread list sidebar */}
      <div className="w-48 shrink-0 border-r pr-3 space-y-1 overflow-auto">
        <Button variant="outline" size="sm" className="w-full mb-3" onClick={() => store.createThread()}>
          New Chat
        </Button>
        {threads.map((t) => (
          <button
            key={t.id}
            onClick={() => store.setActiveThread(t.id)}
            className={`w-full text-left text-sm truncate rounded-sm px-3 py-2 transition-colors ${
              t.id === activeThreadId
                ? "bg-eds-blue-light text-eds-blue font-medium dark:bg-[#0d2137]"
                : "text-muted-foreground hover:bg-accent"
            }`}
          >
            {t.title}
          </button>
        ))}
      </div>

      {/* Chat area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Top controls */}
        <div className="flex items-center gap-3 mb-3 flex-wrap">
          <Select.Root value={mode} onValueChange={store.setMode}>
            <Select.Trigger className="inline-flex items-center gap-2 rounded-sm border px-3 py-1.5 text-sm bg-background hover:bg-accent transition-colors">
              <Select.Value />
              <ChevronDown className="h-3 w-3" />
            </Select.Trigger>
            <Select.Portal>
              <Select.Content className="z-50 rounded-sm border bg-card shadow-md">
                <Select.Viewport className="p-1">
                  {MODES.map((m) => (
                    <Tooltip.Provider key={m.value}>
                      <Tooltip.Root>
                        <Tooltip.Trigger asChild>
                          <Select.Item value={m.value} className="flex items-center gap-2 rounded-sm px-3 py-1.5 text-sm cursor-pointer outline-none data-[highlighted]:bg-accent">
                            <Select.ItemText>{m.label}</Select.ItemText>
                            <Select.ItemIndicator><Check className="h-3 w-3" /></Select.ItemIndicator>
                          </Select.Item>
                        </Tooltip.Trigger>
                        <Tooltip.Portal>
                          <Tooltip.Content side="right" className="z-[60] rounded-sm bg-card border px-2 py-1 text-xs shadow" sideOffset={8}>
                            {m.tip}
                          </Tooltip.Content>
                        </Tooltip.Portal>
                      </Tooltip.Root>
                    </Tooltip.Provider>
                  ))}
                </Select.Viewport>
              </Select.Content>
            </Select.Portal>
          </Select.Root>

          <Button variant="ghost" size="sm" onClick={() => setShowSettings(!showSettings)}>
            {showSettings ? "Hide Settings" : "Settings"}
          </Button>

          <div className="ml-auto flex gap-1">
            <Button variant="ghost" size="sm" onClick={handleCompare} disabled={!thread?.messages.length}>
              <Columns2 className="h-3 w-3 mr-1" /> Compare
            </Button>
            <Button variant="ghost" size="sm" onClick={() => thread && exportChat(thread.messages, "md")}>
              <Download className="h-3 w-3 mr-1" /> MD
            </Button>
            <Button variant="ghost" size="sm" onClick={() => thread && exportChat(thread.messages, "json")}>
              <Download className="h-3 w-3 mr-1" /> JSON
            </Button>
          </div>
        </div>

        {showSettings && <QuerySettingsPanel />}

        {/* Messages */}
        <div className="flex-1 overflow-auto py-4 space-y-1">
          {thread?.messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              msg={msg}
              onRegenerate={msg.role === "assistant" ? () => regenerate(msg.id) : undefined}
              onFeedback={msg.role === "assistant" ? () => setFeedbackMsgId(msg.id) : undefined}
            />
          ))}
          {!thread && (
            <div className="text-center mt-20">
              <p className="text-muted-foreground mb-4">Start a new conversation</p>
              <div className="flex flex-wrap justify-center gap-2">
                {["What entities are in the knowledge graph?", "Summarize the main themes", "Find relationships between concepts"].map((q) => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); }}
                    className="text-xs px-3 py-1.5 bg-eds-blue-light border border-eds-blue rounded-sm text-eds-blue hover:bg-eds-blue hover:text-white transition-colors dark:bg-[#0d2137] dark:border-eds-blue"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t pt-3">
          <div className="flex gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question… (Ctrl+Enter to send)"
              className="flex-1 min-h-[44px] max-h-32 resize-none rounded-sm"
              rows={1}
              disabled={streaming}
            />
            <Button onClick={sendMessage} disabled={!input.trim() || streaming} size="icon" className="shrink-0">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Feedback dialog */}
      {feedbackMsgId && (
        <Dialog.Root open onOpenChange={() => setFeedbackMsgId(null)}>
          <Dialog.Portal>
            <Dialog.Overlay className="fixed inset-0 bg-black/50 z-50" />
            <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-sm border bg-card shadow-lg">
              <Dialog.Title className="sr-only">Feedback</Dialog.Title>
              <FeedbackDialog queryId={feedbackMsgId} onClose={() => setFeedbackMsgId(null)} />
            </Dialog.Content>
          </Dialog.Portal>
        </Dialog.Root>
      )}

      {/* Compare panel */}
      {compareData && <ComparePanel data={compareData} onClose={() => setCompareData(null)} />}
    </div>
  );
}

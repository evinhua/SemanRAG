import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  references?: Record<string, unknown>[];
  groundedCheck?: { claim: string; status: "supported" | "partial" | "unsupported"; evidence?: string }[];
  latencyMs?: number;
  tokensUsed?: Record<string, number>;
  mode?: string;
  timestamp: number;
}

export interface ChatThread {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
}

interface ChatState {
  threads: ChatThread[];
  activeThreadId: string | null;
  mode: string;
  topK: number;
  chunkTopK: number;
  enableRerank: boolean;
  snapshotAt: string | null;
  userGroups: string[];
  setMode: (m: string) => void;
  setTopK: (n: number) => void;
  setChunkTopK: (n: number) => void;
  setEnableRerank: (v: boolean) => void;
  setSnapshotAt: (v: string | null) => void;
  setUserGroups: (g: string[]) => void;
  createThread: () => string;
  setActiveThread: (id: string) => void;
  addMessage: (threadId: string, msg: ChatMessage) => void;
  updateMessage: (threadId: string, msgId: string, patch: Partial<ChatMessage>) => void;
  deleteThread: (id: string) => void;
}

export const useChatStore = create<ChatState>()(
  immer((set) => ({
    threads: [],
    activeThreadId: null,
    mode: "local",
    topK: 20,
    chunkTopK: 5,
    enableRerank: true,
    snapshotAt: null,
    userGroups: [],
    setMode: (m) => set((s) => { s.mode = m; }),
    setTopK: (n) => set((s) => { s.topK = n; }),
    setChunkTopK: (n) => set((s) => { s.chunkTopK = n; }),
    setEnableRerank: (v) => set((s) => { s.enableRerank = v; }),
    setSnapshotAt: (v) => set((s) => { s.snapshotAt = v; }),
    setUserGroups: (g) => set((s) => { s.userGroups = g; }),
    createThread: () => {
      const id = crypto.randomUUID();
      set((s) => {
        s.threads.unshift({ id, title: "New Chat", messages: [], createdAt: Date.now() });
        s.activeThreadId = id;
      });
      return id;
    },
    setActiveThread: (id) => set((s) => { s.activeThreadId = id; }),
    addMessage: (threadId, msg) =>
      set((s) => {
        const t = s.threads.find((th) => th.id === threadId);
        if (t) {
          t.messages.push(msg);
          if (t.messages.length === 1 && msg.role === "user") {
            t.title = msg.content.slice(0, 50);
          }
        }
      }),
    updateMessage: (threadId, msgId, patch) =>
      set((s) => {
        const t = s.threads.find((th) => th.id === threadId);
        if (!t) return;
        const m = t.messages.find((msg) => msg.id === msgId);
        if (m) Object.assign(m, patch);
      }),
    deleteThread: (id) =>
      set((s) => {
        s.threads = s.threads.filter((t) => t.id !== id);
        if (s.activeThreadId === id) s.activeThreadId = s.threads[0]?.id ?? null;
      }),
  }))
);

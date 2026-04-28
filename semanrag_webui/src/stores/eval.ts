import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

export interface EvalRun {
  id: string;
  status: "pending" | "running" | "done" | "failed";
  startedAt: string;
  completedAt?: string;
  metrics: Record<string, number>;
}

interface EvalState {
  evalRuns: EvalRun[];
  latestReport: Record<string, unknown> | null;
  baseline: EvalRun | null;
  setRuns: (runs: EvalRun[]) => void;
  addRun: (run: EvalRun) => void;
  setLatestReport: (r: Record<string, unknown> | null) => void;
  setBaseline: (run: EvalRun | null) => void;
}

export const useEvalStore = create<EvalState>()(
  immer((set) => ({
    evalRuns: [],
    latestReport: null,
    baseline: null,
    setRuns: (runs) => set((s) => { s.evalRuns = runs; }),
    addRun: (run) => set((s) => { s.evalRuns.push(run); }),
    setLatestReport: (r) => set((s) => { s.latestReport = r; }),
    setBaseline: (run) => set((s) => { s.baseline = run; }),
  })),
);

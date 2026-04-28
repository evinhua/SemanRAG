import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

interface GraphState {
  selectedNode: string | null;
  selectedEdge: string | null;
  communityLevel: number;
  snapshotAt: string | null;
  layout: "force" | "circular" | "hierarchical";
  filters: { types: string[]; minWeight: number };
  selectNode: (id: string | null) => void;
  selectEdge: (id: string | null) => void;
  setCommunityLevel: (level: number) => void;
  setSnapshotAt: (ts: string | null) => void;
  setLayout: (layout: GraphState["layout"]) => void;
  setFilters: (filters: Partial<GraphState["filters"]>) => void;
}

export const useGraphStore = create<GraphState>()(
  immer((set) => ({
    selectedNode: null,
    selectedEdge: null,
    communityLevel: 0,
    snapshotAt: null,
    layout: "force",
    filters: { types: [], minWeight: 0 },
    selectNode: (id) => set((s) => { s.selectedNode = id; }),
    selectEdge: (id) => set((s) => { s.selectedEdge = id; }),
    setCommunityLevel: (level) => set((s) => { s.communityLevel = level; }),
    setSnapshotAt: (ts) => set((s) => { s.snapshotAt = ts; }),
    setLayout: (layout) => set((s) => { s.layout = layout; }),
    setFilters: (f) => set((s) => { Object.assign(s.filters, f); }),
  })),
);

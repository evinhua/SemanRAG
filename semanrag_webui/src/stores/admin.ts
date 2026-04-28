import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

interface AdminState {
  users: unknown[];
  groups: unknown[];
  budgetConfig: Record<string, unknown>;
  auditLog: unknown[];
  setUsers: (u: unknown[]) => void;
  setGroups: (g: unknown[]) => void;
  setBudgetConfig: (c: Record<string, unknown>) => void;
  setAuditLog: (log: unknown[]) => void;
}

export const useAdminStore = create<AdminState>()(
  immer((set) => ({
    users: [],
    groups: [],
    budgetConfig: {},
    auditLog: [],
    setUsers: (u) => set((s) => { s.users = u; }),
    setGroups: (g) => set((s) => { s.groups = g; }),
    setBudgetConfig: (c) => set((s) => { s.budgetConfig = c; }),
    setAuditLog: (log) => set((s) => { s.auditLog = log; }),
  })),
);

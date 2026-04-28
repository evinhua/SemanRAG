import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

interface User {
  id: string;
  email: string;
  name: string;
  role: string;
}

interface AppState {
  user: User | null;
  isAuthenticated: boolean;
  sidebarOpen: boolean;
  theme: "light" | "dark" | "system";
  setUser: (user: User | null) => void;
  toggleSidebar: () => void;
  setTheme: (theme: "light" | "dark" | "system") => void;
}

export const useAppStore = create<AppState>()(
  immer((set) => ({
    user: null,
    isAuthenticated: !!localStorage.getItem("access_token"),
    sidebarOpen: true,
    theme: (localStorage.getItem("theme") as AppState["theme"]) ?? "system",
    setUser: (user) =>
      set((s) => {
        s.user = user;
        s.isAuthenticated = !!user;
      }),
    toggleSidebar: () =>
      set((s) => {
        s.sidebarOpen = !s.sidebarOpen;
      }),
    setTheme: (theme) =>
      set((s) => {
        s.theme = theme;
        localStorage.setItem("theme", theme);
      }),
  })),
);

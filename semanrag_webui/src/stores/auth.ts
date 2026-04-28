import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AuthState {
  token: string | null;
  user: { sub: string; scope?: string } | null;
  setAuth: (token: string, user: { sub: string; scope?: string }) => void;
  logout: () => void;
  isAuthenticated: () => boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      setAuth: (token, user) => {
        localStorage.setItem("semanrag_token", token);
        set({ token, user });
      },
      logout: () => {
        localStorage.removeItem("semanrag_token");
        localStorage.removeItem("semanrag_api_key");
        set({ token: null, user: null });
      },
      isAuthenticated: () => !!get().token || !!localStorage.getItem("semanrag_api_key"),
    }),
    { name: "semanrag-auth" }
  )
);

import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

export type Theme = "light" | "dark" | "system";
export type Density = "comfortable" | "compact";
export type Lang = "en" | "zh" | "ja" | "es";

interface SettingsState {
  theme: Theme;
  lang: Lang;
  density: Density;
  colorBlind: boolean;
  setTheme: (t: Theme) => void;
  setLang: (l: Lang) => void;
  setDensity: (d: Density) => void;
  setColorBlind: (v: boolean) => void;
}

export const useSettingsStore = create<SettingsState>()(
  immer((set) => ({
    theme: "system" as Theme,
    lang: "en" as Lang,
    density: "comfortable" as Density,
    colorBlind: false,
    setTheme: (t) => set((s) => { s.theme = t; }),
    setLang: (l) => set((s) => { s.lang = l; }),
    setDensity: (d) => set((s) => { s.density = d; }),
    setColorBlind: (v) => set((s) => { s.colorBlind = v; }),
  }))
);

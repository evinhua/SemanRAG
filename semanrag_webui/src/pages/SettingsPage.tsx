import * as Select from "@radix-ui/react-select";
import * as Switch from "@radix-ui/react-switch";
import { Sun, Moon, Monitor, ChevronDown, Check, Keyboard } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useSettingsStore, type Theme, type Lang, type Density } from "@/stores/settings";

const THEMES: { value: Theme; label: string; icon: typeof Sun }[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
];

const LANGUAGES: { value: Lang; label: string }[] = [
  { value: "en", label: "English" },
  { value: "zh", label: "中文" },
  { value: "ja", label: "日本語" },
  { value: "es", label: "Español" },
];

const DENSITIES: { value: Density; label: string }[] = [
  { value: "comfortable", label: "Comfortable" },
  { value: "compact", label: "Compact" },
];

const SHORTCUTS = [
  { keys: "⌘K", action: "Open command palette" },
  { keys: "Ctrl+Enter", action: "Send message" },
  { keys: "G C", action: "Go to Chat" },
  { keys: "G G", action: "Go to Graph" },
  { keys: "G D", action: "Go to Documents" },
  { keys: "G A", action: "Go to Admin" },
  { keys: "G S", action: "Go to Settings" },
];

function SelectField<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="flex items-center justify-between">
      <Label>{label}</Label>
      <Select.Root value={value} onValueChange={(v) => onChange(v as T)}>
        <Select.Trigger className="inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm bg-background w-40 justify-between">
          <Select.Value />
          <ChevronDown className="h-3 w-3" />
        </Select.Trigger>
        <Select.Portal>
          <Select.Content className="z-50 rounded-md border bg-card shadow-md">
            <Select.Viewport className="p-1">
              {options.map((o) => (
                <Select.Item key={o.value} value={o.value} className="flex items-center gap-2 rounded px-3 py-1.5 text-sm cursor-pointer outline-none data-[highlighted]:bg-accent">
                  <Select.ItemText>{o.label}</Select.ItemText>
                  <Select.ItemIndicator><Check className="h-3 w-3" /></Select.ItemIndicator>
                </Select.Item>
              ))}
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
    </div>
  );
}

export default function SettingsPage() {
  const { theme, lang, density, colorBlind, setTheme, setLang, setDensity, setColorBlind } = useSettingsStore();

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Settings</h1>

      <Card>
        <CardHeader><CardTitle className="text-base">Appearance</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          {/* Theme */}
          <div className="flex items-center justify-between">
            <Label>Theme</Label>
            <div className="flex gap-1">
              {THEMES.map(({ value, label, icon: Icon }) => (
                <button
                  key={value}
                  onClick={() => setTheme(value)}
                  className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${theme === value ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-accent"}`}
                >
                  <Icon className="h-3 w-3" />
                  {label}
                </button>
              ))}
            </div>
          </div>

          <SelectField label="Language" value={lang} options={LANGUAGES} onChange={setLang} />
          <SelectField label="Density" value={density} options={DENSITIES} onChange={setDensity} />

          {/* Color-blind palette */}
          <div className="flex items-center justify-between">
            <Label>Color-blind friendly palette</Label>
            <Switch.Root
              checked={colorBlind}
              onCheckedChange={setColorBlind}
              className="w-9 h-5 rounded-full bg-muted data-[state=checked]:bg-primary transition-colors"
            >
              <Switch.Thumb className="block h-4 w-4 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-4 translate-x-0.5" />
            </Switch.Root>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Keyboard className="h-4 w-4" /> Keyboard Shortcuts
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {SHORTCUTS.map(({ keys, action }) => (
              <div key={keys} className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">{action}</span>
                <kbd className="bg-muted px-2 py-0.5 rounded text-xs font-mono">{keys}</kbd>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

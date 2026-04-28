import { useState, useEffect, type ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  MessageSquare, Network, FileText, Shield, Settings, LogOut,
  Search, Sun, Moon, Menu, X, Command,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useSettingsStore } from "@/stores/settings";
import { useAuthStore } from "@/stores/auth";
import { CommandPalette } from "@/components/CommandPalette";
import * as Toast from "@radix-ui/react-toast";

const NAV = [
  { to: "/chat", icon: MessageSquare, label: "Chat" },
  { to: "/explore", icon: Network, label: "Graph" },
  { to: "/files", icon: FileText, label: "Documents" },
  { to: "/admin-panel", icon: Shield, label: "Admin" },
  { to: "/settings", icon: Settings, label: "Settings" },
] as const;

function ThemeToggle() {
  const { theme, setTheme } = useSettingsStore();
  const next = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
  return (
    <Button variant="ghost" size="icon" onClick={() => setTheme(next)} aria-label="Toggle theme">
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [cmdOpen, setCmdOpen] = useState(false);
  const { theme } = useSettingsStore();
  const { logout, user } = useAuthStore();
  const navigate = useNavigate();

  // Apply theme class
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove("light", "dark");
    if (theme === "system") {
      root.classList.add(window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    } else {
      root.classList.add(theme);
    }
  }, [theme]);

  // ⌘K shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setCmdOpen(true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <Toast.Provider>
      <div className="flex h-screen bg-background text-foreground">
        {/* Sidebar */}
        <aside className={`${sidebarOpen ? "w-56" : "w-14"} flex flex-col border-r bg-card transition-all duration-200`}>
          <div className="flex h-14 items-center justify-between px-3 border-b">
            {sidebarOpen && <span className="font-bold text-lg text-primary">SemanRAG</span>}
            <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(!sidebarOpen)}>
              {sidebarOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </Button>
          </div>
          <nav className="flex-1 py-2 space-y-1 px-2">
            {NAV.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                    isActive ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground hover:bg-accent"
                  }`
                }
              >
                <Icon className="h-4 w-4 shrink-0" />
                {sidebarOpen && <span>{label}</span>}
              </NavLink>
            ))}
          </nav>
          {sidebarOpen && user && (
            <div className="border-t p-3">
              <div className="text-xs text-muted-foreground truncate mb-1">{user.sub}</div>
              <Button variant="ghost" size="sm" className="w-full justify-start gap-2" onClick={() => { logout(); navigate("/login"); }}>
                <LogOut className="h-3 w-3" /> Sign out
              </Button>
            </div>
          )}
        </aside>

        {/* Main area */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Top bar */}
          <header className="flex h-14 items-center gap-3 border-b px-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search… (⌘K)"
                className="pl-9"
                readOnly
                onClick={() => setCmdOpen(true)}
              />
            </div>
            <div className="ml-auto flex items-center gap-1">
              <Button variant="ghost" size="icon" onClick={() => setCmdOpen(true)} aria-label="Command palette">
                <Command className="h-4 w-4" />
              </Button>
              <ThemeToggle />
            </div>
          </header>

          {/* Content */}
          <main className="flex-1 overflow-auto p-4">{children}</main>
        </div>
      </div>

      {/* Command Palette */}
      <CommandPalette open={cmdOpen} onOpenChange={setCmdOpen} />

      {/* Toast viewport */}
      <Toast.Viewport className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-96" />
    </Toast.Provider>
  );
}

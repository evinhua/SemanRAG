import { useState, useEffect, type ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  MessageSquare, Network, FileText, Shield, Settings, LogOut,
  Sun, Moon, Menu, ChevronDown, ChevronRight,
} from "lucide-react";
import { useSettingsStore } from "@/stores/settings";
import { useAuthStore } from "@/stores/auth";
import { CommandPalette } from "@/components/CommandPalette";
import * as Toast from "@radix-ui/react-toast";

const NAV_GROUPS = [
  {
    label: "Intelligence",
    items: [
      { to: "/chat", icon: MessageSquare, label: "Chat" },
      { to: "/explore", icon: Network, label: "Knowledge Graph" },
    ],
  },
  {
    label: "Data Management",
    items: [
      { to: "/files", icon: FileText, label: "Documents" },
    ],
  },
  {
    label: "Operations",
    items: [
      { to: "/admin-panel", icon: Shield, label: "Admin Panel" },
      { to: "/settings", icon: Settings, label: "Settings" },
    ],
  },
] as const;

function ThemeToggle() {
  const { theme, setTheme } = useSettingsStore();
  const next = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
  return (
    <button
      onClick={() => setTheme(next)}
      className="flex items-center gap-1 px-2 py-1 text-xs text-eds-gray-500 hover:text-foreground transition-colors"
      aria-label="Toggle theme"
    >
      {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
    </button>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const [navOpen, setNavOpen] = useState(true);
  const [cmdOpen, setCmdOpen] = useState(false);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({ Intelligence: true, "Data Management": true, Operations: true });
  const { theme } = useSettingsStore();
  const { logout, user } = useAuthStore();
  const navigate = useNavigate();

  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove("light", "dark");
    if (theme === "system") {
      root.classList.add(window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    } else {
      root.classList.add(theme);
    }
  }, [theme]);

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

  const toggleGroup = (label: string) => {
    setOpenGroups((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  return (
    <Toast.Provider>
      <div className="flex flex-col h-screen bg-background text-foreground">
        {/* System Bar (EDS sysbar) */}
        <header className="flex items-center justify-between h-8 px-4 bg-eds-gray-800 text-white text-xs shrink-0">
          <div className="flex items-center gap-2">
            <span className="font-bold tracking-wide">SemanRAG</span>
            <span className="text-eds-gray-400">Semantic Intelligence Platform</span>
          </div>
          <div className="flex items-center gap-3">
            <ThemeToggle />
            {user && (
              <span className="text-eds-gray-400">{user.sub}</span>
            )}
          </div>
        </header>

        {/* App Bar (EDS appbar) */}
        <div className="flex items-center h-11 px-4 border-b bg-card shrink-0">
          <button
            onClick={() => setNavOpen(!navOpen)}
            className="mr-3 p-1 rounded hover:bg-accent transition-colors"
            aria-label="Toggle navigation"
          >
            <Menu className="h-4 w-4" />
          </button>
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">SemanRAG</span>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setCmdOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 text-xs text-muted-foreground border rounded hover:bg-accent transition-colors"
            >
              Search… <kbd className="text-[10px] bg-muted px-1 rounded">⌘K</kbd>
            </button>
            <span className="status-badge success">Healthy</span>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Navigation Tree (EDS appnav) */}
          {navOpen && (
            <aside className="w-56 flex flex-col border-r bg-card overflow-y-auto shrink-0">
              <nav className="flex-1 py-2">
                {NAV_GROUPS.map((group) => (
                  <div key={group.label} className="mb-1">
                    <button
                      onClick={() => toggleGroup(group.label)}
                      className="flex items-center justify-between w-full px-4 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {group.label}
                      {openGroups[group.label] ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronRight className="h-3 w-3" />
                      )}
                    </button>
                    {openGroups[group.label] && (
                      <ul className="mt-0.5">
                        {group.items.map(({ to, icon: Icon, label }) => (
                          <li key={to}>
                            <NavLink
                              to={to}
                              className={({ isActive }) =>
                                `flex items-center gap-2.5 px-4 py-2 text-sm transition-colors ${
                                  isActive
                                    ? "text-eds-blue font-medium bg-eds-blue-light dark:bg-[#0d2137]"
                                    : "text-foreground hover:bg-accent"
                                }`
                              }
                            >
                              <Icon className="h-4 w-4 shrink-0" />
                              <span>{label}</span>
                            </NavLink>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </nav>

              {/* User section at bottom */}
              {user && (
                <div className="border-t p-3">
                  <div className="text-xs text-muted-foreground truncate mb-2">{user.sub}</div>
                  <button
                    onClick={() => { logout(); navigate("/login"); }}
                    className="flex items-center gap-2 w-full px-2 py-1.5 text-xs text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors"
                  >
                    <LogOut className="h-3 w-3" /> Sign out
                  </button>
                </div>
              )}
            </aside>
          )}

          {/* Content Area (EDS appcontent) */}
          <main className="flex-1 overflow-auto p-6">{children}</main>
        </div>
      </div>

      <CommandPalette open={cmdOpen} onOpenChange={setCmdOpen} />
      <Toast.Viewport className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-96" />
    </Toast.Provider>
  );
}

import { useNavigate } from "react-router-dom";
import { Command } from "cmdk";
import * as Dialog from "@radix-ui/react-dialog";
import { MessageSquare, Network, FileText, Shield, Settings, Search, Play, File } from "lucide-react";

const PAGES = [
  { name: "Chat", path: "/chat", icon: MessageSquare, shortcut: "G C" },
  { name: "Graph Explorer", path: "/graph", icon: Network, shortcut: "G G" },
  { name: "Documents", path: "/documents", icon: FileText, shortcut: "G D" },
  { name: "Admin", path: "/admin", icon: Shield, shortcut: "G A" },
  { name: "Settings", path: "/settings", icon: Settings, shortcut: "G S" },
] as const;

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CommandPalette({ open, onOpenChange }: Props) {
  const navigate = useNavigate();

  const go = (path: string) => {
    navigate(path);
    onOpenChange(false);
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 z-50" />
        <Dialog.Content className="fixed left-1/2 top-[20%] z-50 w-full max-w-lg -translate-x-1/2 rounded-lg border bg-card shadow-lg">
          <Command className="flex flex-col" label="Command palette">
            <div className="flex items-center border-b px-3">
              <Search className="mr-2 h-4 w-4 shrink-0 text-muted-foreground" />
              <Command.Input
                placeholder="Type a command or search…"
                className="flex h-11 w-full bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground"
              />
            </div>
            <Command.List className="max-h-72 overflow-y-auto p-2">
              <Command.Empty className="py-6 text-center text-sm text-muted-foreground">No results.</Command.Empty>

              <Command.Group heading="Pages" className="text-xs text-muted-foreground px-2 py-1.5">
                {PAGES.map(({ name, path, icon: Icon, shortcut }) => (
                  <Command.Item
                    key={path}
                    value={name}
                    onSelect={() => go(path)}
                    className="flex items-center gap-3 rounded-md px-3 py-2 text-sm cursor-pointer aria-selected:bg-accent"
                  >
                    <Icon className="h-4 w-4 text-muted-foreground" />
                    <span className="flex-1">{name}</span>
                    <kbd className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">{shortcut}</kbd>
                  </Command.Item>
                ))}
              </Command.Group>

              <Command.Group heading="Actions" className="text-xs text-muted-foreground px-2 py-1.5">
                <Command.Item
                  value="Search entities"
                  onSelect={() => go("/graph")}
                  className="flex items-center gap-3 rounded-md px-3 py-2 text-sm cursor-pointer aria-selected:bg-accent"
                >
                  <Search className="h-4 w-4 text-muted-foreground" />
                  <span className="flex-1">Search entities</span>
                </Command.Item>
                <Command.Item
                  value="Run query"
                  onSelect={() => go("/chat")}
                  className="flex items-center gap-3 rounded-md px-3 py-2 text-sm cursor-pointer aria-selected:bg-accent"
                >
                  <Play className="h-4 w-4 text-muted-foreground" />
                  <span className="flex-1">Run query</span>
                </Command.Item>
                <Command.Item
                  value="Open document"
                  onSelect={() => go("/documents")}
                  className="flex items-center gap-3 rounded-md px-3 py-2 text-sm cursor-pointer aria-selected:bg-accent"
                >
                  <File className="h-4 w-4 text-muted-foreground" />
                  <span className="flex-1">Open document</span>
                </Command.Item>
              </Command.Group>
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

import { useState, useEffect, useRef, useCallback } from "react";
import Graph from "graphology";
import Sigma from "sigma";
import forceAtlas2 from "graphology-layout-forceatlas2";
import louvain from "graphology-communities-louvain";
import * as Dialog from "@radix-ui/react-dialog";
import * as Select from "@radix-ui/react-select";
import {
  Search, Download, Merge, Edit3, MapPin, ChevronDown, Check, X,
  ZoomIn, ZoomOut, Maximize2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { graphApi, type GraphData } from "@/lib/semanrag";

// ── Community colors ─────────────────────────────────────────────────

const COMMUNITY_COLORS = [
  "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#6366f1",
  "#84cc16", "#e11d48", "#0ea5e9", "#d946ef", "#a3e635",
];

function communityColor(id: number) {
  return COMMUNITY_COLORS[id % COMMUNITY_COLORS.length] ?? "#6b7280";
}

// ── Types ────────────────────────────────────────────────────────────

type LayoutType = "force" | "hierarchical" | "circular";

interface NodeDetail {
  id: string;
  type?: string;
  description?: string;
  community?: number;
  [key: string]: unknown;
}

interface EdgeDetail {
  source: string;
  target: string;
  keywords?: string;
  description?: string;
  [key: string]: unknown;
}

// ── Property Edit Dialog ─────────────────────────────────────────────

function PropertyEditDialog({ node, onClose, onSave }: { node: NodeDetail; onClose: () => void; onSave: (data: Record<string, unknown>) => void }) {
  const [type, setType] = useState(node.type ?? "");
  const [description, setDescription] = useState(node.description ?? "");

  return (
    <Dialog.Root open onOpenChange={onClose}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 z-50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border bg-card shadow-lg p-6">
          <Dialog.Title className="text-lg font-semibold mb-4">Edit Entity: {node.id}</Dialog.Title>
          <div className="space-y-3">
            <div><Label>Type</Label><Input value={type} onChange={(e) => setType(e.target.value)} /></div>
            <div><Label>Description</Label><Input value={description} onChange={(e) => setDescription(e.target.value)} /></div>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button onClick={() => onSave({ type, description })}>Save</Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

// ── Entity Merge Dialog ──────────────────────────────────────────────

function MergeDialog({ nodes, onClose, onMerge }: { nodes: string[]; onClose: () => void; onMerge: (target: string, strategy: string) => void }) {
  const [target, setTarget] = useState(nodes[0] ?? "");
  const [strategy, setStrategy] = useState("concatenate");

  return (
    <Dialog.Root open onOpenChange={onClose}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 z-50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border bg-card shadow-lg p-6">
          <Dialog.Title className="text-lg font-semibold mb-4">Merge Entities</Dialog.Title>
          <p className="text-sm text-muted-foreground mb-3">Merging: {nodes.join(", ")}</p>
          <div className="space-y-3">
            <div><Label>Target entity name</Label><Input value={target} onChange={(e) => setTarget(e.target.value)} /></div>
            <div>
              <Label>Strategy</Label>
              <select value={strategy} onChange={(e) => setStrategy(e.target.value)} className="w-full rounded-md border px-3 py-2 text-sm bg-background">
                <option value="concatenate">Concatenate</option>
                <option value="keep_first">Keep first</option>
                <option value="join_unique">Join unique</option>
                <option value="confidence_weighted">Confidence weighted</option>
              </select>
            </div>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button onClick={() => onMerge(target, strategy)}>Merge</Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

// ── Main Page ────────────────────────────────────────────────────────

export default function GraphPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const graphRef = useRef<Graph>(new Graph());

  const [searchQuery, setSearchQuery] = useState("");
  const [snapshotAt, setSnapshotAt] = useState("");
  const [hops, setHops] = useState(1);
  const [layout, setLayout] = useState<LayoutType>("force");
  const [selectedNode, setSelectedNode] = useState<NodeDetail | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<EdgeDetail | null>(null);
  const [editNode, setEditNode] = useState<NodeDetail | null>(null);
  const [mergeNodes, setMergeNodes] = useState<string[] | null>(null);
  const [pathSrc, setPathSrc] = useState("");
  const [pathTgt, setPathTgt] = useState("");
  const [pathResult, setPathResult] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);

  const applyLayout = useCallback((type: LayoutType, g: Graph) => {
    if (g.order === 0) return;
    if (type === "force") {
      forceAtlas2.assign(g, { iterations: 100, settings: { gravity: 1, scalingRatio: 2 } });
    } else if (type === "circular") {
      const nodes = g.nodes();
      nodes.forEach((n, i) => {
        const angle = (2 * Math.PI * i) / nodes.length;
        g.setNodeAttribute(n, "x", Math.cos(angle) * 50);
        g.setNodeAttribute(n, "y", Math.sin(angle) * 50);
      });
    } else {
      const sorted = g.nodes().sort((a, b) => g.degree(b) - g.degree(a));
      sorted.forEach((n, i) => {
        g.setNodeAttribute(n, "x", (i % 10) * 10);
        g.setNodeAttribute(n, "y", Math.floor(i / 10) * 10);
      });
    }
  }, []);

  const buildGraph = useCallback((data: GraphData) => {
    const g = graphRef.current;
    g.clear();

    for (const node of data.nodes) {
      const id = String(node.id ?? node.name ?? node.label ?? "");
      if (!id || g.hasNode(id)) continue;
      g.addNode(id, {
        label: id,
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.min(3 + (Number(node.degree) || 0) * 0.5, 20),
        color: "#6b7280",
        ...node,
      });
    }

    for (const edge of data.edges) {
      const src = String(edge.source ?? edge.src ?? "");
      const tgt = String(edge.target ?? edge.tgt ?? "");
      if (!src || !tgt || !g.hasNode(src) || !g.hasNode(tgt)) continue;
      try { g.addEdge(src, tgt, { label: String(edge.keywords ?? ""), ...edge }); } catch { /* skip duplicates */ }
    }

    // Community detection & coloring
    try {
      const communities = louvain(g);
      g.forEachNode((node) => {
        const cid = communities[node];
        if (cid != null) {
          g.setNodeAttribute(node, "community", cid);
          g.setNodeAttribute(node, "color", communityColor(cid));
        }
      });
    } catch { /* may fail on disconnected graphs */ }

    applyLayout(layout, g);
    sigmaRef.current?.refresh();
  }, [layout, applyLayout]);

  const loadGraph = useCallback(async () => {
    setLoading(true);
    try {
      const data = await graphApi.getGraph({ snapshot_at: snapshotAt || undefined });
      buildGraph(data);
    } catch (err) {
      console.error("Failed to load graph:", err);
    } finally {
      setLoading(false);
    }
  }, [snapshotAt, buildGraph]);

  // Initialize sigma
  useEffect(() => {
    if (!containerRef.current) return;
    const sigma = new Sigma(graphRef.current, containerRef.current, {
      renderEdgeLabels: true,
      defaultEdgeColor: "#d1d5db",
      defaultNodeColor: "#6b7280",
    });

    sigma.on("clickNode", ({ node }) => {
      const attrs = graphRef.current.getNodeAttributes(node);
      setSelectedNode({ id: node, ...attrs } as NodeDetail);
      setSelectedEdge(null);
    });

    sigma.on("clickEdge", ({ edge }) => {
      const attrs = graphRef.current.getEdgeAttributes(edge);
      const [src, tgt] = graphRef.current.extremities(edge);
      setSelectedEdge({ source: src, target: tgt, ...attrs } as EdgeDetail);
      setSelectedNode(null);
    });

    sigmaRef.current = sigma;
    loadGraph();

    return () => { sigma.kill(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { loadGraph(); }, [snapshotAt, loadGraph]);

  useEffect(() => { applyLayout(layout, graphRef.current); sigmaRef.current?.refresh(); }, [layout, applyLayout]);

  // Search highlight
  useEffect(() => {
    const g = graphRef.current;
    if (!searchQuery) {
      g.forEachNode((n) => g.setNodeAttribute(n, "highlighted", false));
    } else {
      const q = searchQuery.toLowerCase();
      g.forEachNode((n) => g.setNodeAttribute(n, "highlighted", n.toLowerCase().includes(q)));
    }
    sigmaRef.current?.refresh();
  }, [searchQuery]);

  const isolateNeighborhood = async (name: string) => {
    try {
      const data = await graphApi.getNeighborhood(name, hops);
      buildGraph(data);
    } catch (err) {
      console.error("Neighborhood fetch failed:", err);
    }
  };

  const findPath = async () => {
    if (!pathSrc || !pathTgt) return;
    try {
      const result = await graphApi.getPath(pathSrc, pathTgt);
      setPathResult(result.path);
      const g = graphRef.current;
      g.forEachNode((n) => g.setNodeAttribute(n, "highlighted", result.path.includes(n)));
      sigmaRef.current?.refresh();
    } catch {
      setPathResult([]);
    }
  };

  const exportGraph = (format: "png" | "json") => {
    if (format === "json") {
      const data = graphRef.current.export();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = "graph.json"; a.click();
      URL.revokeObjectURL(url);
    } else {
      const canvas = containerRef.current?.querySelector("canvas");
      if (!canvas) return;
      const url = (canvas as HTMLCanvasElement).toDataURL("image/png");
      const a = document.createElement("a");
      a.href = url; a.download = "graph.png"; a.click();
    }
  };

  const handleSaveEntity = async (data: Record<string, unknown>) => {
    if (!editNode) return;
    await graphApi.updateEntity(editNode.id, data);
    setEditNode(null);
    loadGraph();
  };

  const handleMerge = async (target: string, strategy: string) => {
    if (!mergeNodes) return;
    await graphApi.mergeEntities({ source_entities: mergeNodes, target_entity: target, merge_strategy: strategy });
    setMergeNodes(null);
    loadGraph();
  };

  return (
    <div className="flex h-full gap-4">
      {/* Controls sidebar */}
      <div className="w-64 shrink-0 space-y-4 overflow-auto">
        <div>
          <Label className="mb-1 block">Search nodes</Label>
          <div className="relative">
            <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} placeholder="Entity name…" className="pl-8" />
          </div>
        </div>

        <div>
          <Label className="mb-1 block">Snapshot at</Label>
          <Input type="datetime-local" value={snapshotAt} onChange={(e) => setSnapshotAt(e.target.value)} />
        </div>

        <div>
          <Label className="mb-1 block">Layout</Label>
          <Select.Root value={layout} onValueChange={(v) => setLayout(v as LayoutType)}>
            <Select.Trigger className="w-full inline-flex items-center justify-between rounded-md border px-3 py-2 text-sm bg-background">
              <Select.Value />
              <ChevronDown className="h-3 w-3" />
            </Select.Trigger>
            <Select.Portal>
              <Select.Content className="z-50 rounded-md border bg-card shadow-md">
                <Select.Viewport className="p-1">
                  {(["force", "hierarchical", "circular"] as const).map((l) => (
                    <Select.Item key={l} value={l} className="flex items-center gap-2 rounded px-3 py-1.5 text-sm cursor-pointer outline-none data-[highlighted]:bg-accent capitalize">
                      <Select.ItemText>{l}</Select.ItemText>
                      <Select.ItemIndicator><Check className="h-3 w-3" /></Select.ItemIndicator>
                    </Select.Item>
                  ))}
                </Select.Viewport>
              </Select.Content>
            </Select.Portal>
          </Select.Root>
        </div>

        <div>
          <Label className="mb-1 block">Neighborhood (n-hop)</Label>
          <div className="flex gap-2">
            <Input type="number" value={hops} onChange={(e) => setHops(Number(e.target.value))} min={1} max={5} className="w-16" />
            <Button size="sm" variant="outline" onClick={() => selectedNode && isolateNeighborhood(selectedNode.id)} disabled={!selectedNode}>
              <MapPin className="h-3 w-3 mr-1" /> Isolate
            </Button>
          </div>
        </div>

        <div>
          <Label className="mb-1 block">Path finding</Label>
          <Input value={pathSrc} onChange={(e) => setPathSrc(e.target.value)} placeholder="Source" className="mb-1" />
          <Input value={pathTgt} onChange={(e) => setPathTgt(e.target.value)} placeholder="Target" className="mb-1" />
          <Button size="sm" variant="outline" className="w-full" onClick={findPath}>Find path</Button>
          {pathResult && (
            <p className="text-xs text-muted-foreground mt-1">
              {pathResult.length > 0 ? pathResult.join(" → ") : "No path found"}
            </p>
          )}
        </div>

        <div className="space-y-2">
          <Button size="sm" variant="outline" className="w-full" onClick={() => loadGraph()}>Reload graph</Button>
          <Button size="sm" variant="outline" className="w-full" onClick={() => selectedNode && setEditNode(selectedNode)} disabled={!selectedNode}>
            <Edit3 className="h-3 w-3 mr-1" /> Edit entity
          </Button>
          <Button size="sm" variant="outline" className="w-full" onClick={() => selectedNode && setMergeNodes([selectedNode.id])} disabled={!selectedNode}>
            <Merge className="h-3 w-3 mr-1" /> Merge entities
          </Button>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" className="flex-1" onClick={() => exportGraph("png")}><Download className="h-3 w-3 mr-1" /> PNG</Button>
            <Button size="sm" variant="outline" className="flex-1" onClick={() => exportGraph("json")}><Download className="h-3 w-3 mr-1" /> JSON</Button>
          </div>
        </div>

        <div className="flex gap-2">
          <Button size="icon" variant="outline" onClick={() => sigmaRef.current?.getCamera().animatedZoom({ duration: 200 })}><ZoomIn className="h-4 w-4" /></Button>
          <Button size="icon" variant="outline" onClick={() => sigmaRef.current?.getCamera().animatedUnzoom({ duration: 200 })}><ZoomOut className="h-4 w-4" /></Button>
          <Button size="icon" variant="outline" onClick={() => sigmaRef.current?.getCamera().animatedReset({ duration: 200 })}><Maximize2 className="h-4 w-4" /></Button>
        </div>
      </div>

      {/* Graph canvas */}
      <div className="flex-1 relative border rounded-lg overflow-hidden">
        {loading && <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10"><span className="text-sm text-muted-foreground">Loading graph…</span></div>}
        <div ref={containerRef} className="w-full h-full" />
        <div className="absolute bottom-2 right-2 w-32 h-24 border rounded bg-card/80 flex items-center justify-center text-[10px] text-muted-foreground">Minimap</div>
      </div>

      {/* Detail panel */}
      {(selectedNode || selectedEdge) && (
        <div className="w-72 shrink-0 border-l pl-4 overflow-auto">
          {selectedNode && (
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">{selectedNode.id}</CardTitle>
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setSelectedNode(null)}><X className="h-3 w-3" /></Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {selectedNode.type && <div><span className="text-muted-foreground">Type:</span> <Badge variant="outline">{String(selectedNode.type)}</Badge></div>}
                {selectedNode.community != null && <div><span className="text-muted-foreground">Community:</span> <Badge style={{ backgroundColor: communityColor(selectedNode.community) }} className="text-white">{selectedNode.community}</Badge></div>}
                {selectedNode.description && <p className="text-muted-foreground">{String(selectedNode.description)}</p>}
                <div className="flex gap-2 pt-2">
                  <Button size="sm" variant="outline" onClick={() => setEditNode(selectedNode)}><Edit3 className="h-3 w-3 mr-1" /> Edit</Button>
                  <Button size="sm" variant="outline" onClick={() => isolateNeighborhood(selectedNode.id)}><MapPin className="h-3 w-3 mr-1" /> Neighbors</Button>
                </div>
              </CardContent>
            </Card>
          )}
          {selectedEdge && (
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">{selectedEdge.source} → {selectedEdge.target}</CardTitle>
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setSelectedEdge(null)}><X className="h-3 w-3" /></Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {selectedEdge.keywords && <div><span className="text-muted-foreground">Keywords:</span> {selectedEdge.keywords}</div>}
                {selectedEdge.description && <p className="text-muted-foreground">{selectedEdge.description}</p>}
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {editNode && <PropertyEditDialog node={editNode} onClose={() => setEditNode(null)} onSave={handleSaveEntity} />}
      {mergeNodes && <MergeDialog nodes={mergeNodes} onClose={() => setMergeNodes(null)} onMerge={handleMerge} />}
    </div>
  );
}

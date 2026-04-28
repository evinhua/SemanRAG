import { useState, useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { graphApi, type GraphData } from "@/lib/semanrag";

interface NodeDatum extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type?: string;
  description?: string;
  degree?: number;
}

interface LinkDatum extends d3.SimulationLinkDatum<NodeDatum> {
  keywords?: string;
  description?: string;
}

export default function GraphPage() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [selectedNode, setSelectedNode] = useState<NodeDatum | null>(null);
  const [nodeCount, setNodeCount] = useState(0);
  const [edgeCount, setEdgeCount] = useState(0);

  const loadGraph = useCallback(async () => {
    if (!svgRef.current) return;
    setLoading(true);
    try {
      const data: GraphData = await graphApi.getGraph({});
      renderGraph(data);
    } catch (err) {
      console.error("Failed to load graph:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const renderGraph = (data: GraphData) => {
    const svg = d3.select(svgRef.current!);
    svg.selectAll("*").remove();

    const width = svgRef.current!.clientWidth;
    const height = svgRef.current!.clientHeight;

    const nodes: NodeDatum[] = data.nodes.map((n: any) => ({
      id: String(n.id ?? n.name ?? ""),
      label: String(n.id ?? n.name ?? ""),
      type: n.type,
      description: n.description,
      degree: n.degree ?? 1,
    }));

    const nodeIds = new Set(nodes.map((n) => n.id));
    const links: LinkDatum[] = data.edges
      .map((e: any) => ({
        source: String(e.source ?? e.src ?? ""),
        target: String(e.target ?? e.tgt ?? ""),
        keywords: e.keywords,
        description: e.description,
      }))
      .filter((l) => nodeIds.has(l.source as string) && nodeIds.has(l.target as string));

    setNodeCount(nodes.length);
    setEdgeCount(links.length);

    // Color by type
    const types = [...new Set(nodes.map((n) => n.type || "Other"))];
    const color = d3.scaleOrdinal(d3.schemeTableau10).domain(types);

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink<NodeDatum, LinkDatum>(links).id((d) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(20));

    // Zoom
    const g = svg.append("g");
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => g.attr("transform", event.transform))
    );

    // Links
    const link = g.append("g")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.4)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 1);

    // Nodes
    const node = g.append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => Math.min(4 + (d.degree || 1) * 0.5, 16))
      .attr("fill", (d) => color(d.type || "Other"))
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .style("cursor", "pointer")
      .on("click", (_, d) => setSelectedNode(d))
      .call(
        d3.drag<SVGCircleElement, NodeDatum>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }) as any
      );

    // Labels (only for larger nodes)
    const labels = g.append("g")
      .selectAll("text")
      .data(nodes.filter((n) => (n.degree || 0) > 3))
      .join("text")
      .text((d) => d.label.length > 20 ? d.label.slice(0, 20) + "…" : d.label)
      .attr("font-size", 9)
      .attr("dx", 12)
      .attr("dy", 4)
      .attr("fill", "#374151");

    // Tooltip on hover
    node.append("title").text((d) => d.label);

    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);
      node.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);
      labels.attr("x", (d) => d.x!).attr("y", (d) => d.y!);
    });

    // Highlight search
    if (search) {
      const lower = search.toLowerCase();
      node.attr("opacity", (d) => d.label.toLowerCase().includes(lower) ? 1 : 0.15);
      link.attr("stroke-opacity", 0.1);
    }
  };

  useEffect(() => { loadGraph(); }, [loadGraph]);

  return (
    <div className="flex h-[calc(100vh-4rem)] gap-4">
      {/* Sidebar */}
      <div className="w-64 shrink-0 space-y-3 overflow-auto p-2">
        <h2 className="font-semibold text-lg">Graph Explorer</h2>
        <div className="text-xs text-muted-foreground">
          {nodeCount} nodes · {edgeCount} edges
        </div>
        <Input
          placeholder="Search nodes…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && loadGraph()}
        />
        <Button size="sm" variant="outline" className="w-full" onClick={loadGraph}>
          Reload
        </Button>

        {selectedNode && (
          <div className="border rounded-lg p-3 space-y-2 text-sm">
            <div className="font-medium">{selectedNode.label}</div>
            <Badge variant="secondary">{selectedNode.type || "Other"}</Badge>
            {selectedNode.description && (
              <p className="text-xs text-muted-foreground line-clamp-6">{selectedNode.description}</p>
            )}
          </div>
        )}
      </div>

      {/* Graph canvas */}
      <div className="flex-1 relative border rounded-lg overflow-hidden bg-white dark:bg-gray-950">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
            <span className="text-sm text-muted-foreground">Loading graph…</span>
          </div>
        )}
        <svg ref={svgRef} className="w-full h-full" />
      </div>
    </div>
  );
}

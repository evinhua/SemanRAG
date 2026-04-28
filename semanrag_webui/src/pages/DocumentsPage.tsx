import { useState, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload, Trash2, RefreshCw, Eye, ChevronLeft, ChevronRight,
  FileText, CheckCircle2, XCircle, Clock, Loader2, AlertTriangle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { documentsApi, type DocSummary, type DocDetail, type PipelineStatus } from "@/lib/semanrag";

// ── Status icon ──────────────────────────────────────────────────────

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed": return <CheckCircle2 className="h-4 w-4 text-green-500" />;
    case "failed": return <XCircle className="h-4 w-4 text-red-500" />;
    case "processing": return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
    case "pending": return <Clock className="h-4 w-4 text-yellow-500" />;
    default: return <AlertTriangle className="h-4 w-4 text-muted-foreground" />;
  }
}

// ── Main Page ────────────────────────────────────────────────────────

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [pageSize] = useState(20);
  const [statusFilter, setStatusFilter] = useState("");
  const [sortCol, setSortCol] = useState<string>("created_at");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<DocDetail | null>(null);
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const [uploading, setUploading] = useState(false);

  const loadDocs = useCallback(async () => {
    const res = await documentsApi.list({ offset: page * pageSize, limit: pageSize, status: statusFilter || undefined });
    setDocs(res.documents);
    setTotal(res.total);
  }, [page, pageSize, statusFilter]);

  const loadPipeline = useCallback(async () => {
    const res = await documentsApi.pipelineStatus();
    setPipeline(res);
  }, []);

  useEffect(() => { loadDocs(); loadPipeline(); }, [loadDocs, loadPipeline]);

  // WebSocket for live status updates
  useEffect(() => {
    let ws: WebSocket | null = null;
    try {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${proto}//${window.location.host}/ws/pipeline`);
      ws.onmessage = () => { loadDocs(); loadPipeline(); };
    } catch {
      // WebSocket not available
    }
    return () => { ws?.close(); };
  }, [loadDocs, loadPipeline]);

  // Drag-and-drop upload
  const onDrop = useCallback(async (files: File[]) => {
    setUploading(true);
    try {
      for (const file of files) {
        await documentsApi.upload(file);
      }
      loadDocs();
      loadPipeline();
    } catch (err) {
      console.error("Upload failed:", err);
    } finally {
      setUploading(false);
    }
  }, [loadDocs, loadPipeline]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { "application/pdf": [".pdf"], "text/plain": [".txt", ".md", ".csv"], "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"], "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"] } });

  // Bulk operations
  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selected.size === docs.length) setSelected(new Set());
    else setSelected(new Set(docs.map((d) => d.id)));
  };

  const bulkDelete = async () => {
    for (const id of selected) {
      await documentsApi.delete(id);
    }
    setSelected(new Set());
    loadDocs();
  };

  const bulkRetry = async () => {
    for (const id of selected) {
      await documentsApi.reingest(id);
    }
    setSelected(new Set());
    loadDocs();
  };

  const openPreview = async (id: string) => {
    const detail = await documentsApi.get(id);
    setPreview(detail);
  };

  // Sort (client-side for current page)
  const sorted = [...docs].sort((a, b) => {
    const av = (a as unknown as Record<string, unknown>)[sortCol];
    const bv = (b as unknown as Record<string, unknown>)[sortCol];
    const cmp = String(av ?? "").localeCompare(String(bv ?? ""), undefined, { numeric: true });
    return sortDir === "asc" ? cmp : -cmp;
  });

  const toggleSort = (col: string) => {
    if (sortCol === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortCol(col); setSortDir("asc"); }
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="flex h-full gap-4">
      <div className="flex-1 flex flex-col min-w-0">
        {/* Pipeline status bar */}
        {pipeline && (
          <div className="flex gap-4 mb-3 text-sm">
            <Badge variant="outline"><Clock className="h-3 w-3 mr-1" /> Pending: {pipeline.pending}</Badge>
            <Badge variant="outline"><Loader2 className="h-3 w-3 mr-1" /> Processing: {pipeline.processing}</Badge>
            <Badge variant="success"><CheckCircle2 className="h-3 w-3 mr-1" /> Completed: {pipeline.completed}</Badge>
            <Badge variant="destructive"><XCircle className="h-3 w-3 mr-1" /> Failed: {pipeline.failed}</Badge>
          </div>
        )}

        {/* Upload zone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors mb-4 ${isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50"}`}
        >
          <input {...getInputProps()} />
          <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
          {uploading ? (
            <p className="text-sm text-muted-foreground">Uploading…</p>
          ) : (
            <p className="text-sm text-muted-foreground">
              {isDragActive ? "Drop files here" : "Drag & drop files, or click to browse"}
            </p>
          )}
          <p className="text-xs text-muted-foreground mt-1">PDF, DOCX, PPTX, XLSX, TXT, MD, CSV</p>
        </div>

        {/* Filters & bulk actions */}
        <div className="flex items-center gap-3 mb-3">
          <select value={statusFilter} onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }} className="rounded-md border px-3 py-1.5 text-sm bg-background">
            <option value="">All statuses</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
          {selected.size > 0 && (
            <div className="flex gap-2 ml-auto">
              <span className="text-sm text-muted-foreground">{selected.size} selected</span>
              <Button size="sm" variant="destructive" onClick={bulkDelete}><Trash2 className="h-3 w-3 mr-1" /> Delete</Button>
              <Button size="sm" variant="outline" onClick={bulkRetry}><RefreshCw className="h-3 w-3 mr-1" /> Retry</Button>
            </div>
          )}
        </div>

        {/* Document table */}
        <div className="border rounded-lg overflow-auto flex-1">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="p-2 w-8">
                  <input type="checkbox" checked={selected.size === docs.length && docs.length > 0} onChange={toggleSelectAll} />
                </th>
                <th className="p-2 text-left cursor-pointer" onClick={() => toggleSort("status")}>Status {sortCol === "status" && (sortDir === "asc" ? "↑" : "↓")}</th>
                <th className="p-2 text-left cursor-pointer" onClick={() => toggleSort("file_path")}>File {sortCol === "file_path" && (sortDir === "asc" ? "↑" : "↓")}</th>
                <th className="p-2 text-right cursor-pointer" onClick={() => toggleSort("content_length")}>Size {sortCol === "content_length" && (sortDir === "asc" ? "↑" : "↓")}</th>
                <th className="p-2 text-right cursor-pointer" onClick={() => toggleSort("chunks_count")}>Chunks {sortCol === "chunks_count" && (sortDir === "asc" ? "↑" : "↓")}</th>
                <th className="p-2 text-left cursor-pointer" onClick={() => toggleSort("created_at")}>Created {sortCol === "created_at" && (sortDir === "asc" ? "↑" : "↓")}</th>
                <th className="p-2 w-20">Actions</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((doc) => (
                <tr key={doc.id} className="border-t hover:bg-muted/30">
                  <td className="p-2"><input type="checkbox" checked={selected.has(doc.id)} onChange={() => toggleSelect(doc.id)} /></td>
                  <td className="p-2"><div className="flex items-center gap-1.5"><StatusIcon status={doc.status} /><span className="capitalize">{doc.status}</span></div></td>
                  <td className="p-2 truncate max-w-[200px]" title={doc.file_path}>{doc.file_path || doc.id}</td>
                  <td className="p-2 text-right">{doc.content_length.toLocaleString()}</td>
                  <td className="p-2 text-right">{doc.chunks_count}</td>
                  <td className="p-2 text-muted-foreground">{new Date(doc.created_at).toLocaleDateString()}</td>
                  <td className="p-2">
                    <div className="flex gap-1">
                      <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => openPreview(doc.id)}><Eye className="h-3 w-3" /></Button>
                      <Button variant="ghost" size="icon" className="h-7 w-7" onClick={async () => { await documentsApi.delete(doc.id); loadDocs(); }}><Trash2 className="h-3 w-3" /></Button>
                    </div>
                  </td>
                </tr>
              ))}
              {docs.length === 0 && (
                <tr><td colSpan={7} className="p-8 text-center text-muted-foreground">No documents found</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between mt-3">
          <span className="text-sm text-muted-foreground">{total} documents total</span>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
              <ChevronLeft className="h-3 w-3" />
            </Button>
            <span className="text-sm">{page + 1} / {totalPages || 1}</span>
            <Button variant="outline" size="sm" disabled={page >= totalPages - 1} onClick={() => setPage((p) => p + 1)}>
              <ChevronRight className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>

      {/* Preview panel */}
      {preview && (
        <div className="w-96 shrink-0 border-l pl-4 overflow-auto">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base truncate">{preview.file_path || preview.id}</CardTitle>
                <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setPreview(null)}>
                  <FileText className="h-3 w-3" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="flex gap-2 flex-wrap">
                <Badge variant={preview.status === "completed" ? "success" : preview.status === "failed" ? "destructive" : "outline"}>
                  {preview.status}
                </Badge>
                <Badge variant="outline">v{preview.version}</Badge>
                <Badge variant="outline">{preview.chunks_count} chunks</Badge>
              </div>

              {/* Content preview */}
              <div>
                <h4 className="font-medium mb-1">Content</h4>
                <pre className="text-xs bg-muted p-2 rounded max-h-48 overflow-auto whitespace-pre-wrap">{preview.content.slice(0, 2000)}{preview.content.length > 2000 ? "…" : ""}</pre>
              </div>

              {/* PII report */}
              {preview.pii_findings.length > 0 && (
                <div>
                  <h4 className="font-medium mb-1 text-yellow-600">PII Findings ({preview.pii_findings.length})</h4>
                  <ul className="space-y-1">
                    {preview.pii_findings.map((f, i) => (
                      <li key={i} className="text-xs bg-yellow-500/10 rounded p-1.5">
                        {String(f.entity_type ?? f.type ?? "PII")}: {String(f.text ?? f.value ?? "")}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Prompt injection flags */}
              {preview.prompt_injection_flags.length > 0 && (
                <div>
                  <h4 className="font-medium mb-1 text-red-600">Injection Flags ({preview.prompt_injection_flags.length})</h4>
                  <ul className="space-y-1">
                    {preview.prompt_injection_flags.map((f, i) => (
                      <li key={i} className="text-xs bg-red-500/10 rounded p-1.5">{String(f.pattern ?? f.description ?? "Flag")}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Version history */}
              <div>
                <h4 className="font-medium mb-1">Version History</h4>
                <div className="text-xs text-muted-foreground space-y-1">
                  <div>Current: v{preview.version}</div>
                  <div>Created: {new Date(preview.created_at).toLocaleString()}</div>
                  {preview.updated_at && <div>Updated: {new Date(preview.updated_at).toLocaleString()}</div>}
                </div>
                <Button size="sm" variant="outline" className="mt-2" onClick={async () => { await documentsApi.reingest(preview.id); openPreview(preview.id); }}>
                  <RefreshCw className="h-3 w-3 mr-1" /> Re-ingest
                </Button>
              </div>

              {/* Error */}
              {preview.error_message && (
                <div className="text-xs text-red-500 bg-red-500/10 rounded p-2">{preview.error_message}</div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

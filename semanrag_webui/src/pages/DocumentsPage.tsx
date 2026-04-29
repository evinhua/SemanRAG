import { useState, useEffect, useCallback, useRef } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload, Trash2, RefreshCw, Eye, ChevronLeft, ChevronRight,
  FileText, CheckCircle2, XCircle, Clock, Loader2, AlertTriangle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { documentsApi, type DocSummary, type DocDetail, type PipelineStatus } from "@/lib/semanrag";

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed": return <CheckCircle2 className="h-4 w-4 text-eds-green" />;
    case "failed": return <XCircle className="h-4 w-4 text-eds-red" />;
    case "processing": return <Loader2 className="h-4 w-4 text-eds-blue animate-spin" />;
    case "pending": return <Clock className="h-4 w-4 text-eds-orange" />;
    default: return <AlertTriangle className="h-4 w-4 text-muted-foreground" />;
  }
}

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
  const [inboxQueue, setInboxQueue] = useState<{ file: string; size: number }[]>([]);
  const [uploadLog, setUploadLog] = useState<{ name: string; status: "uploading" | "queued" | "failed"; error?: string }[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Data loading ──────────────────────────────────────────────────
  const loadDocs = useCallback(async () => {
    try {
      const res = await documentsApi.list({ offset: page * pageSize, limit: pageSize, status: statusFilter || undefined });
      setDocs(res.documents);
      setTotal(res.total);
    } catch { /* ignore */ }
  }, [page, pageSize, statusFilter]);

  const loadPipeline = useCallback(async () => {
    try { const res = await documentsApi.pipelineStatus(); setPipeline(res); } catch { /* ignore */ }
  }, []);

  const loadInbox = useCallback(async () => {
    try { const res = await documentsApi.inboxList(); setInboxQueue(res.files); } catch { /* ignore */ }
  }, []);

  const refreshAll = useCallback(() => { loadDocs(); loadPipeline(); loadInbox(); }, [loadDocs, loadPipeline, loadInbox]);

  useEffect(() => { refreshAll(); }, [refreshAll]);

  // Stable refs for callbacks
  const refreshRef = useRef(refreshAll);
  useEffect(() => { refreshRef.current = refreshAll; }, [refreshAll]);

  // WebSocket — connect once
  useEffect(() => {
    let ws: WebSocket | null = null;
    try {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${proto}//${window.location.host}/ws/pipeline`);
      ws.onmessage = () => refreshRef.current();
    } catch { /* ignore */ }
    return () => { ws?.close(); };
  }, []);

  // Auto-poll when there are files in inbox or processing docs
  useEffect(() => {
    const hasWork = inboxQueue.length > 0 || docs.some((d) => d.status === "processing");
    if (hasWork && !pollRef.current) {
      pollRef.current = setInterval(() => refreshRef.current(), 5000);
    } else if (!hasWork && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  }, [inboxQueue.length, docs]);

  // ── Upload handler ────────────────────────────────────────────────
  const onDrop = useCallback(async (files: File[]) => {
    // Add files to upload log
    const newEntries = files.map((f) => ({ name: f.name, status: "uploading" as const }));
    setUploadLog((prev) => [...newEntries, ...prev]);

    // Upload each file to inbox
    for (let i = 0; i < files.length; i++) {
      try {
        await documentsApi.inboxUpload(files[i]!);
        setUploadLog((prev) => prev.map((e) => e.name === files[i]!.name && e.status === "uploading" ? { ...e, status: "queued" } : e));
      } catch (err) {
        setUploadLog((prev) => prev.map((e) => e.name === files[i]!.name && e.status === "uploading" ? { ...e, status: "failed", error: String(err) } : e));
      }
    }

    // Trigger background ingestion
    try { await documentsApi.inboxScan(); } catch { /* ignore — files stay in inbox for retry */ }

    // Refresh immediately
    refreshRef.current();
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { "application/pdf": [".pdf"], "text/plain": [".txt", ".md", ".csv"], "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"], "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"] } });

  // ── Table helpers ─────────────────────────────────────────────────
  const toggleSelect = (id: string) => setSelected((prev) => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const toggleSelectAll = () => setSelected(selected.size === docs.length ? new Set() : new Set(docs.map((d) => d.id)));
  const bulkDelete = async () => { for (const id of selected) await documentsApi.delete(id); setSelected(new Set()); loadDocs(); };
  const bulkRetry = async () => { for (const id of selected) await documentsApi.reingest(id); setSelected(new Set()); loadDocs(); };
  const openPreview = async (id: string) => { try { setPreview(await documentsApi.get(id)); } catch { /* ignore */ } };

  const sorted = [...docs].sort((a, b) => {
    const av = (a as unknown as Record<string, unknown>)[sortCol];
    const bv = (b as unknown as Record<string, unknown>)[sortCol];
    const cmp = String(av ?? "").localeCompare(String(bv ?? ""), undefined, { numeric: true });
    return sortDir === "asc" ? cmp : -cmp;
  });
  const toggleSort = (col: string) => { if (sortCol === col) setSortDir((d) => d === "asc" ? "desc" : "asc"); else { setSortCol(col); setSortDir("asc"); } };
  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="flex h-full gap-4">
      <div className="flex-1 flex flex-col min-w-0">
        {/* Pipeline KPI row */}
        {pipeline && (
          <div className="flex gap-3 mb-4">
            <div className="kpi-card orange flex-1 !p-3">
              <div className="text-lg font-bold text-eds-orange">{pipeline.pending}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Pending</div>
            </div>
            <div className="kpi-card flex-1 !p-3">
              <div className="text-lg font-bold text-eds-blue">{pipeline.processing}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Processing</div>
            </div>
            <div className="kpi-card green flex-1 !p-3">
              <div className="text-lg font-bold text-eds-green">{pipeline.completed}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Completed</div>
            </div>
            <div className="kpi-card red flex-1 !p-3">
              <div className="text-lg font-bold text-eds-red">{pipeline.failed}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Failed</div>
            </div>
          </div>
        )}

        {/* Upload zone */}
        <div
          {...getRootProps()}
          className={`upload-zone mb-4 ${isDragActive ? "drag-over" : ""}`}
        >
          <input {...getInputProps()} />
          <Upload className="h-8 w-8 mb-2 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            {isDragActive ? "Drop files here" : "Drag & drop files, or click to browse"}
          </p>
          <p className="text-xs text-muted-foreground mt-1">PDF, DOCX, PPTX, XLSX, TXT, MD, CSV</p>
        </div>

        {/* Upload log — shows all files uploaded this session */}
        {uploadLog.length > 0 && (
          <div className="border rounded-sm mb-4 overflow-hidden">
            <div className="px-4 py-2 bg-muted/50 border-b flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Upload History</span>
              <button onClick={() => setUploadLog([])} className="text-xs text-muted-foreground hover:text-foreground">Clear</button>
            </div>
            <div className="divide-y max-h-36 overflow-auto">
              {uploadLog.map((f, i) => (
                <div key={i} className="flex items-center gap-3 px-4 py-1.5 text-sm">
                  <FileText className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                  <span className="truncate flex-1">{f.name}</span>
                  <span className={`status-badge ${f.status === "queued" ? "success" : f.status === "uploading" ? "info" : "error"}`}>
                    {f.status === "uploading" && <Loader2 className="h-3 w-3 animate-spin inline mr-1" />}
                    {f.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Inbox queue — files being ingested in background */}
        {inboxQueue.length > 0 && (
          <div className="notification-banner blue mb-4">
            <div className="flex items-center gap-2 mb-1">
              <Loader2 className="h-4 w-4 animate-spin text-eds-blue" />
              <strong className="text-sm">{inboxQueue.length} file(s) being ingested…</strong>
            </div>
            <div className="space-y-0.5">
              {inboxQueue.map((f) => (
                <div key={f.file} className="text-xs flex items-center gap-2">
                  <FileText className="h-3 w-3" />
                  <span>{f.file}</span>
                  <span className="text-muted-foreground">({(f.size / 1024).toFixed(0)} KB)</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Filter bar */}
        <div className="flex items-center gap-3 mb-3">
          <select
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
            className="rounded-sm border border-eds-gray-300 px-3 py-1.5 text-sm bg-background"
          >
            <option value="">All statuses</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
          <Button variant="outline" size="sm" onClick={refreshAll}><RefreshCw className="h-3 w-3 mr-1" /> Refresh</Button>
          {selected.size > 0 && (
            <div className="flex gap-2 ml-auto items-center">
              <span className="text-sm text-muted-foreground">{selected.size} selected</span>
              <Button size="sm" variant="destructive" onClick={bulkDelete}><Trash2 className="h-3 w-3 mr-1" /> Delete</Button>
              <Button size="sm" variant="outline" onClick={bulkRetry}><RefreshCw className="h-3 w-3 mr-1" /> Retry</Button>
            </div>
          )}
        </div>

        {/* Document table */}
        <div className="border rounded-sm overflow-auto flex-1">
          <table className="eds-table">
            <thead>
              <tr>
                <th className="!w-8 !px-2"><input type="checkbox" checked={selected.size === docs.length && docs.length > 0} onChange={toggleSelectAll} /></th>
                <th className="cursor-pointer" onClick={() => toggleSort("status")}>Status</th>
                <th className="cursor-pointer" onClick={() => toggleSort("file_path")}>File</th>
                <th className="cursor-pointer text-right" onClick={() => toggleSort("content_length")}>Size</th>
                <th className="cursor-pointer text-right" onClick={() => toggleSort("chunks_count")}>Chunks</th>
                <th className="cursor-pointer" onClick={() => toggleSort("created_at")}>Created</th>
                <th className="!w-20">Actions</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((doc) => (
                <tr key={doc.id}>
                  <td className="!px-2"><input type="checkbox" checked={selected.has(doc.id)} onChange={() => toggleSelect(doc.id)} /></td>
                  <td>
                    <div className="flex items-center gap-1.5">
                      <StatusIcon status={doc.status} />
                      <span className={`status-badge ${doc.status === "completed" ? "success" : doc.status === "failed" ? "error" : doc.status === "processing" ? "info" : "warning"}`}>
                        {doc.status}
                      </span>
                    </div>
                  </td>
                  <td className="truncate max-w-[200px]" title={doc.file_path}>{doc.file_path?.split("/").pop() || doc.id}</td>
                  <td className="text-right">{doc.content_length.toLocaleString()}</td>
                  <td className="text-right">{doc.chunks_count}</td>
                  <td className="text-muted-foreground text-xs">{new Date(doc.created_at).toLocaleDateString()}</td>
                  <td>
                    <div className="flex gap-1">
                      <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => openPreview(doc.id)}><Eye className="h-3 w-3" /></Button>
                      <Button variant="ghost" size="icon" className="h-7 w-7" onClick={async () => { await documentsApi.delete(doc.id); loadDocs(); }}><Trash2 className="h-3 w-3" /></Button>
                    </div>
                  </td>
                </tr>
              ))}
              {docs.length === 0 && (
                <tr><td colSpan={7} className="!py-12 text-center text-muted-foreground">
                  <FileText className="h-10 w-10 mx-auto mb-2 opacity-30" />
                  No documents found
                </td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between mt-3">
          <span className="text-xs text-muted-foreground">{total} documents total</span>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}><ChevronLeft className="h-3 w-3" /></Button>
            <span className="text-sm">{page + 1} / {totalPages || 1}</span>
            <Button variant="outline" size="sm" disabled={page >= totalPages - 1} onClick={() => setPage((p) => p + 1)}><ChevronRight className="h-3 w-3" /></Button>
          </div>
        </div>
      </div>

      {/* Preview panel */}
      {preview && (
        <div className="w-96 shrink-0 border-l pl-4 overflow-auto">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="truncate">{preview.file_path?.split("/").pop() || preview.id}</CardTitle>
                <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setPreview(null)}><XCircle className="h-3 w-3" /></Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="flex gap-2 flex-wrap">
                <span className={`status-badge ${preview.status === "completed" ? "success" : preview.status === "failed" ? "error" : "info"}`}>{preview.status}</span>
                <span className="status-badge info">v{preview.version}</span>
                <span className="status-badge purple">{preview.chunks_count} chunks</span>
              </div>
              <div>
                <h4 className="font-semibold text-xs uppercase tracking-wider text-muted-foreground mb-2">Content Preview</h4>
                <pre className="text-xs bg-muted p-3 rounded-sm max-h-48 overflow-auto whitespace-pre-wrap font-mono">{preview.content.slice(0, 2000)}{preview.content.length > 2000 ? "…" : ""}</pre>
              </div>
              {preview.error_message && (
                <div className="notification-banner red"><p className="text-xs">{preview.error_message}</p></div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

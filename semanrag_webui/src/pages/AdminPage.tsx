import { useState, useEffect, useCallback } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from "recharts";
import {
  Users, Activity, Database, GitBranch, FileSearch, FlaskConical, ShieldAlert,
  ChevronLeft, ChevronRight, Trash2, Play,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { adminApi, documentsApi } from "@/lib/semanrag";

function UsersTab() {
  const [users, setUsers] = useState<string[]>([]);
  const [groups, setGroups] = useState<string[]>([]);

  useEffect(() => {
    adminApi.getUsers().then((r) => setUsers(r.users));
    adminApi.getGroups().then((r) => setGroups(r.groups));
  }, []);

  return (
    <div className="grid grid-cols-2 gap-4">
      <Card className="border-t-[3px] border-t-eds-blue">
        <CardHeader><CardTitle>Users</CardTitle></CardHeader>
        <CardContent>
          {users.length === 0 ? <p className="text-sm text-muted-foreground">No users tracked yet</p> : (
            <ul className="space-y-1">{users.map((u) => <li key={u} className="text-sm flex items-center gap-2"><Users className="h-3 w-3 text-eds-blue" />{u}</li>)}</ul>
          )}
        </CardContent>
      </Card>
      <Card className="border-t-[3px] border-t-eds-purple">
        <CardHeader><CardTitle>Groups</CardTitle></CardHeader>
        <CardContent>
          {groups.length === 0 ? <p className="text-sm text-muted-foreground">No groups found</p> : (
            <div className="flex flex-wrap gap-2">{groups.map((g) => <span key={g} className="status-badge purple">{g}</span>)}</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function TokenUsageTab() {
  const [report, setReport] = useState<{ user_usage: Record<string, number>; workspace_usage: Record<string, number> } | null>(null);
  const [budgetUser, setBudgetUser] = useState(100000);
  const [budgetWs, setBudgetWs] = useState(500000);

  useEffect(() => { adminApi.getCostReport().then(setReport); }, []);

  const chartData = report ? Object.entries(report.user_usage).map(([name, tokens]) => ({ name, tokens })) : [];

  const saveBudget = async () => {
    await adminApi.configureBudget({ max_tokens_per_user_per_day: budgetUser, max_tokens_per_workspace_per_day: budgetWs });
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader><CardTitle>Token Usage by User</CardTitle></CardHeader>
        <CardContent>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ebebeb" />
                <XAxis dataKey="name" fontSize={12} />
                <YAxis fontSize={12} />
                <RechartsTooltip />
                <Legend />
                <Bar dataKey="tokens" fill="#1174e6" name="Tokens used" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">No usage data yet</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Budget Configuration</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-end gap-4">
            <div>
              <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Max tokens/user/day</Label>
              <Input type="number" value={budgetUser} onChange={(e) => setBudgetUser(Number(e.target.value))} className="w-40 mt-1" />
            </div>
            <div>
              <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Max tokens/workspace/day</Label>
              <Input type="number" value={budgetWs} onChange={(e) => setBudgetWs(Number(e.target.value))} className="w-40 mt-1" />
            </div>
            <Button onClick={saveBudget}>Save Budget</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function CacheTab() {
  const [purging, setPurging] = useState(false);

  const purge = async (scope: string) => {
    setPurging(true);
    await adminApi.purgeCache(scope);
    setPurging(false);
  };

  return (
    <Card>
      <CardHeader><CardTitle>Cache Management</CardTitle></CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">Purge cached data to force fresh computation on next request.</p>
        <div className="flex flex-wrap gap-3">
          {["all", "llm", "vectors", "lexical"].map((scope) => (
            <Button key={scope} variant="outline" size="sm" disabled={purging} onClick={() => purge(scope)}>
              <Trash2 className="h-3 w-3 mr-1" /> Purge {scope}
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function PipelineTab() {
  const [status, setStatus] = useState<{ pending: number; processing: number; completed: number; failed: number } | null>(null);

  const load = useCallback(async () => {
    const res = await documentsApi.pipelineStatus();
    setStatus(res);
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <Card>
      <CardHeader><CardTitle>Pipeline Status</CardTitle></CardHeader>
      <CardContent>
        {status ? (
          <div className="grid grid-cols-4 gap-3">
            <div className="kpi-card orange">
              <div className="text-2xl font-bold text-eds-orange">{status.pending}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground mt-1">Pending</div>
            </div>
            <div className="kpi-card">
              <div className="text-2xl font-bold text-eds-blue">{status.processing}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground mt-1">Processing</div>
            </div>
            <div className="kpi-card green">
              <div className="text-2xl font-bold text-eds-green">{status.completed}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground mt-1">Completed</div>
            </div>
            <div className="kpi-card red">
              <div className="text-2xl font-bold text-eds-red">{status.failed}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground mt-1">Failed</div>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2"><span className="loading-spinner"></span> Loading…</div>
        )}
        <Button variant="outline" size="sm" className="mt-4" onClick={load}>
          <Activity className="h-3 w-3 mr-1" /> Refresh
        </Button>
      </CardContent>
    </Card>
  );
}

function AuditLogTab() {
  const [entries, setEntries] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const pageSize = 20;

  useEffect(() => {
    adminApi.getAuditLog({ offset: page * pageSize, limit: pageSize }).then((r) => {
      setEntries(r.entries);
      setTotal(r.total);
    });
  }, [page]);

  return (
    <Card>
      <CardHeader><CardTitle>Audit Log</CardTitle></CardHeader>
      <CardContent>
        <div className="border rounded-sm overflow-auto">
          <table className="eds-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Action</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e, i) => (
                <tr key={i}>
                  <td className="text-muted-foreground text-xs">{String(e.timestamp ?? "")}</td>
                  <td><span className="status-badge info">{String(e.action ?? "")}</span></td>
                  <td className="truncate max-w-[300px]">{String(e.detail ?? "")}</td>
                </tr>
              ))}
              {entries.length === 0 && <tr><td colSpan={3} className="!py-8 text-center text-muted-foreground">No audit entries</td></tr>}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between mt-3">
          <span className="text-xs text-muted-foreground">{total} entries</span>
          <div className="flex gap-2 items-center">
            <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}><ChevronLeft className="h-3 w-3" /></Button>
            <span className="text-sm">{page + 1}</span>
            <Button variant="outline" size="sm" disabled={(page + 1) * pageSize >= total} onClick={() => setPage((p) => p + 1)}><ChevronRight className="h-3 w-3" /></Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function EvalTab() {
  const [runs, setRuns] = useState<Record<string, unknown>[]>([]);
  const [running, setRunning] = useState(false);

  useEffect(() => { adminApi.getEvalHistory().then((r) => setRuns(r.runs)); }, []);

  const startRun = async () => {
    setRunning(true);
    await adminApi.runEval({ metrics: ["relevance", "faithfulness"] });
    const r = await adminApi.getEvalHistory();
    setRuns(r.runs);
    setRunning(false);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Button onClick={startRun} disabled={running}>
          <Play className="h-3 w-3 mr-1" /> {running ? "Running…" : "Run Evaluation"}
        </Button>
      </div>
      <Card>
        <CardHeader><CardTitle>Run History</CardTitle></CardHeader>
        <CardContent>
          {runs.length === 0 ? <p className="text-sm text-muted-foreground">No evaluation runs yet</p> : (
            <div className="border rounded-sm overflow-auto">
              <table className="eds-table">
                <thead>
                  <tr>
                    <th>Status</th>
                    <th>Run ID</th>
                    <th>Started</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r, i) => (
                    <tr key={i}>
                      <td>
                        <span className={`status-badge ${String(r.status) === "completed" ? "success" : String(r.status) === "failed" ? "error" : "info"}`}>
                          {String(r.status ?? "")}
                        </span>
                      </td>
                      <td className="font-mono text-xs">{String(r.run_id ?? "").slice(0, 8)}</td>
                      <td className="text-muted-foreground text-xs">{String(r.started_at ?? "")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function PiiReportsTab() {
  const [report, setReport] = useState<{ total_docs_with_pii: number; documents: Record<string, unknown>[] } | null>(null);

  useEffect(() => { adminApi.getPiiReport().then(setReport); }, []);

  return (
    <Card>
      <CardHeader><CardTitle>PII Reports</CardTitle></CardHeader>
      <CardContent>
        {report ? (
          <>
            <div className="notification-banner orange mb-4">
              <strong>{report.total_docs_with_pii}</strong> documents with PII findings
            </div>
            <div className="space-y-2">
              {report.documents.map((d, i) => (
                <div key={i} className="border rounded-sm p-3 text-sm hover:bg-accent/50 transition-colors">
                  <div className="font-medium">{String(d.file_path ?? d.doc_id ?? "")}</div>
                  <div className="text-xs text-muted-foreground mt-1">{(d.pii_findings as unknown[])?.length ?? 0} findings</div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="flex items-center gap-2"><span className="loading-spinner"></span> Loading…</div>
        )}
      </CardContent>
    </Card>
  );
}

const TABS = [
  { value: "users", label: "Users & Groups", icon: Users },
  { value: "tokens", label: "Token Usage", icon: Activity },
  { value: "cache", label: "Cache", icon: Database },
  { value: "pipeline", label: "Pipeline", icon: GitBranch },
  { value: "audit", label: "Audit Log", icon: FileSearch },
  { value: "eval", label: "Evaluation", icon: FlaskConical },
  { value: "pii", label: "PII Reports", icon: ShieldAlert },
] as const;

export default function AdminPage() {
  return (
    <Tabs.Root defaultValue="users" className="h-full flex flex-col">
      <Tabs.List className="flex border-b mb-4 overflow-x-auto">
        {TABS.map(({ value, label, icon: Icon }) => (
          <Tabs.Trigger
            key={value}
            value={value}
            className="flex items-center gap-2 px-4 py-2.5 text-sm border-b-2 border-transparent data-[state=active]:border-eds-blue data-[state=active]:text-eds-blue text-muted-foreground hover:text-foreground transition-colors whitespace-nowrap"
          >
            <Icon className="h-4 w-4" />
            {label}
          </Tabs.Trigger>
        ))}
      </Tabs.List>

      <div className="flex-1 overflow-auto">
        <Tabs.Content value="users"><UsersTab /></Tabs.Content>
        <Tabs.Content value="tokens"><TokenUsageTab /></Tabs.Content>
        <Tabs.Content value="cache"><CacheTab /></Tabs.Content>
        <Tabs.Content value="pipeline"><PipelineTab /></Tabs.Content>
        <Tabs.Content value="audit"><AuditLogTab /></Tabs.Content>
        <Tabs.Content value="eval"><EvalTab /></Tabs.Content>
        <Tabs.Content value="pii"><PiiReportsTab /></Tabs.Content>
      </div>
    </Tabs.Root>
  );
}

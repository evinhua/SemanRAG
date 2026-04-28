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
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { adminApi, documentsApi } from "@/lib/semanrag";

// ── Users & Groups Tab ───────────────────────────────────────────────

function UsersTab() {
  const [users, setUsers] = useState<string[]>([]);
  const [groups, setGroups] = useState<string[]>([]);

  useEffect(() => {
    adminApi.getUsers().then((r) => setUsers(r.users));
    adminApi.getGroups().then((r) => setGroups(r.groups));
  }, []);

  return (
    <div className="grid grid-cols-2 gap-4">
      <Card>
        <CardHeader><CardTitle className="text-base">Users</CardTitle></CardHeader>
        <CardContent>
          {users.length === 0 ? <p className="text-sm text-muted-foreground">No users tracked yet</p> : (
            <ul className="space-y-1">{users.map((u) => <li key={u} className="text-sm flex items-center gap-2"><Users className="h-3 w-3" />{u}</li>)}</ul>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle className="text-base">Groups</CardTitle></CardHeader>
        <CardContent>
          {groups.length === 0 ? <p className="text-sm text-muted-foreground">No groups found</p> : (
            <div className="flex flex-wrap gap-2">{groups.map((g) => <Badge key={g} variant="outline">{g}</Badge>)}</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ── Token Usage Tab ──────────────────────────────────────────────────

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
        <CardHeader><CardTitle className="text-base">Token Usage by User</CardTitle></CardHeader>
        <CardContent>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Bar dataKey="tokens" fill="hsl(var(--primary))" name="Tokens used" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">No usage data yet</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="text-base">Budget Configuration</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-end gap-4">
            <div>
              <Label>Max tokens/user/day</Label>
              <Input type="number" value={budgetUser} onChange={(e) => setBudgetUser(Number(e.target.value))} className="w-40" />
            </div>
            <div>
              <Label>Max tokens/workspace/day</Label>
              <Input type="number" value={budgetWs} onChange={(e) => setBudgetWs(Number(e.target.value))} className="w-40" />
            </div>
            <Button onClick={saveBudget}>Save Budget</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ── Cache Tab ────────────────────────────────────────────────────────

function CacheTab() {
  const [purging, setPurging] = useState(false);

  const purge = async (scope: string) => {
    setPurging(true);
    await adminApi.purgeCache(scope);
    setPurging(false);
  };

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Cache Management</CardTitle></CardHeader>
      <CardContent>
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

// ── Pipeline Tab ─────────────────────────────────────────────────────

function PipelineTab() {
  const [status, setStatus] = useState<{ pending: number; processing: number; completed: number; failed: number } | null>(null);

  const load = useCallback(async () => {
    const res = await documentsApi.pipelineStatus();
    setStatus(res);
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Pipeline Status</CardTitle></CardHeader>
      <CardContent>
        {status ? (
          <div className="grid grid-cols-4 gap-4">
            {(["pending", "processing", "completed", "failed"] as const).map((key) => (
              <div key={key} className="text-center p-4 rounded-lg bg-muted/50">
                <div className="text-2xl font-bold">{status[key]}</div>
                <div className="text-xs text-muted-foreground capitalize">{key}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">Loading…</p>
        )}
        <Button variant="outline" size="sm" className="mt-4" onClick={load}>
          <Activity className="h-3 w-3 mr-1" /> Refresh
        </Button>
      </CardContent>
    </Card>
  );
}

// ── Audit Log Tab ────────────────────────────────────────────────────

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
      <CardHeader><CardTitle className="text-base">Audit Log</CardTitle></CardHeader>
      <CardContent>
        <div className="border rounded-lg overflow-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="p-2 text-left">Timestamp</th>
                <th className="p-2 text-left">Action</th>
                <th className="p-2 text-left">Detail</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e, i) => (
                <tr key={i} className="border-t">
                  <td className="p-2 text-muted-foreground">{String(e.timestamp ?? "")}</td>
                  <td className="p-2"><Badge variant="outline">{String(e.action ?? "")}</Badge></td>
                  <td className="p-2 truncate max-w-[300px]">{String(e.detail ?? "")}</td>
                </tr>
              ))}
              {entries.length === 0 && <tr><td colSpan={3} className="p-4 text-center text-muted-foreground">No audit entries</td></tr>}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between mt-3">
          <span className="text-xs text-muted-foreground">{total} entries</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}><ChevronLeft className="h-3 w-3" /></Button>
            <span className="text-sm">{page + 1}</span>
            <Button variant="outline" size="sm" disabled={(page + 1) * pageSize >= total} onClick={() => setPage((p) => p + 1)}><ChevronRight className="h-3 w-3" /></Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ── Evaluation Tab ───────────────────────────────────────────────────

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

  const latest = runs[runs.length - 1];
  const baseline = runs[runs.length - 2];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Button onClick={startRun} disabled={running}>
          <Play className="h-3 w-3 mr-1" /> {running ? "Running…" : "Run Evaluation"}
        </Button>
      </div>
      {latest && baseline && (
        <Card>
          <CardHeader><CardTitle className="text-base">Latest vs Baseline</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div><span className="text-muted-foreground">Latest:</span> {String(latest.run_id ?? "").slice(0, 8)} — {String(latest.status ?? "")}</div>
              <div><span className="text-muted-foreground">Baseline:</span> {String(baseline.run_id ?? "").slice(0, 8)} — {String(baseline.status ?? "")}</div>
            </div>
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader><CardTitle className="text-base">Run History</CardTitle></CardHeader>
        <CardContent>
          {runs.length === 0 ? <p className="text-sm text-muted-foreground">No evaluation runs yet</p> : (
            <ul className="space-y-2">
              {runs.map((r, i) => (
                <li key={i} className="flex items-center gap-3 text-sm">
                  <Badge variant={String(r.status) === "completed" ? "success" : "outline"}>{String(r.status ?? "")}</Badge>
                  <span className="text-muted-foreground">{String(r.run_id ?? "").slice(0, 8)}</span>
                  <span className="text-muted-foreground">{String(r.started_at ?? "")}</span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ── PII Reports Tab ──────────────────────────────────────────────────

function PiiReportsTab() {
  const [report, setReport] = useState<{ total_docs_with_pii: number; documents: Record<string, unknown>[] } | null>(null);

  useEffect(() => { adminApi.getPiiReport().then(setReport); }, []);

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">PII Reports</CardTitle></CardHeader>
      <CardContent>
        {report ? (
          <>
            <p className="text-sm mb-3">{report.total_docs_with_pii} documents with PII findings</p>
            <div className="space-y-2">
              {report.documents.map((d, i) => (
                <div key={i} className="border rounded p-2 text-sm">
                  <div className="font-medium">{String(d.file_path ?? d.doc_id ?? "")}</div>
                  <div className="text-xs text-muted-foreground">{(d.pii_findings as unknown[])?.length ?? 0} findings</div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Loading…</p>
        )}
      </CardContent>
    </Card>
  );
}

// ── Main Page ────────────────────────────────────────────────────────

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
            className="flex items-center gap-2 px-4 py-2 text-sm border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:text-primary text-muted-foreground hover:text-foreground transition-colors whitespace-nowrap"
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

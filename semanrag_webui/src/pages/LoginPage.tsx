import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { LogIn, Key, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuthStore } from "@/stores/auth";
import { authApi } from "@/lib/semanrag";

export default function LoginPage() {
  const [tab, setTab] = useState<"password" | "apikey">("password");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { setAuth } = useAuthStore();
  const navigate = useNavigate();

  const loginPassword = async () => {
    setError("");
    setLoading(true);
    try {
      const res = await authApi.login(username, password);
      setAuth(res.access_token, { sub: username });
      navigate("/chat");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  const loginApiKey = () => {
    if (!apiKey.trim()) return;
    localStorage.setItem("semanrag_api_key", apiKey.trim());
    setAuth(apiKey.trim(), { sub: "api-key-user" });
    navigate("/chat");
  };

  const ssoRedirect = () => {
    window.location.href = "/auth/sso/redirect";
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* System bar */}
      <header className="flex items-center h-8 px-4 bg-eds-gray-800 text-white text-xs shrink-0">
        <span className="font-bold tracking-wide">SemanRAG</span>
        <span className="text-eds-gray-400 ml-2">Semantic Intelligence Platform</span>
      </header>

      <div className="flex flex-1 items-center justify-center p-4">
        <div className="w-full max-w-sm border rounded-sm bg-card">
          {/* Header */}
          <div className="p-6 text-center border-b">
            <h1 className="text-xl font-bold text-eds-blue">SemanRAG</h1>
            <p className="text-xs text-muted-foreground mt-1">Sign in to continue</p>
          </div>

          <div className="p-6 space-y-4">
            {/* Tab toggle */}
            <div className="flex rounded-sm border overflow-hidden">
              <button
                onClick={() => setTab("password")}
                className={`flex-1 py-2 text-sm font-medium transition-colors ${tab === "password" ? "bg-eds-blue text-white" : "bg-muted text-muted-foreground hover:text-foreground"}`}
              >
                Password
              </button>
              <button
                onClick={() => setTab("apikey")}
                className={`flex-1 py-2 text-sm font-medium transition-colors ${tab === "apikey" ? "bg-eds-blue text-white" : "bg-muted text-muted-foreground hover:text-foreground"}`}
              >
                API Key
              </button>
            </div>

            {tab === "password" ? (
              <form onSubmit={(e) => { e.preventDefault(); loginPassword(); }} className="space-y-3">
                <div>
                  <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Username</Label>
                  <Input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoFocus className="mt-1 rounded-sm" />
                </div>
                <div>
                  <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Password</Label>
                  <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" className="mt-1 rounded-sm" />
                </div>
                {error && <div className="notification-banner red !p-2 text-xs">{error}</div>}
                <Button type="submit" className="w-full" disabled={loading || !username || !password}>
                  <LogIn className="h-4 w-4 mr-2" /> {loading ? "Signing in…" : "Sign in"}
                </Button>
                <button type="button" className="text-xs text-eds-blue hover:underline w-full text-center" onClick={() => window.location.href = "/auth/reset-password"}>
                  Forgot password?
                </button>
              </form>
            ) : (
              <div className="space-y-3">
                <div>
                  <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">API Key</Label>
                  <Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-…" className="mt-1 rounded-sm" />
                </div>
                {error && <div className="notification-banner red !p-2 text-xs">{error}</div>}
                <Button className="w-full" onClick={loginApiKey} disabled={!apiKey.trim()}>
                  <Key className="h-4 w-4 mr-2" /> Authenticate
                </Button>
              </div>
            )}

            <div className="relative">
              <div className="absolute inset-0 flex items-center"><span className="w-full border-t" /></div>
              <div className="relative flex justify-center text-xs"><span className="bg-card px-2 text-muted-foreground">or</span></div>
            </div>

            <Button variant="outline" className="w-full" onClick={ssoRedirect}>
              <ExternalLink className="h-4 w-4 mr-2" /> Sign in with SSO
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

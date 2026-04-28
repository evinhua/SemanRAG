import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { LogIn, Key, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
    // Redirect to OIDC provider — configured server-side
    window.location.href = "/auth/sso/redirect";
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="w-full max-w-sm">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">SemanRAG</CardTitle>
          <p className="text-sm text-muted-foreground">Sign in to continue</p>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Tab toggle */}
          <div className="flex rounded-md border overflow-hidden">
            <button
              onClick={() => setTab("password")}
              className={`flex-1 py-1.5 text-sm ${tab === "password" ? "bg-primary text-primary-foreground" : "bg-muted"}`}
            >
              Password
            </button>
            <button
              onClick={() => setTab("apikey")}
              className={`flex-1 py-1.5 text-sm ${tab === "apikey" ? "bg-primary text-primary-foreground" : "bg-muted"}`}
            >
              API Key
            </button>
          </div>

          {tab === "password" ? (
            <form onSubmit={(e) => { e.preventDefault(); loginPassword(); }} className="space-y-3">
              <div>
                <Label>Username</Label>
                <Input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoFocus />
              </div>
              <div>
                <Label>Password</Label>
                <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" />
              </div>
              {error && <p className="text-sm text-destructive">{error}</p>}
              <Button type="submit" className="w-full" disabled={loading || !username || !password}>
                <LogIn className="h-4 w-4 mr-2" /> {loading ? "Signing in…" : "Sign in"}
              </Button>
              <button type="button" className="text-xs text-primary hover:underline w-full text-center" onClick={() => window.location.href = "/auth/reset-password"}>
                Forgot password?
              </button>
            </form>
          ) : (
            <div className="space-y-3">
              <div>
                <Label>API Key</Label>
                <Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-…" />
              </div>
              {error && <p className="text-sm text-destructive">{error}</p>}
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
        </CardContent>
      </Card>
    </div>
  );
}

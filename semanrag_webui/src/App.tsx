import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Layout } from "@/components/Layout";
import { useAuthStore } from "@/stores/auth";

const ChatPage = lazy(() => import("@/pages/ChatPage"));
const GraphPage = lazy(() => import("@/pages/GraphPage"));
const DocumentsPage = lazy(() => import("@/pages/DocumentsPage"));
const AdminPage = lazy(() => import("@/pages/AdminPage"));
const SettingsPage = lazy(() => import("@/pages/SettingsPage"));
const LoginPage = lazy(() => import("@/pages/LoginPage"));

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
});

function Loading() {
  return <div className="flex h-full items-center justify-center text-muted-foreground">Loading…</div>;
}

function AuthGuard({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuthStore();
  if (!isAuthenticated()) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route
              path="/*"
              element={
                <AuthGuard>
                  <Layout>
                    <Suspense fallback={<Loading />}>
                      <Routes>
                        <Route path="/chat" element={<ChatPage />} />
                        <Route path="/graph" element={<GraphPage />} />
                        <Route path="/documents" element={<DocumentsPage />} />
                        <Route path="/admin" element={<AdminPage />} />
                        <Route path="/settings" element={<SettingsPage />} />
                        <Route path="*" element={<Navigate to="/chat" replace />} />
                      </Routes>
                    </Suspense>
                  </Layout>
                </AuthGuard>
              }
            />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

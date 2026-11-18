import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Activity } from "lucide-react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import "./index.css";
import { HttpError, useSession } from "./lib/api";
import { PrefsProvider } from "./lib/prefs";
import { AccessPage } from "./pages/AccessPage";
import { AssessPage } from "./pages/AssessPage";
import { LandingPage } from "./pages/landing/LandingPage";
import { LoginPage } from "./pages/LoginPage";
import { ModelFactsPage } from "./pages/ModelFactsPage";
import { PatientPage } from "./pages/PatientPage";
import { PatientPortal } from "./pages/PatientPortal";
import { WardPage } from "./pages/WardPage";

// If any request finds the session expired or revoked, fall back to the sign-in page.
const onAuthError = (err: unknown) => {
  if (err instanceof HttpError && err.status === 401) queryClient.setQueryData(["me"], null);
};

const queryClient: QueryClient = new QueryClient({
  queryCache: new QueryCache({ onError: onAuthError }),
  mutationCache: new MutationCache({ onError: onAuthError }),
  defaultOptions: { queries: { retry: (n, e) => !(e instanceof HttpError && e.status < 500) && n < 1, refetchOnWindowFocus: false, staleTime: 30_000 } },
});

function Splash() {
  return (
    <div className="flex h-full items-center justify-center">
      <span className="flex size-10 animate-pulse items-center justify-center rounded-xl bg-brand text-white">
        <Activity className="size-5" strokeWidth={2.4} />
      </span>
    </div>
  );
}

function Root() {
  const { data: user, isLoading } = useSession();
  if (isLoading) return <Splash />;
  if (!user) {
    // Signed out: the landing page explains SepsisSense; any deep link goes to sign-in.
    return (
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="*" element={<LoginPage />} />
      </Routes>
    );
  }

  if (user.role === "patient") {
    // Patients only ever get their own portal, whatever URL they open.
    return (
      <Routes>
        <Route path="*" element={<PatientPortal />} />
      </Routes>
    );
  }

  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<WardPage />} />
        <Route path="/patients/:pid" element={<PatientPage />} />
        <Route path="/assess" element={<AssessPage />} />
        <Route path="/model" element={<ModelFactsPage />} />
        <Route path="/access" element={<AccessPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <PrefsProvider>
        <BrowserRouter>
          <Root />
        </BrowserRouter>
      </PrefsProvider>
    </QueryClientProvider>
  </StrictMode>,
);

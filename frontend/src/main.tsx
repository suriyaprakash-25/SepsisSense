import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import "./index.css";
import { PrefsProvider } from "./lib/prefs";
import { ModelFactsPage } from "./pages/ModelFactsPage";
import { PatientPage } from "./pages/PatientPage";
import { WardPage } from "./pages/WardPage";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 30_000 } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <PrefsProvider>
        <BrowserRouter>
          <AppShell>
            <Routes>
              <Route path="/" element={<WardPage />} />
              <Route path="/patients/:pid" element={<PatientPage />} />
              <Route path="/model" element={<ModelFactsPage />} />
              <Route path="*" element={<WardPage />} />
            </Routes>
          </AppShell>
        </BrowserRouter>
      </PrefsProvider>
    </QueryClientProvider>
  </StrictMode>,
);

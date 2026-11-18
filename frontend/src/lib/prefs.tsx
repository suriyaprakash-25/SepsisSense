import { createContext, useContext, useEffect, type ReactNode } from "react";
import { usePersisted } from "./store";

interface Prefs {
  theme: "light" | "dark";
  setTheme: (t: "light" | "dark") => void;
  research: boolean;
  setResearch: (v: boolean) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (v: boolean) => void;
}

const Ctx = createContext<Prefs | null>(null);

function systemTheme(): "light" | "dark" {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

export function PrefsProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeRaw] = usePersisted<"light" | "dark" | null>("ss.theme.v1", null);
  const [research, setResearch] = usePersisted("ss.research.v1", false);
  const [sidebarCollapsed, setSidebarCollapsed] = usePersisted("ss.sidebar.v1", false);
  const effective = theme ?? systemTheme();

  useEffect(() => {
    document.documentElement.classList.toggle("dark", effective === "dark");
    try {
      localStorage.setItem("ss.theme", effective);
    } catch {
      /* ignore */
    }
  }, [effective]);

  return (
    <Ctx.Provider
      value={{
        theme: effective,
        setTheme: (t) => {
          // Switch the class before React re-renders so charts read the new tokens.
          document.documentElement.classList.toggle("dark", t === "dark");
          setThemeRaw(t);
        },
        research,
        setResearch,
        sidebarCollapsed,
        setSidebarCollapsed,
      }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function usePrefs(): Prefs {
  const v = useContext(Ctx);
  if (!v) throw new Error("usePrefs outside PrefsProvider");
  return v;
}

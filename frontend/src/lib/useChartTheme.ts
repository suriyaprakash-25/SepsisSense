import { useMemo } from "react";
import { usePrefs } from "./prefs";

/** Concrete colours for SVG charts (Recharts sets presentation attributes, not CSS). */
export function useChartTheme() {
  const { theme } = usePrefs();
  return useMemo(() => {
    const css = getComputedStyle(document.documentElement);
    const v = (name: string) => css.getPropertyValue(name).trim();
    return {
      ink: v("--ink"),
      muted: v("--muted"),
      faint: v("--faint"),
      grid: v("--chart-grid"),
      band: v("--chart-band"),
      line: v("--line"),
      surface: v("--surface"),
      brand: v("--brand"),
      alert: v("--alert"),
      alertSoft: v("--alert-soft"),
      warn: v("--warn"),
      warnSoft: v("--warn-soft"),
      ok: v("--ok"),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
}

import clsx from "clsx";
import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import { useMemo, useState } from "react";
import { Segmented } from "../../components/ui";
import { ago, num } from "../../lib/format";
import type { LabRow } from "../../lib/types";

const SEPSIS_PANEL = ["Lactate", "WBC", "Platelets", "Creatinine", "Bilirubin_total", "pH", "HCO3", "BaseExcess"];

function MiniHistory({ lab }: { lab: LabRow }) {
  const h = lab.history;
  if (h.length < 2) return <span className="text-[11px] text-faint">single value</span>;
  const w = 64;
  const ht = 20;
  const vals = h.map((x) => x.value);
  const lo = Math.min(...vals, lab.low);
  const hi = Math.max(...vals, lab.high);
  const span = hi - lo || 1;
  const pts = h.map((x, i) => [2 + (i * (w - 4)) / (h.length - 1), ht - 2 - ((x.value - lo) / span) * (ht - 4)]);
  const yLo = ht - 2 - ((lab.low - lo) / span) * (ht - 4);
  const yHi = ht - 2 - ((lab.high - lo) / span) * (ht - 4);
  return (
    <svg width={w} height={ht} aria-hidden>
      <rect x={0} y={yHi} width={w} height={Math.max(1, yLo - yHi)} style={{ fill: "var(--chart-band)" }} />
      <polyline
        points={pts.map((p) => p.join(",")).join(" ")}
        fill="none"
        style={{ stroke: "var(--muted)" }}
        strokeWidth={1.25}
      />
      {pts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={i === pts.length - 1 ? 2.2 : 1.4} style={{ fill: "var(--ink)" }} />
      ))}
    </svg>
  );
}

function Trend({ lab }: { lab: LabRow }) {
  if (lab.previous == null || lab.value == null) return null;
  const d = lab.value - lab.previous;
  const tol = Math.max(Math.abs(lab.previous) * 0.05, 10 ** -lab.dp);
  if (Math.abs(d) < tol) return <Minus className="size-3 text-faint" aria-label="unchanged" />;
  return d > 0 ? (
    <ArrowUp className="size-3 text-muted" aria-label="rising" />
  ) : (
    <ArrowDown className="size-3 text-muted" aria-label="falling" />
  );
}

export function LabsPanel({ labs }: { labs: LabRow[] }) {
  const [view, setView] = useState<"sepsis" | "abnormal" | "all">("sepsis");
  const rows = useMemo(() => {
    if (view === "sepsis") return SEPSIS_PANEL.map((n) => labs.find((l) => l.name === n)).filter((l): l is LabRow => !!l);
    if (view === "abnormal") return labs.filter((l) => l.flag);
    return labs;
  }, [labs, view]);
  const missingSepsis = SEPSIS_PANEL.filter((n) => !labs.some((l) => l.name === n));

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 pb-3">
        <Segmented
          size="sm"
          label="Lab view"
          value={view}
          onChange={setView}
          options={[
            { value: "sepsis", label: "Sepsis panel" },
            { value: "abnormal", label: "Abnormal", count: labs.filter((l) => l.flag).length },
            { value: "all", label: "All", count: labs.length },
          ]}
        />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-y border-line bg-subtle/60 text-[11px] font-semibold tracking-wide text-faint uppercase">
              <th className="px-4 py-2 text-left">Test</th>
              <th className="px-3 py-2 text-right">Latest</th>
              <th className="px-3 py-2 text-left">Drawn</th>
              <th className="hidden px-3 py-2 text-left sm:table-cell">Recent values</th>
              <th className="px-4 py-2 text-right">Reference</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((l) => (
              <tr key={l.name} className="border-b border-line last:border-0">
                <td className="px-4 py-2">
                  <div className="font-medium text-ink">{l.label}</div>
                  <div className="text-[11px] text-faint">{l.group}</div>
                </td>
                <td className="px-3 py-2 text-right">
                  <span className="inline-flex items-center gap-1 tnum">
                    <Trend lab={l} />
                    <span className={clsx(l.flag ? "font-semibold text-warn-ink" : "text-ink")}>{num(l.value, l.dp)}</span>
                    {l.flag && (
                      <span className="rounded bg-warn-soft px-1 text-[10px] font-bold text-warn-ink">
                        {l.flag === "high" ? "H" : "L"}
                      </span>
                    )}
                  </span>
                  <div className="text-[11px] text-faint">{l.unit}</div>
                </td>
                <td className={clsx("px-3 py-2 text-xs tnum", (l.hours_ago ?? 0) > 24 ? "text-warn-ink" : "text-muted")}>
                  {ago(l.hours_ago)}
                </td>
                <td className="hidden px-3 py-2 sm:table-cell">
                  <MiniHistory lab={l} />
                </td>
                <td className="px-4 py-2 text-right text-xs text-faint tnum">
                  {num(l.low, l.dp)}–{num(l.high, l.dp)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <p className="px-4 py-6 text-center text-sm text-muted">No results in this view.</p>}
      </div>
      {view === "sepsis" && missingSepsis.length > 0 && (
        <p className="border-t border-line px-4 py-2.5 text-xs text-warn-ink">
          Not yet drawn this stay: {missingSepsis.map((n) => (n === "Bilirubin_total" ? "Bilirubin" : n)).join(", ")}
        </p>
      )}
    </div>
  );
}

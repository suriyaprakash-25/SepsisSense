import clsx from "clsx";
import { ChevronDown } from "lucide-react";
import { useState } from "react";
import { num } from "../../lib/format";
import type { Driver, MeasurementMeta } from "../../lib/types";

function primaryText(d: Driver, meas: Record<string, MeasurementMeta>) {
  if (!d.primary) return null;
  const f = d.features.find((x) => x.feature === d.primary);
  const m = meas[d.primary];
  if (!f || !m) return null;
  const flag = f.value > m.high ? "H" : f.value < m.low ? "L" : null;
  return { text: `${num(f.value, m.dp)} ${m.unit}`.trim(), flag };
}

export function DriversPanel({ drivers, measurements }: { drivers: Driver[]; measurements: Record<string, MeasurementMeta> }) {
  const [open, setOpen] = useState<string | null>(null);
  const max = Math.max(0.05, ...drivers.map((d) => Math.abs(d.contribution)));

  return (
    <div className="px-4 pb-4">
      <div className="mb-2 flex justify-between text-[11px] font-medium text-faint">
        <span>← Lowers risk</span>
        <span>Raises risk →</span>
      </div>
      <ul className="space-y-1">
        {drivers.map((d) => {
          const up = d.contribution > 0;
          const w = (Math.abs(d.contribution) / max) * 50;
          const p = primaryText(d, measurements);
          const isOpen = open === d.concept;
          return (
            <li key={d.concept} className="rounded-lg">
              <button
                onClick={() => setOpen(isOpen ? null : d.concept)}
                aria-expanded={isOpen}
                className="w-full rounded-lg px-2 py-1.5 text-left hover:bg-hover"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-[13px] font-medium text-ink">{d.label}</span>
                  <span className="flex shrink-0 items-center gap-1.5">
                    {p && (
                      <span className="text-xs text-muted tnum">
                        {p.text}
                        {p.flag && (
                          <span className="ml-1 rounded bg-warn-soft px-1 text-[10px] font-bold text-warn-ink">{p.flag}</span>
                        )}
                      </span>
                    )}
                    <ChevronDown className={clsx("size-3.5 text-faint transition-transform", isOpen && "rotate-180")} />
                  </span>
                </div>
                <div className="relative mt-1.5 h-1.5 rounded-full bg-subtle">
                  <div className="absolute inset-y-[-2px] left-1/2 w-px bg-line-strong" />
                  <div
                    className={clsx("absolute inset-y-0 rounded-full", up ? "bg-alert" : "bg-brand")}
                    style={up ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }}
                  />
                </div>
              </button>
              {isOpen && (
                <div className="mx-2 mt-1 mb-2 rounded-lg bg-subtle px-3 py-2">
                  <div className="mb-1 text-[11px] font-semibold tracking-wide text-faint uppercase">Model inputs</div>
                  <table className="w-full text-xs">
                    <tbody>
                      {d.features.map((f) => (
                        <tr key={f.feature}>
                          <td className="py-0.5 pr-2 text-muted">{f.label}</td>
                          <td className="py-0.5 pr-2 text-right text-ink tnum">{num(f.value, Math.abs(f.value) < 10 ? 2 : 0)}</td>
                          <td
                            className={clsx(
                              "w-12 py-0.5 text-right font-medium tnum",
                              f.contribution > 0 ? "text-alert-ink" : "text-brand-ink",
                            )}
                          >
                            {f.contribution > 0 ? "+" : "−"}
                            {Math.abs(f.contribution).toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-[11px] leading-4 text-faint">
        TreeSHAP contributions (log-odds) from the XGBoost and LightGBM members, grouped by clinical concept. They show
        what moved this hour's score; they do not show causes, and they are approximate for the final calibrated risk.
      </p>
    </div>
  );
}

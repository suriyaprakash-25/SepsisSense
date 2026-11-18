import clsx from "clsx";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { num, VITAL_META } from "../../lib/format";
import type { SeriesRow, VitalKey } from "../../lib/types";
import { useChartTheme } from "../../lib/useChartTheme";

const PANELS: { key: VitalKey; extra?: VitalKey[]; domain: [number, number] }[] = [
  { key: "HR", domain: [40, 160] },
  { key: "MAP", extra: ["SBP", "DBP"], domain: [40, 160] },
  { key: "Resp", domain: [5, 40] },
  { key: "Temp", domain: [35, 40] },
  { key: "O2Sat", domain: [80, 100] },
];

export function VitalsPanel({
  series,
  viewIcuHour,
  onPick,
}: {
  series: SeriesRow[];
  viewIcuHour: number;
  onPick: (index: number) => void;
}) {
  const c = useChartTheme();
  const first = series[0]?.icu_hour ?? 0;
  const last = series[series.length - 1]?.icu_hour ?? 1;
  const atView = (k: string) => {
    for (let j = series.length - 1; j >= 0; j--) {
      if (series[j].icu_hour > viewIcuHour) continue;
      if (series[j][k] != null) return { v: series[j][k] as number, ago: viewIcuHour - series[j].icu_hour };
    }
    return null;
  };

  return (
    <div className="divide-y divide-line">
      {PANELS.map(({ key, extra, domain }) => {
        const m = VITAL_META[key];
        const cur = atView(key);
        const flag = cur ? (cur.v > m.high ? "H" : cur.v < m.low ? "L" : null) : null;
        const values = series.map((r) => r[key]).filter((v): v is number => v != null);
        const lo = Math.min(domain[0], ...values);
        const hi = Math.max(domain[1], ...values);
        return (
          <div key={key} className="grid grid-cols-[7.5rem_1fr] items-center gap-2 py-1.5 pr-2 sm:grid-cols-[9rem_1fr]">
            <div className="pl-4">
              <div className="text-xs text-muted">
                {m.label} <span className="text-faint">{m.unit}</span>
              </div>
              <div className="flex items-baseline gap-1.5">
                <span className={clsx("text-lg font-semibold tnum", flag ? "text-warn-ink" : "text-ink")}>
                  {cur ? num(cur.v, m.dp) : "—"}
                </span>
                {flag && <span className="rounded bg-warn-soft px-1 text-[10px] font-bold text-warn-ink">{flag}</span>}
                {key === "MAP" && cur && (
                  <span className="text-xs text-faint tnum">
                    {num(atView("SBP")?.v, 0)}/{num(atView("DBP")?.v, 0)}
                  </span>
                )}
              </div>
              {cur && cur.ago > 0 && <div className="text-[11px] text-faint">{cur.ago} h ago</div>}
            </div>
            <div className="h-[72px]">
              <ResponsiveContainer>
                <LineChart
                  data={series}
                  margin={{ top: 6, right: 8, bottom: 2, left: 0 }}
                  syncId="vitals"
                  onClick={(e) => {
                    const idx = (e as { activeTooltipIndex?: number } | null)?.activeTooltipIndex;
                    if (typeof idx === "number" && series[idx]) onPick(series[idx].i);
                  }}
                  style={{ cursor: "pointer" }}
                >
                  <CartesianGrid stroke={c.grid} vertical={false} />
                  <ReferenceArea y1={m.low} y2={m.high} fill={c.band} fillOpacity={1} stroke="none" ifOverflow="hidden" />
                  <XAxis dataKey="icu_hour" type="number" domain={[first, last]} hide />
                  <YAxis domain={[Math.floor(lo), Math.ceil(hi)]} hide />
                  <Tooltip
                    cursor={{ stroke: c.line }}
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const r = payload[0].payload as SeriesRow;
                      return (
                        <div className="rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs shadow-pop tnum">
                          <span className="text-faint">Hr {r.icu_hour} · </span>
                          <span className="font-medium text-ink">
                            {r[key] == null ? "not charted" : `${num(r[key] as number, m.dp)} ${m.unit}`}
                          </span>
                          {key === "MAP" && r.SBP != null && (
                            <span className="text-muted"> ({num(r.SBP, 0)}/{num(r.DBP, 0)})</span>
                          )}
                        </div>
                      );
                    }}
                  />
                  {extra?.map((k) => (
                    <Line
                      key={k}
                      dataKey={k}
                      stroke={c.faint}
                      strokeWidth={1}
                      strokeOpacity={0.6}
                      dot={false}
                      connectNulls
                      isAnimationActive={false}
                    />
                  ))}
                  <Line
                    dataKey={key}
                    stroke={c.ink}
                    strokeWidth={1.6}
                    dot={series.length < 30 ? { r: 1.5, fill: c.ink } : false}
                    activeDot={{ r: 3 }}
                    connectNulls
                    isAnimationActive={false}
                  />
                  <ReferenceLine x={viewIcuHour} stroke={c.brand} strokeWidth={1.25} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
}

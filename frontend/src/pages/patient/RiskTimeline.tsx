import {
  Area,
  CartesianGrid,
  ComposedChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { RISK_SCALE_MAX, riskTicks } from "../../components/clinical";
import { pct } from "../../lib/format";
import type { SeriesRow } from "../../lib/types";
import { useChartTheme } from "../../lib/useChartTheme";

interface Props {
  series: SeriesRow[];
  viewIcuHour: number;
  elevatedCutoff: number;
  onPick: (index: number) => void;
  research?: { label: number | null; onset: number | null } | null;
}

function alertRuns(series: SeriesRow[]): [number, number][] {
  const runs: [number, number][] = [];
  let start: number | null = null;
  series.forEach((r, i) => {
    if (r.alert && start === null) start = r.icu_hour;
    const next = series[i + 1];
    if (start !== null && (!next || !next.alert)) {
      runs.push([start - 0.5, r.icu_hour + 0.5]);
      start = null;
    }
  });
  return runs;
}

export function RiskTimeline({ series, viewIcuHour, elevatedCutoff, onPick, research }: Props) {
  const c = useChartTheme();
  const runs = alertRuns(series);
  const first = series[0]?.icu_hour ?? 0;
  const last = series[series.length - 1]?.icu_hour ?? 1;

  return (
    <div className="h-56 w-full select-none" role="img" aria-label="Sepsis risk over the ICU stay">
      <ResponsiveContainer>
        <ComposedChart
          data={series}
          margin={{ top: 8, right: 16, bottom: 12, left: -12 }}
          onClick={(e) => {
            const idx = (e as { activeTooltipIndex?: number } | null)?.activeTooltipIndex;
            if (typeof idx === "number" && series[idx]) onPick(series[idx].i);
          }}
          style={{ cursor: "pointer" }}
        >
          <defs>
            <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={c.brand} stopOpacity={0.28} />
              <stop offset="100%" stopColor={c.brand} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={c.grid} vertical={false} />
          {runs.map(([a, b]) => (
            <ReferenceArea key={a} x1={a} x2={b} fill={c.alert} fillOpacity={0.12} stroke="none" ifOverflow="hidden" />
          ))}
          <ReferenceLine
            y={elevatedCutoff}
            stroke={c.warn}
            strokeDasharray="4 4"
            label={{ value: "Elevated", position: "insideTopRight", fill: c.warn, fontSize: 11 }}
          />
          {research?.label != null && research.label >= first && research.label <= last && (
            <ReferenceLine
              x={research.label}
              stroke={c.muted}
              strokeDasharray="2 3"
              label={{ value: "Label (onset − 6 h)", position: "insideTopLeft", fill: c.muted, fontSize: 11 }}
            />
          )}
          <XAxis
            dataKey="icu_hour"
            type="number"
            domain={[first, last]}
            allowDecimals={false}
            tick={{ fill: c.faint, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: c.line }}
            label={{ value: "ICU hour", position: "insideBottom", offset: -8, fill: c.faint, fontSize: 11 }}
            height={30}
          />
          <YAxis
            domain={[0, RISK_SCALE_MAX]}
            ticks={riskTicks(4)}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={{ fill: c.faint, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={44}
          />
          <Tooltip
            cursor={{ stroke: c.line, strokeWidth: 1 }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const r = payload[0].payload as SeriesRow;
              return (
                <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-pop">
                  <div className="font-medium text-ink">ICU hour {r.icu_hour}</div>
                  <div className="mt-0.5 text-muted tnum">
                    Risk <span className="font-semibold text-ink">{pct(r.risk)}</span>
                    {r.alert ? <span className="ml-2 font-semibold text-alert-ink">Model alert</span> : null}
                  </div>
                  <div className="mt-1 text-faint">Click to view the chart as of this hour</div>
                </div>
              );
            }}
          />
          <Area
            type="stepAfter"
            dataKey="risk"
            stroke={c.brand}
            strokeWidth={2}
            fill="url(#riskFill)"
            isAnimationActive={false}
            activeDot={{ r: 4, fill: c.brand, stroke: c.surface, strokeWidth: 2 }}
          />
          <ReferenceLine x={viewIcuHour} stroke={c.ink} strokeWidth={1.5} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

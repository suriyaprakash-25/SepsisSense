import clsx from "clsx";
import {
  ArrowUpDown,
  BedDouble,
  BellRing,
  ChevronRight,
  ClockAlert,
  Info,
  Search,
  TrendingUp,
  TriangleAlert,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertStatus,
  CriteriaChips,
  LastLactate,
  RiskDelta,
  RiskPill,
  Sparkline,
  VitalValue,
} from "../components/clinical";
import { Badge, Card, EmptyState, Kbd, Segmented, Skeleton, Tooltip } from "../components/ui";
import { useMeta, useReplay, useWard } from "../lib/api";
import { hasStaleVitals, icuDay, isRising, VITAL_META } from "../lib/format";
import { usePrefs } from "../lib/prefs";
import type { PatientSummary } from "../lib/types";

type Filter = "all" | "alerts" | "elevated" | "rising" | "stale";
type Sort = "priority" | "bed" | "risk";
type Unit = "all" | "MICU" | "SICU";

function priority(p: PatientSummary): number {
  if (p.alert && !p.ack) return 3;
  if (p.alert) return 2;
  if (p.tier === "elevated") return 1;
  return 0;
}

export function WardPage() {
  const { research } = usePrefs();
  const replay = useReplay();
  const { data, isLoading, isError, error, isFetching } = useWard(replay.data?.offset);
  const meta = useMeta();
  const navigate = useNavigate();
  const [filter, setFilter] = useState<Filter>("all");
  const [unit, setUnit] = useState<Unit>("all");
  const [sort, setSort] = useState<Sort>("priority");
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const searchRef = useRef<HTMLInputElement>(null);

  const all = data?.patients ?? [];
  const counts = useMemo(
    () => ({
      alerts: all.filter((p) => p.alert).length,
      unacked: all.filter((p) => p.alert && !p.ack).length,
      elevated: all.filter((p) => p.tier === "elevated").length,
      rising: all.filter(isRising).length,
      stale: all.filter(hasStaleVitals).length,
    }),
    [all],
  );

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    let r = all.filter((p) => {
      if (unit !== "all" && p.unit !== unit) return false;
      if (needle && !`${p.bed} ${p.pid}`.toLowerCase().includes(needle)) return false;
      if (filter === "alerts") return p.alert;
      if (filter === "elevated") return p.tier === "elevated";
      if (filter === "rising") return isRising(p);
      if (filter === "stale") return hasStaleVitals(p);
      return true;
    });
    r = [...r].sort((a, b) => {
      if (sort === "bed") return a.bed.localeCompare(b.bed);
      if (sort === "risk") return b.risk - a.risk;
      const pa = priority(a);
      const pb = priority(b);
      return pb - pa || b.risk - a.risk || a.bed.localeCompare(b.bed);
    });
    return r;
  }, [all, filter, unit, sort, q]);

  useEffect(() => setSel((s) => Math.min(s, Math.max(0, rows.length - 1))), [rows.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.key === "/" && !t.closest("input, textarea")) {
        e.preventDefault();
        searchRef.current?.focus();
        return;
      }
      if (t.closest("input, textarea") || e.metaKey || e.ctrlKey) return;
      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        setSel((s) => Math.min(rows.length - 1, s + 1));
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        setSel((s) => Math.max(0, s - 1));
      } else if (e.key === "Enter" && rows[sel]) {
        navigate(`/patients/${rows[sel].pid}`);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows, sel, navigate]);

  useEffect(() => {
    document.querySelector(`[data-row="${sel}"]`)?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  const units = { MICU: all.filter((p) => p.unit === "MICU").length, SICU: all.filter((p) => p.unit === "SICU").length };

  return (
    <div className="mx-auto max-w-[1440px] space-y-5 px-4 py-5 sm:px-6">
      {/* Title row */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Ward overview</h1>
          <p className="mt-0.5 text-sm text-muted">
            {isLoading ? "Loading patients…" : `${all.length} patients · sorted by who needs attention first`}
            {isFetching && !isLoading && <span className="ml-2 text-faint">updating…</span>}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Segmented<Unit>
            label="Unit"
            value={unit}
            onChange={setUnit}
            options={[
              { value: "all", label: "All units", count: all.length },
              { value: "MICU", label: "MICU", count: units.MICU },
              { value: "SICU", label: "SICU", count: units.SICU },
            ]}
          />
          <div className="relative">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-faint" />
            <input
              ref={searchRef}
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && (setQ(""), e.currentTarget.blur())}
              placeholder="Bed or patient ID"
              aria-label="Search patients"
              className="h-9 w-52 rounded-lg border border-line bg-surface pr-8 pl-8 text-sm text-ink shadow-card placeholder:text-faint focus:border-brand focus:outline-none"
            />
            {q ? (
              <button
                onClick={() => setQ("")}
                className="absolute top-1/2 right-2 -translate-y-1/2 text-faint hover:text-ink"
                aria-label="Clear search"
              >
                <X className="size-4" />
              </button>
            ) : (
              <span className="absolute top-1/2 right-2 -translate-y-1/2">
                <Kbd>/</Kbd>
              </span>
            )}
          </div>
        </div>
      </div>

      {/* KPI filter tiles */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <FilterTile
          active={filter === "alerts"}
          onClick={() => setFilter(filter === "alerts" ? "all" : "alerts")}
          icon={<BellRing className="size-4" />}
          tone="alert"
          label="Model alerts"
          value={counts.alerts}
          hint={counts.unacked ? `${counts.unacked} not yet reviewed` : counts.alerts ? "All reviewed" : "None active"}
          emphasize={counts.unacked > 0}
        />
        <FilterTile
          active={filter === "elevated"}
          onClick={() => setFilter(filter === "elevated" ? "all" : "elevated")}
          icon={<TriangleAlert className="size-4" />}
          tone="warn"
          label="Elevated risk"
          value={counts.elevated}
          hint={`≥ ${Math.round((meta.data?.elevated_cutoff ?? 0.08) * 100)}% but below alert`}
        />
        <FilterTile
          active={filter === "rising"}
          onClick={() => setFilter(filter === "rising" ? "all" : "rising")}
          icon={<TrendingUp className="size-4" />}
          tone="warn"
          label="Rising in 6 h"
          value={counts.rising}
          hint="Risk doubled, +3 pts or more"
        />
        <FilterTile
          active={filter === "stale"}
          onClick={() => setFilter(filter === "stale" ? "all" : "stale")}
          icon={<ClockAlert className="size-4" />}
          tone="neutral"
          label="Vitals overdue"
          value={counts.stale}
          hint="HR, MAP, RR or SpO₂ > 4 h old"
        />
      </div>

      {/* Patient list */}
      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium text-ink">
              {filter === "all" ? "All patients" : FILTER_LABEL[filter]}
            </span>
            <span className="text-faint tnum">{rows.length}</span>
            {filter !== "all" && (
              <button onClick={() => setFilter("all")} className="text-[13px] text-brand hover:underline">
                Clear filter
              </button>
            )}
          </div>
          <div className="flex items-center gap-2">
            <ArrowUpDown className="size-3.5 text-faint" />
            <Segmented<Sort>
              size="sm"
              label="Sort by"
              value={sort}
              onChange={setSort}
              options={[
                { value: "priority", label: "Priority" },
                { value: "risk", label: "Risk" },
                { value: "bed", label: "Bed" },
              ]}
            />
          </div>
        </div>

        {isError ? (
          <EmptyState icon={<TriangleAlert className="size-5" />} title="Could not reach the SepsisSense API">
            {String((error as Error)?.message ?? error)}. Start it with <code>uvicorn api.main:app --port 8000</code>.
          </EmptyState>
        ) : isLoading ? (
          <div className="space-y-2 p-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-12" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <EmptyState icon={<BedDouble className="size-5" />} title="No patients match">
            Try a different filter or clear the search.
          </EmptyState>
        ) : (
          <>
            <WardTable rows={rows} sel={sel} setSel={setSel} research={research} />
            <WardCards rows={rows} research={research} />
          </>
        )}
      </Card>

      <p className="flex items-start gap-2 text-xs leading-5 text-muted">
        <Info className="mt-0.5 size-3.5 shrink-0" />
        <span>
          <strong className="font-medium text-ink">Risk</strong> is the model's calibrated hourly probability that the stay
          is within 6 h of meeting Sepsis-3 criteria (ICU baseline ≈ {Math.round((meta.data?.base_rate ?? 0.023) * 1000) / 10}%).
          A <strong className="font-medium text-ink">model alert</strong> uses a threshold that balances catching sepsis against
          false alarms
          {meta.data &&
            `: on held-out patients it alerted on ${Math.round((meta.data.pool.septic_patients_alerted / Math.max(1, meta.data.pool.septic_patients)) * 100)}% of septic and ${Math.round((meta.data.pool.non_septic_patients_alerted / Math.max(1, meta.data.pool.patients - meta.data.pool.septic_patients)) * 100)}% of non-septic patients`}
          . <strong className="font-medium text-ink">A low score does not rule out sepsis.</strong>
        </span>
      </p>
    </div>
  );
}

const FILTER_LABEL: Record<Filter, string> = {
  all: "All patients",
  alerts: "Model alerts",
  elevated: "Elevated risk",
  rising: "Rising in the last 6 h",
  stale: "Vitals overdue",
};

function FilterTile({
  active,
  onClick,
  icon,
  tone,
  label,
  value,
  hint,
  emphasize,
}: {
  active: boolean;
  onClick: () => void;
  icon: ReactNode;
  tone: "alert" | "warn" | "neutral";
  label: string;
  value: number;
  hint: string;
  emphasize?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={clsx(
        "group rounded-xl border bg-surface p-3.5 text-left shadow-card transition-all hover:border-line-strong",
        active ? "border-brand ring-2 ring-brand/20" : "border-line",
      )}
    >
      <div className="flex items-center justify-between">
        <span
          className={clsx(
            "flex size-7 items-center justify-center rounded-lg",
            tone === "alert" && value > 0 ? "bg-alert-soft text-alert" : tone === "warn" && value > 0 ? "bg-warn-soft text-warn" : "bg-subtle text-faint",
            emphasize && "pulse-alert",
          )}
        >
          {icon}
        </span>
        <span className={clsx("text-2xl font-semibold tnum", value > 0 && tone === "alert" ? "text-alert-ink" : "text-ink")}>
          {value}
        </span>
      </div>
      <div className="mt-2.5 text-sm font-medium text-ink">{label}</div>
      <div className="mt-0.5 truncate text-xs text-muted">{hint}</div>
    </button>
  );
}

function ResearchCell({ p }: { p: PatientSummary }) {
  const r = p.research;
  if (!r.septic_stay) return <span className="text-xs text-faint">No</span>;
  const before = p.icu_hour < (r.sepsis_label_icu_hour ?? 0);
  return (
    <Tooltip content={`Sepsis-3 onset at ICU hour ${r.sepsis_onset_icu_hour}. PhysioNet labels hours from ${r.sepsis_label_icu_hour} (6 h early).`}>
      <Badge tone={before ? "neutral" : "brand"}>{before ? `Label in ${(r.sepsis_label_icu_hour ?? 0) - p.icu_hour} h` : `Label active`}</Badge>
    </Tooltip>
  );
}

function WardTable({
  rows,
  sel,
  setSel,
  research,
}: {
  rows: PatientSummary[];
  sel: number;
  setSel: (i: number) => void;
  research: boolean;
}) {
  const navigate = useNavigate();
  const th = "px-3 py-2 text-left text-[11px] font-semibold tracking-wide text-faint uppercase whitespace-nowrap";
  return (
    <div className="hidden overflow-x-auto lg:block">
      <table className="w-full border-collapse text-sm">
        <thead className="bg-subtle/60">
          <tr className="border-b border-line">
            <th className={clsx(th, "pl-4")}>Bed</th>
            <th className={th}>Patient</th>
            <th className={th}>Risk</th>
            <th className={th}>Last 12 h</th>
            {(["HR", "MAP", "Resp"] as const).map((k) => (
              <th key={k} className={clsx(th, "text-right")}>
                {VITAL_META[k].short}
              </th>
            ))}
            <th className={clsx(th, "hidden text-right xl:table-cell")}>Temp</th>
            <th className={clsx(th, "hidden text-right xl:table-cell")}>SpO₂</th>
            <th className={th}>Lactate</th>
            <th className={clsx(th, "hidden 2xl:table-cell")}>Criteria</th>
            {research && <th className={th}>Sepsis-3 label</th>}
            <th className={th}>Status</th>
            <th className="w-8" />
          </tr>
        </thead>
        <tbody>
          {rows.map((p, i) => {
            const ack = p.ack;
            return (
              <tr
                key={p.pid}
                data-row={i}
                onClick={() => navigate(`/patients/${p.pid}`)}
                onMouseEnter={() => setSel(i)}
                className={clsx(
                  "cursor-pointer border-b border-line last:border-0 transition-colors",
                  p.alert && !ack ? "bg-alert-soft/50 hover:bg-alert-soft" : "hover:bg-hover",
                  sel === i && "outline-2 -outline-offset-2 outline-brand/60",
                )}
              >
                <td className="relative py-2.5 pr-3 pl-4">
                  {p.alert && <span className={clsx("absolute inset-y-0 left-0 w-1", ack ? "bg-alert/40" : "bg-alert")} />}
                  <div className="font-semibold text-ink">{p.bed}</div>
                </td>
                <td className="px-3 py-2.5">
                  <div className="font-medium text-ink">{p.pid}</div>
                  <div className="text-xs text-muted tnum">
                    {p.age} {p.sex} · {icuDay(p.icu_hour)}
                  </div>
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <RiskPill risk={p.risk} tier={p.tier} />
                    {p.relative_risk !== null && p.relative_risk >= 2 && (
                      <span className="text-xs text-muted tnum" title="Relative to the ICU hourly baseline">
                        {p.relative_risk.toFixed(0)}×
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <Sparkline values={p.spark} tier={p.tier} />
                    <RiskDelta now={p.risk} before={p.risk_6h_ago} />
                  </div>
                </td>
                {(["HR", "MAP", "Resp"] as const).map((k) => (
                  <td key={k} className="px-3 py-2.5 text-right">
                    <VitalValue m={p.vitals[k]} dp={VITAL_META[k].dp} compact />
                  </td>
                ))}
                <td className="hidden px-3 py-2.5 text-right xl:table-cell">
                  <VitalValue m={p.vitals.Temp} dp={1} compact />
                </td>
                <td className="hidden px-3 py-2.5 text-right xl:table-cell">
                  <VitalValue m={p.vitals.O2Sat} dp={0} compact />
                </td>
                <td className="px-3 py-2.5">
                  <LastLactate m={p.lactate} />
                </td>
                <td className="hidden px-3 py-2.5 2xl:table-cell">
                  <CriteriaChips sirs={p.sirs} qsofa={p.qsofa} />
                </td>
                {research && (
                  <td className="px-3 py-2.5">
                    <ResearchCell p={p} />
                  </td>
                )}
                <td className="px-3 py-2.5">
                  <AlertStatus p={p} />
                  {p.record_ended && <div className="mt-0.5 text-[11px] text-faint">Record ended</div>}
                </td>
                <td className="pr-3 text-faint">
                  <ChevronRight className="size-4" />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function WardCards({ rows, research }: { rows: PatientSummary[]; research: boolean }) {
  const navigate = useNavigate();
  return (
    <ul className="divide-y divide-line lg:hidden">
      {rows.map((p) => {
        const ack = p.ack;
        return (
          <li key={p.pid}>
            <button
              onClick={() => navigate(`/patients/${p.pid}`)}
              className={clsx(
                "relative block w-full px-4 py-3 text-left",
                p.alert && !ack ? "bg-alert-soft/50" : "hover:bg-hover",
              )}
            >
              {p.alert && <span className={clsx("absolute inset-y-0 left-0 w-1", ack ? "bg-alert/40" : "bg-alert")} />}
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="font-semibold text-ink">
                    {p.bed} <span className="font-normal text-muted">· {p.pid}</span>
                  </div>
                  <div className="text-xs text-muted tnum">
                    {p.age} {p.sex} · {icuDay(p.icu_hour)}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <RiskPill risk={p.risk} tier={p.tier} />
                  <AlertStatus p={p} />
                </div>
              </div>
              <div className="mt-2 flex items-center justify-between gap-3">
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
                  {(["HR", "MAP", "Resp", "O2Sat"] as const).map((k) => (
                    <span key={k} className="inline-flex items-baseline gap-1">
                      <span className="text-xs text-faint">{VITAL_META[k].short}</span>
                      <VitalValue m={p.vitals[k]} dp={VITAL_META[k].dp} compact />
                    </span>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <Sparkline values={p.spark} tier={p.tier} width={64} height={22} />
                  <RiskDelta now={p.risk} before={p.risk_6h_ago} />
                </div>
              </div>
              {research && (
                <div className="mt-2">
                  <ResearchCell p={p} />
                </div>
              )}
            </button>
          </li>
        );
      })}
    </ul>
  );
}

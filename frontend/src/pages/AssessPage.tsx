import clsx from "clsx";
import {
  BrainCircuit,
  Calculator,
  ChevronDown,
  CircleAlert,
  ClipboardCheck,
  ClipboardPlus,
  Eraser,
  ListPlus,
  RefreshCw,
  Sparkles,
  TriangleAlert,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Area, AreaChart, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { RISK_SCALE_MAX, riskTicks } from "../components/clinical";
import { CriteriaCard, RiskSummaryCard } from "../components/riskCards";
import { Button, Card, CardHeader, EmptyState, Kbd, Segmented, Skeleton } from "../components/ui";
import { HttpError, useAssess, useAssessFields, useMeta } from "../lib/api";
import { pct } from "../lib/format";
import { usePersisted } from "../lib/store";
import type { AssessField, AssessRequest, AssessResponse } from "../lib/types";
import { useChartTheme } from "../lib/useChartTheme";
import { BundlePanel } from "./patient/BundlePanel";
import { DriversPanel } from "./patient/DriversPanel";

// ------------------------------------------------------------------ draft

interface Column {
  id: string;
  hoursAgo: number;
  values: Record<string, string>;
}

interface Draft {
  age: string;
  sex: "M" | "F";
  unit: "MICU" | "SICU";
  icuHours: string;
  hospToIcu: string;
  columns: Column[];
}

const EMPTY: Draft = {
  age: "",
  sex: "M",
  unit: "MICU",
  icuHours: "24",
  hospToIcu: "0",
  columns: [{ id: "now", hoursAgo: 0, values: {} }],
};

const CORE = ["HR", "MAP", "SBP", "DBP", "Resp", "Temp", "O2Sat", "Lactate", "WBC", "Platelets", "Creatinine", "Bilirubin_total", "pH"];

const EXAMPLES: { id: string; label: string; hint: string; draft: Draft }[] = [
  {
    id: "deteriorating",
    label: "Deteriorating",
    hint: "Fever, rising HR and RR, falling MAP over 6 h, lactate 3.8",
    draft: {
      age: "68",
      sex: "F",
      unit: "MICU",
      icuHours: "30",
      hospToIcu: "6",
      columns: [
        { id: "c6", hoursAgo: 6, values: { HR: "92", MAP: "78", SBP: "118", DBP: "62", Resp: "20", Temp: "37.9", O2Sat: "96", WBC: "13.1" } },
        { id: "c3", hoursAgo: 3, values: { HR: "108", MAP: "70", SBP: "104", DBP: "55", Resp: "24", Temp: "38.6", O2Sat: "94" } },
        {
          id: "now",
          hoursAgo: 0,
          values: { HR: "121", MAP: "62", SBP: "92", DBP: "48", Resp: "28", Temp: "39.1", O2Sat: "92", Lactate: "3.8", WBC: "17.4", Platelets: "118", Creatinine: "1.9", pH: "7.31" },
        },
      ],
    },
  },
  {
    id: "stable",
    label: "Stable",
    hint: "Normal vitals and labs at two time points",
    draft: {
      age: "54",
      sex: "M",
      unit: "SICU",
      icuHours: "20",
      hospToIcu: "2",
      columns: [
        { id: "c4", hoursAgo: 4, values: { HR: "80", MAP: "84", SBP: "124", DBP: "64", Resp: "16", Temp: "36.9", O2Sat: "98" } },
        { id: "now", hoursAgo: 0, values: { HR: "76", MAP: "86", SBP: "126", DBP: "66", Resp: "15", Temp: "36.8", O2Sat: "98", Lactate: "1.1", WBC: "8.2", Platelets: "240", Creatinine: "0.9" } },
      ],
    },
  },
];

type CellState = { value: number | null; error?: string; flag?: "H" | "L" };

function parseCell(raw: string | undefined, f: AssessField): CellState {
  const s = (raw ?? "").trim().replace(",", ".");
  if (!s) return { value: null };
  const v = Number(s);
  if (!Number.isFinite(v)) return { value: null, error: "Not a number" };
  if (v < f.min || v > f.max) return { value: v, error: `Accepted range ${f.min}–${f.max} ${f.unit}` };
  return { value: v, flag: v > f.ref_high ? "H" : v < f.ref_low ? "L" : undefined };
}

function buildRequest(d: Draft, fields: AssessField[]): { req: AssessRequest | null; problems: string[] } {
  const problems: string[] = [];
  const age = Number(d.age);
  if (!d.age.trim()) problems.push("Enter the patient's age.");
  else if (!Number.isFinite(age) || age < 18 || age > 105) problems.push("Age must be 18–105 (adult ICU model).");
  const icu = Math.round(Number(d.icuHours));
  if (!Number.isFinite(icu) || icu < 1) problems.push("Hours in ICU must be at least 1.");
  const hosp = Number(d.hospToIcu || 0);
  if (!Number.isFinite(hosp) || hosp < 0) problems.push("Hours from hospital arrival to ICU cannot be negative.");

  let any = false;
  let invalid = 0;
  const readings = d.columns.map((c) => {
    const values: Record<string, number | null> = {};
    for (const f of fields) {
      const cell = parseCell(c.values[f.name], f);
      if (cell.error) invalid++;
      else if (cell.value !== null) {
        values[f.name] = cell.value;
        any = true;
      }
    }
    return { hours_ago: c.hoursAgo, values };
  });
  if (invalid) problems.push(`${invalid} value${invalid > 1 ? "s are" : " is"} out of range or not a number.`);
  if (!any) problems.push("Enter at least one vital sign or lab value.");
  if (problems.length) return { req: null, problems };
  return {
    req: { age, sex: d.sex, unit: d.unit, icu_hours: icu, hosp_to_icu_hours: hosp, readings },
    problems,
  };
}

// ------------------------------------------------------------------- page

export function AssessPage() {
  const fieldsQ = useAssessFields();
  const meta = useMeta();
  const [draft, setDraft] = usePersisted<Draft>("ss.assess.draft.v1", EMPTY);
  const [showAll, setShowAll] = useState(false);
  const [attempted, setAttempted] = useState(false);
  const [lastSent, setLastSent] = useState<string | null>(null);
  const mutation = useAssess();
  const resultRef = useRef<HTMLDivElement>(null);

  const fields = fieldsQ.data?.fields ?? [];
  const maxBack = fieldsQ.data?.max_hours_back ?? 24;
  const columns = useMemo(() => [...draft.columns].sort((a, b) => b.hoursAgo - a.hoursAgo), [draft.columns]);
  const { req, problems } = useMemo(() => buildRequest(draft, fields), [draft, fields]);
  const reqKey = req ? JSON.stringify(req) : null;
  const stale = Boolean(mutation.data && lastSent && reqKey !== lastSent);

  const extraHasValues = fields.some((f) => !CORE.includes(f.name) && draft.columns.some((c) => (c.values[f.name] ?? "").trim()));
  const visible = fields.filter((f) => showAll || extraHasValues || CORE.includes(f.name));
  const hiddenCount = fields.length - visible.length;

  const update = (fn: (d: Draft) => Draft) => setDraft((d) => fn(d));
  const setCell = (colId: string, name: string, value: string) =>
    update((d) => ({
      ...d,
      columns: d.columns.map((c) => (c.id === colId ? { ...c, values: { ...c.values, [name]: value } } : c)),
    }));

  const addColumn = () =>
    update((d) => {
      const used = new Set(d.columns.map((c) => c.hoursAgo));
      const furthest = Math.max(...d.columns.map((c) => c.hoursAgo));
      let h = Math.min(maxBack, furthest + 3);
      while (used.has(h) && h > 0) h--;
      if (used.has(h)) for (h = 1; h <= maxBack && used.has(h); h++);
      if (h > maxBack || used.has(h)) return d;
      return { ...d, columns: [...d.columns, { id: `c${Date.now()}`, hoursAgo: h, values: {} }] };
    });

  const submit = () => {
    setAttempted(true);
    if (!req) return;
    setLastSent(JSON.stringify(req));
    mutation.mutate(req, {
      onSuccess: () => {
        if (window.matchMedia("(max-width: 1023px)").matches) {
          setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
        }
      },
    });
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        submit();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-ink">
            <ClipboardPlus className="size-5 text-brand" /> New assessment
          </h1>
          <p className="mt-0.5 max-w-2xl text-sm text-muted">
            Enter vitals and labs to score a patient with the same model used on the ward. Add earlier readings so the model
            can see trends; leave anything you don't have blank.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-faint">Load example:</span>
          {EXAMPLES.map((ex) => (
            <Button
              key={ex.id}
              size="sm"
              icon={<Sparkles className="size-3.5" />}
              title={ex.hint}
              onClick={() => {
                setDraft(ex.draft);
                setAttempted(false);
                mutation.reset();
              }}
            >
              {ex.label}
            </Button>
          ))}
          <Button
            size="sm"
            variant="ghost"
            icon={<Eraser className="size-3.5" />}
            onClick={() => {
              setDraft(EMPTY);
              setAttempted(false);
              mutation.reset();
            }}
          >
            Clear
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        {/* ------------------------------------------------------- form */}
        <div className="space-y-5 lg:col-span-3">
          <Card>
            <CardHeader title="Patient" subtitle="Used by the model; no identifiers are needed or stored" />
            <div className="grid grid-cols-2 gap-4 px-4 pb-4 sm:grid-cols-3">
              <NumberField
                label="Age"
                suffix="years"
                value={draft.age}
                onChange={(v) => update((d) => ({ ...d, age: v }))}
                invalid={attempted && problems.some((p) => p.startsWith("Enter the patient's age") || p.startsWith("Age"))}
                autoFocus
              />
              <div>
                <FieldLabel>Sex</FieldLabel>
                <Segmented
                  label="Sex"
                  value={draft.sex}
                  onChange={(v) => update((d) => ({ ...d, sex: v }))}
                  options={[
                    { value: "F", label: "Female" },
                    { value: "M", label: "Male" },
                  ]}
                />
              </div>
              <div>
                <FieldLabel>Unit</FieldLabel>
                <Segmented
                  label="Unit"
                  value={draft.unit}
                  onChange={(v) => update((d) => ({ ...d, unit: v }))}
                  options={[
                    { value: "MICU", label: "Medical" },
                    { value: "SICU", label: "Surgical" },
                  ]}
                />
              </div>
              <NumberField
                label="Hours in ICU (now)"
                suffix="h"
                value={draft.icuHours}
                onChange={(v) => update((d) => ({ ...d, icuHours: v }))}
                hint="Strongly affects this model's score"
              />
              <NumberField
                label="Hospital arrival → ICU"
                suffix="h"
                value={draft.hospToIcu}
                onChange={(v) => update((d) => ({ ...d, hospToIcu: v }))}
                hint="0 if admitted directly"
              />
            </div>
          </Card>

          <Card className="overflow-hidden">
            <CardHeader
              title="Readings"
              subtitle="Oldest on the left, now on the right · amber = outside reference range"
              action={
                <Button
                  size="sm"
                  icon={<ListPlus className="size-4" />}
                  onClick={addColumn}
                  disabled={draft.columns.length >= 6}
                >
                  Add earlier reading
                </Button>
              }
            />
            {fieldsQ.isLoading ? (
              <div className="space-y-2 p-4">
                {Array.from({ length: 8 }).map((_, i) => (
                  <Skeleton key={i} className="h-9" />
                ))}
              </div>
            ) : (
              <div className="overflow-x-auto scrollbar-thin">
                <table className="w-full border-collapse text-sm">
                  <thead>
                    <tr className="border-y border-line bg-subtle/60">
                      <th className="sticky left-0 z-10 bg-subtle px-4 py-2 text-left text-[11px] font-semibold tracking-wide text-faint uppercase">
                        Measurement
                      </th>
                      {columns.map((c) => (
                        <th key={c.id} className="min-w-[7rem] px-2 py-1.5 text-left">
                          {c.hoursAgo === 0 ? (
                            <span className="inline-flex h-8 items-center px-1 text-[13px] font-semibold text-ink">Now</span>
                          ) : (
                            <span className="flex items-center gap-1">
                              <span className="relative">
                                <select
                                  value={c.hoursAgo}
                                  onChange={(e) => {
                                    const h = Number(e.target.value);
                                    update((d) => ({
                                      ...d,
                                      columns: d.columns.map((x) => (x.id === c.id ? { ...x, hoursAgo: h } : x)),
                                    }));
                                  }}
                                  aria-label="Hours before now"
                                  className="h-8 appearance-none rounded-md border border-line bg-surface pr-6 pl-2 text-[13px] font-semibold text-ink focus:border-brand focus:outline-none"
                                >
                                  {Array.from({ length: maxBack }, (_, i) => i + 1)
                                    .filter((h) => h === c.hoursAgo || !draft.columns.some((x) => x.hoursAgo === h))
                                    .map((h) => (
                                      <option key={h} value={h}>
                                        −{h} h
                                      </option>
                                    ))}
                                </select>
                                <ChevronDown className="pointer-events-none absolute top-1/2 right-1.5 size-3.5 -translate-y-1/2 text-faint" />
                              </span>
                              <button
                                onClick={() => update((d) => ({ ...d, columns: d.columns.filter((x) => x.id !== c.id) }))}
                                className="rounded p-1 text-faint hover:bg-hover hover:text-ink"
                                aria-label={`Remove reading from ${c.hoursAgo} hours ago`}
                                title="Remove this reading"
                              >
                                <X className="size-3.5" />
                              </button>
                            </span>
                          )}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(["Vital signs", "Labs"] as const).map((section) => {
                      const rows = visible.filter((f) => (section === "Vital signs" ? f.kind === "vital" : f.kind === "lab"));
                      if (!rows.length) return null;
                      return (
                        <SectionRows key={section} title={section} span={columns.length + 1}>
                          {rows.map((f) => (
                            <tr key={f.name} className="border-b border-line last:border-0">
                              <th
                                scope="row"
                                className="sticky left-0 z-10 bg-surface px-4 py-1.5 text-left font-normal whitespace-nowrap"
                              >
                                <div className="text-[13px] font-medium text-ink">
                                  {f.label} <span className="text-xs font-normal text-faint">{f.unit}</span>
                                </div>
                                <div className="text-[11px] text-faint tnum">
                                  ref {f.ref_low}–{f.ref_high}
                                </div>
                              </th>
                              {columns.map((c) => (
                                <td key={c.id} className="px-2 py-1.5">
                                  <Cell
                                    field={f}
                                    raw={c.values[f.name] ?? ""}
                                    onChange={(v) => setCell(c.id, f.name, v)}
                                    label={`${f.label}, ${c.hoursAgo === 0 ? "now" : `${c.hoursAgo} hours ago`}`}
                                  />
                                </td>
                              ))}
                            </tr>
                          ))}
                        </SectionRows>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {(hiddenCount > 0 || (showAll && !extraHasValues)) && (
              <button
                onClick={() => setShowAll(!showAll)}
                className="flex w-full items-center justify-center gap-1 border-t border-line py-2.5 text-[13px] font-medium text-brand hover:bg-hover"
              >
                <ChevronDown className={clsx("size-4 transition-transform", showAll && "rotate-180")} />
                {showAll ? "Show fewer measurements" : `Show ${hiddenCount} more measurements (blood gas, chemistry, EtCO₂…)`}
              </button>
            )}
          </Card>

          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-0 text-[13px]">
              {attempted && problems.length > 0 ? (
                <ul className="space-y-0.5 text-alert-ink">
                  {problems.map((p) => (
                    <li key={p} className="flex items-start gap-1.5">
                      <CircleAlert className="mt-0.5 size-3.5 shrink-0" />
                      {p}
                    </li>
                  ))}
                </ul>
              ) : (
                <span className="text-faint">
                  {draft.columns.length} time point{draft.columns.length > 1 ? "s" : ""} · draft saved in this browser
                </span>
              )}
            </div>
            <Button
              variant="primary"
              icon={mutation.isPending ? <RefreshCw className="size-4 animate-spin" /> : <Calculator className="size-4" />}
              onClick={submit}
              disabled={mutation.isPending}
              className="h-10 px-5"
            >
              {mutation.data ? "Re-assess" : "Assess risk"}
              <span className="ml-1 hidden rounded bg-white/20 px-1 text-[11px] sm:inline">Ctrl ↵</span>
            </Button>
          </div>
        </div>

        {/* ---------------------------------------------------- results */}
        <div ref={resultRef} className="space-y-5 lg:col-span-2">
          <div className="space-y-5 lg:sticky lg:top-5">
            {mutation.isError ? (
              <Card className="border-alert-line">
                <div className="flex gap-3 p-4">
                  <TriangleAlert className="mt-0.5 size-5 shrink-0 text-alert" />
                  <div>
                    <div className="text-sm font-semibold text-ink">Could not assess</div>
                    <ul className="mt-1 space-y-0.5 text-[13px] text-muted">
                      {(mutation.error instanceof HttpError ? mutation.error.errors : [String(mutation.error)]).map((e) => (
                        <li key={e}>{e}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Card>
            ) : mutation.isPending && !mutation.data ? (
              <>
                <Skeleton className="h-72" />
                <Skeleton className="h-64" />
              </>
            ) : mutation.data ? (
              <Results data={mutation.data} stale={stale} meta={meta.data} onRerun={submit} />
            ) : (
              <Card>
                <EmptyState icon={<Calculator className="size-5" />} title="Results appear here">
                  Fill in what you have and press <Kbd>Ctrl</Kbd> <Kbd>Enter</Kbd>. Try an example above to see how it works.
                </EmptyState>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------- pieces

function FieldLabel({ children }: { children: ReactNode }) {
  return <span className="mb-1.5 block text-[13px] font-medium text-ink">{children}</span>;
}

function NumberField({
  label,
  suffix,
  value,
  onChange,
  hint,
  invalid,
  autoFocus,
}: {
  label: string;
  suffix?: string;
  value: string;
  onChange: (v: string) => void;
  hint?: string;
  invalid?: boolean;
  autoFocus?: boolean;
}) {
  return (
    <label className="block">
      <FieldLabel>{label}</FieldLabel>
      <span className="relative block">
        <input
          inputMode="decimal"
          value={value}
          autoFocus={autoFocus}
          onChange={(e) => onChange(e.target.value)}
          className={clsx(
            "h-9 w-full rounded-lg border bg-surface px-3 pr-12 text-sm text-ink tnum focus:outline-none",
            invalid ? "border-alert focus:border-alert" : "border-line focus:border-brand",
          )}
        />
        {suffix && <span className="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2 text-xs text-faint">{suffix}</span>}
      </span>
      {hint && <span className="mt-1 block text-[11px] text-faint">{hint}</span>}
    </label>
  );
}

function SectionRows({ title, span, children }: { title: string; span: number; children: ReactNode }) {
  return (
    <>
      <tr>
        <td colSpan={span} className="sticky left-0 bg-surface px-4 pt-3 pb-1 text-[11px] font-semibold tracking-wide text-faint uppercase">
          {title}
        </td>
      </tr>
      {children}
    </>
  );
}

function Cell({ field, raw, onChange, label }: { field: AssessField; raw: string; onChange: (v: string) => void; label: string }) {
  const st = parseCell(raw, field);
  return (
    <span className="relative block">
      <input
        inputMode="decimal"
        value={raw}
        onChange={(e) => onChange(e.target.value)}
        aria-label={label}
        aria-invalid={Boolean(st.error)}
        title={st.error}
        placeholder="—"
        className={clsx(
          "h-8 w-full min-w-[5.5rem] rounded-md border bg-surface px-2 pr-6 text-right text-[13px] tnum placeholder:text-line-strong focus:outline-none",
          st.error
            ? "border-alert bg-alert-soft text-alert-ink focus:border-alert"
            : st.flag
              ? "border-warn-line bg-warn-soft/50 font-semibold text-warn-ink focus:border-warn"
              : "border-line text-ink focus:border-brand",
        )}
      />
      {(st.flag || st.error) && (
        <span
          className={clsx(
            "pointer-events-none absolute top-1/2 right-1.5 -translate-y-1/2 text-[10px] font-bold",
            st.error ? "text-alert" : "text-warn-ink",
          )}
        >
          {st.error ? "!" : st.flag}
        </span>
      )}
    </span>
  );
}

function Results({
  data,
  stale,
  meta,
  onRerun,
}: {
  data: AssessResponse;
  stale: boolean;
  meta: Parameters<typeof RiskSummaryCard>[0]["meta"];
  onRerun: () => void;
}) {
  return (
    <>
      {stale && (
        <button
          onClick={onRerun}
          className="flex w-full items-center gap-2 rounded-xl border border-brand bg-brand-soft px-4 py-2.5 text-left text-[13px] font-medium text-brand-ink"
        >
          <RefreshCw className="size-4 shrink-0" />
          Inputs changed since this result. Click to re-assess.
        </button>
      )}
      <div className={clsx("space-y-5 transition-opacity", stale && "opacity-60")}>
        {data.warnings.length > 0 && (
          <Card className="border-warn-line bg-warn-soft/40">
            <ul className="space-y-2 p-4">
              {data.warnings.map((w) => (
                <li key={w} className="flex gap-2 text-[13px] leading-5 text-warn-ink">
                  <TriangleAlert className="mt-0.5 size-4 shrink-0" />
                  {w}
                </li>
              ))}
            </ul>
          </Card>
        )}
        <RiskSummaryCard s={data.summary} meta={meta} context="assessment" />
        {data.series.length > 1 && <WindowTrend data={data} cutoff={meta?.elevated_cutoff ?? 0.08} />}
        <Card>
          <CardHeader icon={<BrainCircuit className="size-4" />} title="What is driving this score" subtitle="Click a row for model inputs" />
          <DriversPanel drivers={data.drivers} measurements={data.measurements} />
        </Card>
        <CriteriaCard s={data.summary} />
        <Card>
          <CardHeader icon={<ClipboardCheck className="size-4" />} title="Hour-1 bundle" subtitle="Surviving Sepsis Campaign 2021" />
          <BundlePanel pid="assessment" items={data.bundle} />
        </Card>
        <p className="text-[11px] leading-4 text-faint">
          Scored on the server and not stored. Hours between your readings carry the last entered value forward, as the
          training pipeline does.
        </p>
      </div>
    </>
  );
}

function WindowTrend({ data, cutoff }: { data: AssessResponse; cutoff: number }) {
  const c = useChartTheme();
  return (
    <Card>
      <CardHeader title="Risk across your readings" subtitle="Filled dots mark hours where you entered values" />
      <div className="h-40 px-2 pb-3">
        <ResponsiveContainer>
          <AreaChart data={data.series} margin={{ top: 8, right: 16, bottom: 0, left: -12 }}>
            <XAxis
              dataKey="hours_ago"
              tickFormatter={(v) => (v === 0 ? "now" : `−${v}h`)}
              tick={{ fill: c.faint, fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: c.line }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, RISK_SCALE_MAX]}
              ticks={riskTicks(2)}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              tick={{ fill: c.faint, fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={44}
            />
            <ReferenceLine y={cutoff} stroke={c.warn} strokeDasharray="4 4" />
            <RTooltip
              cursor={{ stroke: c.line }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const r = payload[0].payload as AssessResponse["series"][number];
                return (
                  <div className="rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs shadow-pop tnum">
                    <span className="text-faint">{r.hours_ago === 0 ? "Now" : `${r.hours_ago} h ago`} · </span>
                    <span className="font-medium text-ink">{pct(r.risk)}</span>
                    {!r.entered && <span className="text-faint"> (carried forward)</span>}
                  </div>
                );
              }}
            />
            <Area
              type="stepAfter"
              dataKey="risk"
              stroke={c.brand}
              strokeWidth={2}
              fill={c.brand}
              fillOpacity={0.12}
              isAnimationActive={false}
              dot={(p: { cx?: number; cy?: number; payload?: { entered: boolean }; index?: number }) => (
                <circle
                  key={p.index}
                  cx={p.cx}
                  cy={p.cy}
                  r={p.payload?.entered ? 3.5 : 0}
                  fill={c.brand}
                  stroke={c.surface}
                  strokeWidth={1.5}
                />
              )}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

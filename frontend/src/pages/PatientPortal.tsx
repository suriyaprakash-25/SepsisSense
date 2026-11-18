import clsx from "clsx";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BellRing,
  CircleCheck,
  Droplets,
  Eye,
  HeartPulse,
  Info,
  LogOut,
  MessageCircleQuestion,
  Moon,
  ShieldAlert,
  Sun,
  Thermometer,
  TriangleAlert,
  Wind,
  type LucideIcon,
} from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { RISK_SCALE_MAX } from "../components/clinical";
import { Card, EmptyState, IconButton, Skeleton } from "../components/ui";
import { useLogout, usePatient, useReplay, useSession } from "../lib/api";
import { num } from "../lib/format";
import { usePrefs } from "../lib/prefs";
import type { LabRow, Measurement, PatientDetail, SeriesRow } from "../lib/types";
import { useChartTheme } from "../lib/useChartTheme";

// ------------------------------------------------------- plain language

const LAB_PLAIN: Record<string, string> = {
  Lactate: "Shows how well oxygen is reaching your body's tissues. A high level can be an early sign of sepsis.",
  WBC: "White blood cells fight infection. Counts that are too high or too low can mean your body is fighting one.",
  Platelets: "Tiny cells that help your blood clot.",
  Creatinine: "Shows how well your kidneys are filtering your blood.",
  BUN: "Another test of how well your kidneys are working.",
  Bilirubin_total: "Shows how well your liver is working.",
  Bilirubin_direct: "Another liver test.",
  AST: "A liver enzyme.",
  Alkalinephos: "An enzyme from the liver and bones.",
  pH: "How acidic your blood is. Your body keeps this in a very narrow range.",
  HCO3: "Bicarbonate, which helps keep your blood's acid balance steady.",
  BaseExcess: "Another measure of your blood's acid balance.",
  PaCO2: "Carbon dioxide in your blood. Shows how well you are breathing it out.",
  SaO2: "How much oxygen your blood is carrying, from a blood sample.",
  FiO2: "How much oxygen you are being given. 0.21 is normal room air.",
  Glucose: "The sugar level in your blood.",
  Potassium: "A body salt that helps your heart and muscles work.",
  Chloride: "A body salt that helps balance fluids.",
  Calcium: "A mineral needed for bones, nerves and muscles.",
  Magnesium: "A mineral that helps nerves and muscles work.",
  Phosphate: "A mineral that helps your body make energy.",
  Hgb: "Hemoglobin carries oxygen in your red blood cells.",
  Hct: "The share of your blood made up of red blood cells.",
  PTT: "How long your blood takes to clot.",
  Fibrinogen: "A protein that helps your blood clot.",
  TroponinI: "Shows whether the heart muscle is under strain.",
};

// Model inputs that are not about the patient's health (length of stay, which labs
// were ordered, age/unit). Shown to clinicians, hidden from the patient view.
const NON_CLINICAL = new Set(["time", "lab_pattern", "demographics"]);

const KEY_LABS = ["Lactate", "WBC", "Platelets", "Creatinine", "Bilirubin_total", "pH"];

const VITALS: { key: string; label: string; plain: string; unit: string; dp: number; icon: LucideIcon }[] = [
  { key: "HR", label: "Heart rate", plain: "Beats per minute", unit: "bpm", dp: 0, icon: HeartPulse },
  { key: "MAP", label: "Blood pressure", plain: "Top / bottom number", unit: "mmHg", dp: 0, icon: Activity },
  { key: "Resp", label: "Breathing rate", plain: "Breaths per minute", unit: "/min", dp: 0, icon: Wind },
  { key: "Temp", label: "Temperature", plain: "Body temperature", unit: "°C", dp: 1, icon: Thermometer },
  { key: "O2Sat", label: "Oxygen level", plain: "Oxygen in your blood", unit: "%", dp: 0, icon: Droplets },
];

function statusWord(flag: "high" | "low" | null | undefined) {
  if (flag === "high") return "Higher than usual";
  if (flag === "low") return "Lower than usual";
  return "In the usual range";
}

function agoPlain(h: number | null | undefined) {
  if (h === null || h === undefined) return "not measured yet";
  if (h === 0) return "within the last hour";
  if (h === 1) return "1 hour ago";
  if (h < 48) return `${h} hours ago`;
  return `${Math.round(h / 24)} days ago`;
}

function inHundred(risk: number) {
  return Math.max(risk > 0 ? 1 : 0, Math.round(risk * 100));
}

// --------------------------------------------------------------- shell

export function PatientPortal() {
  const { data: user } = useSession();
  const replay = useReplay(true);
  const { data, isLoading, isError, error } = usePatient(user?.pid ?? undefined, replay.data?.offset, null, true);

  return (
    <PortalShell>
      {isError ? (
        <EmptyState icon={<TriangleAlert className="size-5" />} title="We couldn't load your record">
          {String((error as Error)?.message ?? error)}. Please ask your nurse for help.
        </EmptyState>
      ) : isLoading || !data ? (
        <div className="space-y-4">
          <Skeleton className="h-40" />
          <Skeleton className="h-64" />
          <Skeleton className="h-48" />
        </div>
      ) : (
        <PortalContent d={data} name={user?.display_name ?? ""} />
      )}
    </PortalShell>
  );
}

function PortalShell({ children }: { children: ReactNode }) {
  const { data: user } = useSession();
  const logout = useLogout();
  const { theme, setTheme } = usePrefs();
  return (
    <div className="flex min-h-full flex-col">
      <div className="flex items-center gap-2 border-b border-warn-line bg-warn-soft px-4 py-1.5 text-xs text-warn-ink">
        <ShieldAlert className="size-3.5 shrink-0" />
        <span className="truncate">
          <strong className="font-semibold">Research demo.</strong> Not medical advice — always talk to your care team.
        </span>
      </div>
      <header className="sticky top-0 z-20 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-5xl items-center gap-3 px-4">
          <span className="flex size-8 items-center justify-center rounded-lg bg-brand text-white">
            <Activity className="size-[18px]" strokeWidth={2.4} />
          </span>
          <div className="leading-tight">
            <div className="text-[15px] font-semibold tracking-tight text-ink">My ICU record</div>
            <div className="text-[11px] text-faint">SepsisSense patient view</div>
          </div>
          <div className="ml-auto flex items-center gap-1">
            <span className="mr-2 hidden text-right text-xs leading-tight text-muted sm:block">
              <span className="block font-medium text-ink">{user?.display_name}</span>
              {user?.bed && <span>Bed {user.bed}</span>}
            </span>
            <IconButton label={theme === "dark" ? "Light mode" : "Dark mode"} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
              {theme === "dark" ? <Sun className="size-[18px]" /> : <Moon className="size-[18px]" />}
            </IconButton>
            <button
              onClick={() => logout.mutate()}
              className="inline-flex h-9 items-center gap-1.5 rounded-lg px-2.5 text-sm font-medium text-muted hover:bg-hover hover:text-ink"
            >
              <LogOut className="size-4" />
              <span className="hidden sm:inline">Sign out</span>
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6">{children}</main>
    </div>
  );
}

// ------------------------------------------------------------- content

function PortalContent({ d, name }: { d: PatientDetail; name: string }) {
  const s = d.summary;
  const n = inHundred(s.risk);
  const change = s.risk - s.risk_6h_ago;
  const direction = Math.abs(change) < 0.01 ? "same" : change > 0 ? "up" : "down";
  const ack = d.now_ack;
  const day = Math.floor((s.icu_hour - 1) / 24) + 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">
          {name && !/^Patient in /i.test(name) ? `Hello, ${name.split(/\s+/)[0]}` : "Welcome back"}
        </h1>
        <p className="mt-1 flex flex-wrap items-center gap-x-2 text-sm text-muted">
          <span>Bed {s.bed}</span>
          <span className="text-line-strong">·</span>
          <span>Day {day} in the intensive care unit</span>
          <span className="text-line-strong">·</span>
          <span className="inline-flex items-center gap-1.5">
            <span className="relative flex size-2">
              <span className="absolute inline-flex size-full animate-ping rounded-full bg-ok opacity-60" />
              <span className="relative inline-flex size-2 rounded-full bg-ok" />
            </span>
            Updates automatically
          </span>
          {s.record_ended && (
            <>
              <span className="text-line-strong">·</span>
              <span>Your recorded stay ends here</span>
            </>
          )}
        </p>
      </div>

      {/* Status */}
      <StatusCard tier={s.tier} ackBy={ack?.by} ackAt={ack?.at} />

      <div className="grid gap-6 md:grid-cols-5">
        {/* Risk as N in 100 */}
        <Card className="p-5 md:col-span-3">
          <h2 className="text-base font-semibold text-ink">Your sepsis risk estimate</h2>
          <p className="mt-1 text-sm text-muted">
            A computer model looks at your vital signs and blood tests every hour.
          </p>
          <div className="mt-5 flex flex-col gap-5 sm:flex-row sm:items-center">
            <IconArray filled={n} tier={s.tier} />
            <div>
              <div className="text-4xl font-semibold tracking-tight text-ink tnum">
                {n} <span className="text-xl font-medium text-muted">in 100</span>
              </div>
              <p className="mt-2 max-w-xs text-sm leading-6 text-muted">
                Out of 100 patients with readings like yours, about <strong className="text-ink">{n}</strong> may develop
                sepsis in roughly the next 6 hours.
              </p>
            </div>
          </div>
        </Card>

        {/* Outlook (future prediction) */}
        <Card className="p-5 md:col-span-2">
          <h2 className="text-base font-semibold text-ink">Looking ahead</h2>
          <p className="mt-1 text-sm text-muted">The next 6 hours, compared with 6 hours ago</p>
          <div className="mt-5 flex items-center gap-3">
            <span
              className={clsx(
                "flex size-12 items-center justify-center rounded-full",
                direction === "up" ? "bg-warn-soft text-warn" : direction === "down" ? "bg-ok-soft text-ok" : "bg-subtle text-muted",
              )}
            >
              {direction === "up" ? (
                <ArrowUpRight className="size-6" />
              ) : direction === "down" ? (
                <ArrowDownRight className="size-6" />
              ) : (
                <ArrowRight className="size-6" />
              )}
            </span>
            <div>
              <div className="text-lg font-semibold text-ink">
                {direction === "up" ? "Going up" : direction === "down" ? "Going down" : "About the same"}
              </div>
              <div className="text-sm text-muted tnum">
                {inHundred(s.risk_6h_ago)} in 100 six hours ago → {n} in 100 now
              </div>
            </div>
          </div>
          <p className="mt-4 rounded-lg bg-subtle px-3 py-2 text-xs leading-5 text-muted">
            This is an estimate, not a promise of what will happen. The model misses many cases, so a low number does not
            mean sepsis is impossible.
          </p>
        </Card>
      </div>

      <RiskHistory series={d.series} />

      {/* Vitals */}
      <section>
        <SectionTitle title="Your vital signs" subtitle="The most recent bedside readings" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {VITALS.map((v) => (
            <VitalTile key={v.key} v={v} s={s} series={d.series} />
          ))}
        </div>
      </section>

      {/* Labs */}
      <LabsSection labs={d.labs} />

      {/* Drivers, in plain words */}
      <Card className="p-5">
        <h2 className="text-base font-semibold text-ink">Which of your readings matter most</h2>
        <p className="mt-1 text-sm text-muted">
          The readings with the biggest influence on your estimate this hour. These are patterns, not causes.
        </p>
        <ul className="mt-4 grid gap-2 sm:grid-cols-2">
          {d.drivers.filter((dr) => !NON_CLINICAL.has(dr.concept)).slice(0, 4).map((dr) => (
            <li key={dr.concept} className="flex items-center gap-3 rounded-lg border border-line px-3 py-2.5">
              <span
                className={clsx(
                  "flex size-8 shrink-0 items-center justify-center rounded-full",
                  dr.contribution > 0 ? "bg-warn-soft text-warn" : "bg-ok-soft text-ok",
                )}
              >
                {dr.contribution > 0 ? <ArrowUpRight className="size-4" /> : <ArrowDownRight className="size-4" />}
              </span>
              <span>
                <span className="block text-sm font-medium text-ink">{dr.label}</span>
                <span className="block text-xs text-muted">
                  {dr.contribution > 0 ? "Pushing the estimate up" : "Pulling the estimate down"}
                </span>
              </span>
            </li>
          ))}
        </ul>
      </Card>

      {/* Safety */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="border-warn-line p-5">
          <h2 className="flex items-center gap-2 text-base font-semibold text-ink">
            <TriangleAlert className="size-5 text-warn" /> Tell your nurse straight away if you
          </h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm leading-6 text-ink">
            <li>feel confused, very drowsy, or not like yourself</li>
            <li>are short of breath or breathing very fast</li>
            <li>are shivering, very cold, or have mottled or bluish skin</li>
            <li>have much more pain than before</li>
            <li>are passing much less urine than usual</li>
          </ul>
        </Card>
        <Card className="p-5">
          <h2 className="flex items-center gap-2 text-base font-semibold text-ink">
            <MessageCircleQuestion className="size-5 text-brand" /> Questions you might ask
          </h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm leading-6 text-ink">
            <li>What do my latest results mean for my treatment?</li>
            <li>Are you checking for an infection? What tests are planned?</li>
            <li>When will my next blood tests be done?</li>
            <li>What changes should I tell you about?</li>
          </ul>
        </Card>
      </div>

      <p className="flex items-start gap-2 pb-4 text-xs leading-5 text-faint">
        <Info className="mt-0.5 size-3.5 shrink-0" />
        This page shows a research demonstration using de-identified hospital data. Your care team makes all decisions
        about your treatment. Values outside the usual range are common in intensive care and are not always a concern.
      </p>
    </div>
  );
}

function SectionTitle({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="mb-3">
      <h2 className="text-base font-semibold text-ink">{title}</h2>
      {subtitle && <p className="text-sm text-muted">{subtitle}</p>}
    </div>
  );
}

function StatusCard({ tier, ackBy, ackAt }: { tier: "alert" | "elevated" | "low"; ackBy?: string; ackAt?: string }) {
  const time = ackAt ? new Date(ackAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : null;
  const cfg =
    tier === "alert"
      ? {
          icon: <BellRing className="size-6" />,
          tone: "border-alert-line bg-alert-soft",
          iconTone: "bg-alert text-white",
          title: "Your care team has been alerted",
          body: ackBy
            ? `The early-warning tool flagged a change in your readings. ${ackBy} reviewed it${time ? ` at ${time}` : ""}.`
            : "The early-warning tool flagged a change in your readings. A member of your care team will review it.",
        }
      : tier === "elevated"
        ? {
            icon: <Eye className="size-6" />,
            tone: "border-warn-line bg-warn-soft",
            iconTone: "bg-warn text-white",
            title: "Your team is keeping a closer eye",
            body: "Your estimate is a little higher than for most ICU patients, but no alert has been raised.",
          }
        : {
            icon: <CircleCheck className="size-6" />,
            tone: "border-line bg-surface",
            iconTone: "bg-subtle text-muted",
            title: "No warning right now",
            body: "The early-warning tool has not flagged anything. It does not catch every case, so tell your nurse if you feel worse.",
          };
  return (
    <div className={clsx("flex items-start gap-4 rounded-2xl border p-5 shadow-card", cfg.tone)}>
      <span className={clsx("flex size-12 shrink-0 items-center justify-center rounded-full", cfg.iconTone)}>{cfg.icon}</span>
      <div>
        <h2 className="text-lg font-semibold text-ink">{cfg.title}</h2>
        <p className="mt-1 text-sm leading-6 text-muted">{cfg.body}</p>
      </div>
    </div>
  );
}

function IconArray({ filled, tier }: { filled: number; tier: "alert" | "elevated" | "low" }) {
  const color = tier === "alert" ? "bg-alert" : tier === "elevated" ? "bg-warn" : "bg-brand";
  return (
    <div className="grid w-fit grid-cols-10 gap-1" role="img" aria-label={`${filled} out of 100 people highlighted`}>
      {Array.from({ length: 100 }, (_, i) => (
        <span key={i} className={clsx("size-2.5 rounded-full sm:size-3", i < filled ? color : "bg-line")} />
      ))}
    </div>
  );
}

function RiskHistory({ series }: { series: SeriesRow[] }) {
  const c = useChartTheme();
  const [range, setRange] = useState<24 | 72>(24);
  const data = series.slice(-range);
  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-ink">How your estimate has changed</h2>
          <p className="text-sm text-muted">Chance of sepsis, out of 100</p>
        </div>
        <div className="inline-flex rounded-lg border border-line bg-subtle p-0.5 text-xs">
          {([24, 72] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={clsx("h-7 rounded-md px-2.5 font-medium", range === r ? "bg-surface text-ink shadow-card" : "text-muted")}
            >
              Last {r === 24 ? "day" : "3 days"}
            </button>
          ))}
        </div>
      </div>
      <div className="mt-4 h-44">
        <ResponsiveContainer>
          <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -18 }}>
            <CartesianGrid stroke={c.grid} vertical={false} />
            <XAxis
              dataKey="icu_hour"
              tickFormatter={(h) => `Day ${Math.floor((h - 1) / 24) + 1}, ${((h - 1) % 24) + 1}h`}
              tick={{ fill: c.faint, fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: c.line }}
              minTickGap={40}
            />
            <YAxis
              domain={[0, RISK_SCALE_MAX]}
              ticks={[0, 0.1, 0.2, 0.3, 0.4]}
              tickFormatter={(v) => `${Math.round(v * 100)}`}
              tick={{ fill: c.faint, fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              cursor={{ stroke: c.line }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const r = payload[0].payload as SeriesRow;
                return (
                  <div className="rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs shadow-pop">
                    <span className="text-faint">ICU hour {r.icu_hour} · </span>
                    <span className="font-medium text-ink">{inHundred(r.risk)} in 100</span>
                  </div>
                );
              }}
            />
            <Area type="stepAfter" dataKey="risk" stroke={c.brand} strokeWidth={2} fill={c.brand} fillOpacity={0.12} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function VitalTile({
  v,
  s,
  series,
}: {
  v: (typeof VITALS)[number];
  s: PatientDetail["summary"];
  series: SeriesRow[];
}) {
  const m: Measurement = s.vitals[v.key as keyof typeof s.vitals];
  const isBp = v.key === "MAP";
  const sbp = s.vitals.SBP;
  const dbp = s.vitals.DBP;
  const trend = useMemo(() => series.slice(-12).map((r) => r[v.key]).filter((x): x is number => x != null), [series, v.key]);
  const Icon = v.icon;
  const flagged = m.flag !== null && m.value !== null;
  return (
    <Card className="p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-ink">
        <Icon className="size-4 text-faint" />
        {v.label}
      </div>
      <div className="mt-2 flex items-baseline gap-1">
        <span className="text-2xl font-semibold text-ink tnum">
          {isBp && sbp.value !== null && dbp.value !== null ? `${num(sbp.value, 0)}/${num(dbp.value, 0)}` : num(m.value, v.dp)}
        </span>
        <span className="text-xs text-faint">{v.unit}</span>
      </div>
      <MiniTrend values={trend} />
      <div className={clsx("mt-2 text-xs font-medium", flagged ? "text-warn-ink" : "text-muted")}>
        {m.value === null ? "Not measured" : statusWord(m.flag)}
      </div>
      <div className="text-[11px] text-faint">{m.value === null ? v.plain : `Measured ${agoPlain(m.hours_ago)}`}</div>
    </Card>
  );
}

function MiniTrend({ values }: { values: number[] }) {
  if (values.length < 2) return <div className="mt-2 h-6" />;
  const w = 100;
  const h = 24;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${h - 2 - ((v - lo) / span) * (h - 4)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="mt-2 h-6 w-full" aria-hidden>
      <polyline points={pts} fill="none" style={{ stroke: "var(--faint)" }} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function LabsSection({ labs }: { labs: LabRow[] }) {
  const [all, setAll] = useState(false);
  const key = KEY_LABS.map((k) => labs.find((l) => l.name === k)).filter((l): l is LabRow => !!l);
  const rows = all ? labs : key;
  return (
    <section>
      <SectionTitle title="Your blood test results" subtitle="Latest result for each test, with what it measures" />
      <Card className="divide-y divide-line">
        {rows.length === 0 && <p className="p-5 text-sm text-muted">No blood test results yet.</p>}
        {rows.map((l) => {
          const flagged = Boolean(l.flag);
          const trend = l.previous != null && l.value != null ? Math.sign(l.value - l.previous) : 0;
          return (
            <div key={l.name} className="grid gap-2 p-4 sm:grid-cols-[1fr_auto] sm:items-center">
              <div>
                <div className="text-sm font-semibold text-ink">{l.label}</div>
                <div className="mt-0.5 text-[13px] leading-5 text-muted">{LAB_PLAIN[l.name] ?? l.group}</div>
              </div>
              <div className="flex items-center gap-4 sm:justify-end">
                <div className="text-right">
                  <div className="flex items-center justify-end gap-1">
                    {trend !== 0 &&
                      (trend > 0 ? (
                        <ArrowUpRight className="size-3.5 text-faint" aria-label="up from last time" />
                      ) : (
                        <ArrowDownRight className="size-3.5 text-faint" aria-label="down from last time" />
                      ))}
                    <span className={clsx("text-lg font-semibold tnum", flagged ? "text-warn-ink" : "text-ink")}>
                      {num(l.value, l.dp)}
                    </span>
                    <span className="text-xs text-faint">{l.unit}</span>
                  </div>
                  <div className="text-[11px] text-faint tnum">
                    usual {num(l.low, l.dp)}–{num(l.high, l.dp)}
                  </div>
                </div>
                <div className="w-32 text-right">
                  <span
                    className={clsx(
                      "inline-block rounded-md px-2 py-0.5 text-xs font-medium",
                      flagged ? "bg-warn-soft text-warn-ink" : "bg-subtle text-muted",
                    )}
                  >
                    {statusWord(l.flag)}
                  </span>
                  <div className="mt-1 text-[11px] text-faint">Taken {agoPlain(l.hours_ago)}</div>
                </div>
              </div>
            </div>
          );
        })}
        {labs.length > key.length && (
          <button
            onClick={() => setAll(!all)}
            className="w-full py-3 text-center text-sm font-medium text-brand hover:bg-hover"
          >
            {all ? "Show only the main tests" : `Show all ${labs.length} tests`}
          </button>
        )}
      </Card>
    </section>
  );
}

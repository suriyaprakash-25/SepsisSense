import clsx from "clsx";
import {
  ArrowDownRight,
  ArrowUpRight,
  BellRing,
  CircleCheck,
  Clock3,
  Minus,
  Siren,
  Users,
} from "lucide-react";
import { ago, num, pct, TIER_LABEL } from "../lib/format";
import type { Measurement, PatientSummary, Tier } from "../lib/types";
import type { Ack } from "../lib/store";
import { ACK_REASONS } from "../lib/store";
import { Badge, Tooltip } from "./ui";

/** Calibrated risk never exceeds ~0.39, so all risk visuals share a 0–40 % scale. */
export const RISK_SCALE_MAX = 0.4;

export const tierText: Record<Tier, string> = {
  alert: "text-alert-ink",
  elevated: "text-warn-ink",
  low: "text-ink",
};

// ---------------------------------------------------------------- RiskPill

export function RiskPill({ risk, tier, size = "md" }: { risk: number; tier: Tier; size?: "md" | "lg" }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-lg font-semibold tnum",
        size === "lg" ? "px-2.5 py-1 text-base" : "px-2 py-0.5 text-[13px]",
        tier === "alert" && "bg-alert text-white",
        tier === "elevated" && "bg-warn-soft text-warn-ink ring-1 ring-warn-line ring-inset",
        tier === "low" && "bg-subtle text-muted",
      )}
    >
      {tier === "alert" && <Siren className={size === "lg" ? "size-4" : "size-3.5"} aria-hidden />}
      {pct(risk)}
      <span className="sr-only">{TIER_LABEL[tier]}</span>
    </span>
  );
}

// ---------------------------------------------------------------- RiskBar

export function RiskBar({ risk, tier, cutoff }: { risk: number; tier: Tier; cutoff: number }) {
  const w = Math.min(1, risk / RISK_SCALE_MAX) * 100;
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-subtle" aria-hidden>
      <div
        className={clsx(
          "h-full rounded-full transition-[width] duration-500",
          tier === "alert" ? "bg-alert" : tier === "elevated" ? "bg-warn" : "bg-faint",
        )}
        style={{ width: `${Math.max(2, w)}%` }}
      />
      <div className="absolute inset-y-0 w-px bg-ink/30" style={{ left: `${(cutoff / RISK_SCALE_MAX) * 100}%` }} />
    </div>
  );
}

// --------------------------------------------------------------- Sparkline

export function Sparkline({
  values,
  tier,
  width = 88,
  height = 28,
}: {
  values: number[];
  tier: Tier;
  width?: number;
  height?: number;
}) {
  if (values.length < 2) return <div style={{ width, height }} />;
  const pad = 2;
  const max = RISK_SCALE_MAX;
  const step = (width - pad * 2) / (values.length - 1);
  const pts = values.map((v, i) => [pad + i * step, height - pad - (Math.min(v, max) / max) * (height - pad * 2)]);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = `${d} L${pts[pts.length - 1][0].toFixed(1)},${height - pad} L${pad},${height - pad} Z`;
  const color = tier === "alert" ? "var(--alert)" : tier === "elevated" ? "var(--warn)" : "var(--faint)";
  const [lx, ly] = pts[pts.length - 1];
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden className="overflow-visible">
      <path d={area} style={{ fill: color, opacity: 0.12 }} />
      <path d={d} style={{ stroke: color }} fill="none" strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={lx} cy={ly} r={2.4} style={{ fill: color }} />
    </svg>
  );
}

// ------------------------------------------------------------------ Delta

export function RiskDelta({ now, before }: { now: number; before: number }) {
  const d = (now - before) * 100;
  const rising = d >= 3 && now >= before * 2;
  if (Math.abs(d) < 0.5) {
    return (
      <span className="inline-flex items-center gap-0.5 text-xs text-faint tnum">
        <Minus className="size-3" aria-hidden />
        steady
      </span>
    );
  }
  const up = d > 0;
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-0.5 text-xs font-medium tnum",
        up ? (rising ? "text-alert-ink" : "text-warn-ink") : "text-muted",
      )}
      title="Change in calibrated risk over the last 6 hours (percentage points)"
    >
      {up ? <ArrowUpRight className="size-3.5" aria-hidden /> : <ArrowDownRight className="size-3.5" aria-hidden />}
      {up ? "+" : "−"}
      {Math.abs(d).toFixed(0)} pts
    </span>
  );
}

// ------------------------------------------------------------- VitalValue

export function VitalValue({
  m,
  dp,
  unit,
  compact,
}: {
  m: Measurement;
  dp: number;
  unit?: string;
  compact?: boolean;
}) {
  if (m.value === null) {
    return <span className="text-faint">—</span>;
  }
  const flagged = m.flag !== null;
  const value = (
    <span
      className={clsx(
        "inline-flex items-baseline gap-1 tnum",
        m.stale ? "text-faint" : flagged ? "font-semibold text-ink" : "text-ink",
      )}
    >
      {num(m.value, dp)}
      {!compact && unit && <span className="text-[11px] font-normal text-faint">{unit}</span>}
      {flagged && (
        <span
          className={clsx(
            "rounded px-0.5 text-[10px] leading-4 font-bold",
            m.stale ? "text-faint" : "bg-warn-soft text-warn-ink",
          )}
          aria-label={m.flag === "high" ? "above range" : "below range"}
        >
          {m.flag === "high" ? "H" : "L"}
        </span>
      )}
      {m.stale && <Clock3 className="size-3 self-center" aria-label="stale" />}
    </span>
  );
  if (m.stale || (m.hours_ago ?? 0) > 0) {
    return (
      <Tooltip content={`Last charted ${ago(m.hours_ago)}${m.stale ? " — may not reflect current state" : ""}`}>
        {value}
      </Tooltip>
    );
  }
  return value;
}

// ------------------------------------------------------------ Status cell

export function AlertStatus({ p, ack }: { p: PatientSummary; ack?: Ack }) {
  if (!p.alert) {
    if (p.tier === "elevated") return <span className="text-xs text-warn-ink">Watch</span>;
    return <span className="text-xs text-faint">—</span>;
  }
  if (ack) {
    const reason = ACK_REASONS.find((r) => r.id === ack.reason);
    return (
      <Tooltip content={`Acknowledged: ${reason?.label ?? ack.reason}${ack.note ? ` — “${ack.note}”` : ""}`}>
        <Badge tone="neutral" icon={<CircleCheck className="size-3" />}>
          Acknowledged
        </Badge>
      </Tooltip>
    );
  }
  return (
    <Badge tone="alert" icon={<BellRing className="size-3" />}>
      New · {p.alert_hours} h
    </Badge>
  );
}

// ------------------------------------------------------------ Agreement

export function AgreementBadge({ level, members }: { level: "high" | "moderate" | "low"; members: Record<string, number> }) {
  const names: Record<string, string> = { xgboost: "XGBoost", lightgbm: "LightGBM", random_forest: "Random forest" };
  return (
    <Tooltip
      content={
        <span className="block">
          <span className="mb-1 block font-medium">Raw score from each ensemble member</span>
          {Object.entries(members).map(([k, v]) => (
            <span key={k} className="flex justify-between gap-4 tnum">
              <span>{names[k] ?? k}</span>
              <span>{v.toFixed(2)}</span>
            </span>
          ))}
        </span>
      }
    >
      <Badge tone={level === "low" ? "warn" : "neutral"} icon={<Users className="size-3" />}>
        {level === "high" ? "Models agree" : level === "moderate" ? "Partial agreement" : "Models disagree"}
      </Badge>
    </Tooltip>
  );
}

// ------------------------------------------------------------ Criteria

export function CriteriaChips({ sirs, qsofa }: { sirs: string[]; qsofa: string[] }) {
  return (
    <span className="inline-flex gap-1">
      <Tooltip content={sirs.length ? `SIRS: ${sirs.join(", ")}` : "No SIRS criteria met in charted data"}>
        <Badge tone={sirs.length >= 2 ? "warn" : "neutral"} className="tnum">
          SIRS {sirs.length}/4
        </Badge>
      </Tooltip>
      <Tooltip content={`qSOFA (no GCS in dataset): ${qsofa.length ? qsofa.join(", ") : "none met"}`}>
        <Badge tone={qsofa.length >= 2 ? "warn" : "neutral"} className="tnum">
          qSOFA {qsofa.length}/2*
        </Badge>
      </Tooltip>
    </span>
  );
}

export function LastLactate({ m }: { m: Measurement }) {
  if (m.value === null) return <span className="text-xs text-faint">not drawn</span>;
  return (
    <span className="inline-flex flex-col leading-tight">
      <span className={clsx("tnum", m.flag === "high" ? "font-semibold text-warn-ink" : "text-ink")}>
        {num(m.value, 1)}
      </span>
      <span className={clsx("text-[11px] tnum", (m.hours_ago ?? 0) > 6 ? "text-warn-ink" : "text-faint")}>
        {ago(m.hours_ago)}
      </span>
    </span>
  );
}

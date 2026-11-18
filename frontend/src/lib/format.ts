import type { Measurement, PatientSummary, Tier } from "./types";

export const VITAL_META = {
  HR: { label: "Heart rate", short: "HR", unit: "bpm", dp: 0, low: 60, high: 100 },
  MAP: { label: "Mean arterial pressure", short: "MAP", unit: "mmHg", dp: 0, low: 65, high: 110 },
  SBP: { label: "Systolic BP", short: "SBP", unit: "mmHg", dp: 0, low: 90, high: 140 },
  DBP: { label: "Diastolic BP", short: "DBP", unit: "mmHg", dp: 0, low: 60, high: 90 },
  Resp: { label: "Respiratory rate", short: "RR", unit: "/min", dp: 0, low: 12, high: 20 },
  Temp: { label: "Temperature", short: "Temp", unit: "°C", dp: 1, low: 36, high: 38 },
  O2Sat: { label: "SpO₂", short: "SpO₂", unit: "%", dp: 0, low: 94, high: 100 },
  EtCO2: { label: "End-tidal CO₂", short: "EtCO₂", unit: "mmHg", dp: 0, low: 35, high: 45 },
} as const;

export const TIER_LABEL: Record<Tier, string> = {
  alert: "Model alert",
  elevated: "Elevated",
  low: "Not elevated",
};

export function pct(v: number, digits = 0): string {
  const x = v * 100;
  if (x > 0 && x < 1) return "<1%";
  return `${x.toFixed(digits)}%`;
}

export function num(v: number | null | undefined, dp = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(dp);
}

export function ago(hours: number | null | undefined): string {
  if (hours === null || hours === undefined) return "never";
  if (hours === 0) return "this hour";
  if (hours < 24) return `${hours} h ago`;
  const d = Math.floor(hours / 24);
  const h = hours % 24;
  return h ? `${d} d ${h} h ago` : `${d} d ago`;
}

export function icuDay(icuHour: number): string {
  const day = Math.floor((icuHour - 1) / 24) + 1;
  return `Day ${day}, hr ${icuHour}`;
}

export function riskDelta(p: Pick<PatientSummary, "risk" | "risk_6h_ago">): number {
  return p.risk - p.risk_6h_ago;
}

/** "Rising" = calibrated risk at least doubled and up by >= 3 percentage points in 6 h. */
export function isRising(p: Pick<PatientSummary, "risk" | "risk_6h_ago">): boolean {
  return p.risk - p.risk_6h_ago >= 0.03 && p.risk >= p.risk_6h_ago * 2;
}

export function hasStaleVitals(p: PatientSummary): boolean {
  return (["HR", "MAP", "Resp", "O2Sat"] as const).some((k) => p.vitals[k].stale || p.vitals[k].value === null);
}

export function measurementText(m: Measurement, dp: number): string {
  return num(m.value, dp);
}

export const sexLabel = (s: "M" | "F") => (s === "M" ? "Male" : "Female");

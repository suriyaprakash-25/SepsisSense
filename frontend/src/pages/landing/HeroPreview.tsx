import clsx from "clsx";
import { ArrowUpRight, BellRing, CircleCheck, HeartPulse, Siren } from "lucide-react";

/**
 * A hand-built, static preview of the product for the landing page. Values mirror
 * the demo ward on model v3 (held-out test patients; SICU-03 is the highest-risk
 * alert, and its real top drivers are shown, including time in ICU).
 */

const ROWS = [
  { bed: "SICU-03", who: "69 M · Day 4", risk: 13, tier: "alert", spark: [14, 17, 14, 14, 14, 14, 14, 14, 13], hr: 89, map: 78 },
  { bed: "MICU-10", who: "50 M · Day 4", risk: 11, tier: "alert", spark: [14, 10, 9, 11, 10, 10, 10, 13, 11], hr: 94, map: 92 },
  { bed: "MICU-14", who: "56 M · Day 3", risk: 7, tier: "alert", spark: [7, 7, 8, 9, 9, 9, 7, 7, 7], hr: 62, map: 76 },
  { bed: "MICU-11", who: "70 M · Day 3", risk: 4, tier: "elevated", spark: [3, 2, 2, 3, 4, 4, 6, 4, 4], hr: 84, map: 60 },
] as const;

function Spark({ values, tier }: { values: readonly number[]; tier: string }) {
  const w = 64;
  const h = 20;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${h - 2 - (v / 20) * (h - 4)}`).join(" ");
  const color = tier === "alert" ? "var(--alert)" : tier === "elevated" ? "var(--warn)" : "var(--faint)";
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden className="overflow-visible">
      <polyline points={pts} fill="none" style={{ stroke: color }} strokeWidth={1.6} strokeLinejoin="round" className="landing-draw" />
    </svg>
  );
}

function Pill({ risk, tier }: { risk: number; tier: string }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-semibold tnum",
        tier === "alert" && "bg-alert text-white",
        tier === "elevated" && "bg-warn-soft text-warn-ink ring-1 ring-warn-line ring-inset",
        tier === "low" && "bg-subtle text-muted",
      )}
    >
      {tier === "alert" && <Siren className="size-3" />}
      {risk}%
    </span>
  );
}

export function HeroPreview() {
  return (
    <div className="relative mx-auto w-full max-w-[560px] select-none" aria-label="Preview of the SepsisSense ward overview and patient view" role="img">
      {/* Ward card */}
      <div className="relative rounded-2xl border border-line bg-surface shadow-pop">
        <div className="flex items-center gap-1.5 border-b border-line px-4 py-2.5">
          <span className="size-2.5 rounded-full bg-line-strong" />
          <span className="size-2.5 rounded-full bg-line-strong" />
          <span className="size-2.5 rounded-full bg-line-strong" />
          <span className="ml-3 text-xs font-medium text-muted">Ward overview · sorted by who needs attention first</span>
        </div>
        <div className="grid grid-cols-3 gap-2 px-4 pt-3">
          {[
            { k: "Model alerts", v: "5", tone: "text-alert-ink" },
            { k: "Elevated", v: "1", tone: "text-warn-ink" },
            { k: "Beds", v: "18", tone: "text-ink" },
          ].map((t) => (
            <div key={t.k} className="rounded-lg border border-line px-2.5 py-2">
              <div className={clsx("text-lg font-semibold tnum", t.tone)}>{t.v}</div>
              <div className="text-[10px] text-muted">{t.k}</div>
            </div>
          ))}
        </div>
        <table className="mt-3 w-full text-[12px]">
          <thead>
            <tr className="border-y border-line bg-subtle/60 text-[9px] font-semibold tracking-wide text-faint uppercase">
              <th className="px-4 py-1.5 text-left">Bed</th>
              <th className="px-2 py-1.5 text-left">Risk</th>
              <th className="px-2 py-1.5 text-left">12 h</th>
              <th className="px-2 py-1.5 text-right">HR</th>
              <th className="px-4 py-1.5 text-right">MAP</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.bed} className={clsx("border-b border-line last:border-0", r.tier === "alert" && "bg-alert-soft/60")}>
                <td className="relative px-4 py-2">
                  {r.tier === "alert" && <span className="absolute inset-y-0 left-0 w-1 bg-alert" />}
                  <div className="font-semibold text-ink">{r.bed}</div>
                  <div className="text-[10px] text-muted">{r.who}</div>
                </td>
                <td className="px-2 py-2">
                  <Pill risk={r.risk} tier={r.tier} />
                </td>
                <td className="px-2 py-2">
                  <Spark values={r.spark} tier={r.tier} />
                </td>
                <td className="px-2 py-2 text-right text-ink tnum">
                  {r.hr}
                  {r.hr > 100 && <span className="ml-0.5 rounded bg-warn-soft px-0.5 text-[9px] font-bold text-warn-ink">H</span>}
                </td>
                <td className="px-4 py-2 text-right text-ink tnum">
                  {r.map}
                  {r.map < 65 && <span className="ml-0.5 rounded bg-warn-soft px-0.5 text-[9px] font-bold text-warn-ink">L</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Why card */}
      <div className="landing-float absolute -bottom-28 -left-4 hidden w-56 rounded-xl border border-line bg-surface p-3 shadow-pop sm:block lg:-left-12">
        <div className="text-[11px] font-semibold text-ink">Why SICU-03 is flagged</div>
        <ul className="mt-2 space-y-1.5">
          {[
            { k: "Time in ICU 74 h", w: 90 },
            { k: "WBC 30.5, up 4.2", w: 34 },
            { k: "Respiratory rate 30", w: 25 },
          ].map((d) => (
            <li key={d.k}>
              <div className="flex items-center justify-between text-[10px] text-muted">
                <span>{d.k}</span>
                <ArrowUpRight className="size-3 text-alert" />
              </div>
              <div className="mt-0.5 h-1 rounded-full bg-subtle">
                <div className="h-full rounded-full bg-alert" style={{ width: `${d.w}%` }} />
              </div>
            </li>
          ))}
        </ul>
      </div>

      {/* Patient card */}
      <div className="landing-float-delayed absolute -right-3 -bottom-32 hidden w-60 rounded-xl border border-line bg-surface p-3 shadow-pop sm:block lg:-right-10">
        <div className="flex items-center gap-1.5 text-[11px] font-semibold text-ink">
          <HeartPulse className="size-3.5 text-ok" /> Patient view
        </div>
        <div className="mt-2 flex items-start gap-2 rounded-lg bg-alert-soft p-2">
          <BellRing className="mt-0.5 size-3.5 shrink-0 text-alert" />
          <div className="text-[10px] leading-snug text-ink">
            Your care team has been alerted.
            <span className="mt-0.5 flex items-center gap-1 text-muted">
              <CircleCheck className="size-3 text-ok" /> Reviewed by Dr. Demo
            </span>
          </div>
        </div>
        <div className="mt-2.5 flex items-center gap-3">
          <div className="grid grid-cols-10 gap-[2px]">
            {Array.from({ length: 100 }, (_, i) => (
              <span key={i} className={clsx("size-[5px] rounded-full", i < 13 ? "bg-alert" : "bg-line")} />
            ))}
          </div>
          <div>
            <div className="text-xl font-semibold text-ink tnum">
              13 <span className="text-xs font-medium text-muted">in 100</span>
            </div>
            <div className="text-[10px] leading-tight text-muted">chance of sepsis in the next ~6 h</div>
          </div>
        </div>
      </div>
    </div>
  );
}

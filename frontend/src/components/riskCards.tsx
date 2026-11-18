import clsx from "clsx";
import { BellRing, ClipboardCheck, TriangleAlert } from "lucide-react";
import { pct, TIER_LABEL } from "../lib/format";
import type { Meta, PatientSummary } from "../lib/types";
import { AgreementBadge, RiskBar, RiskDelta, tierText } from "./clinical";
import { Badge, Card, CardHeader, Tooltip } from "./ui";

export function RiskSummaryCard({
  s,
  meta,
  viewing = false,
  context = "stay",
}: {
  s: PatientSummary;
  meta?: Meta;
  viewing?: boolean;
  context?: "stay" | "assessment";
}) {
  const precision = meta?.pool.alert_precision;
  const inTen = precision != null ? Math.max(1, Math.round(precision * 10)) : null;
  return (
    <Card
      className={clsx(
        "overflow-hidden",
        s.tier === "alert" && "border-alert-line",
        s.tier === "elevated" && "border-warn-line",
      )}
    >
      <div
        className={clsx(
          "flex items-center justify-between gap-2 px-4 py-2 text-[13px] font-semibold",
          s.tier === "alert" && "bg-alert text-white",
          s.tier === "elevated" && "bg-warn-soft text-warn-ink",
          s.tier === "low" && "bg-subtle text-muted",
        )}
      >
        <span className="inline-flex items-center gap-1.5">
          {s.tier === "alert" ? <BellRing className="size-4" /> : s.tier === "elevated" ? <TriangleAlert className="size-4" /> : null}
          {TIER_LABEL[s.tier]}
          {s.alert && context === "stay" && ` · ${s.alert_hours} h`}
        </span>
        {viewing && <span className="text-xs font-medium opacity-80">historical view</span>}
      </div>
      <div className="p-4">
        <div className="flex items-end justify-between gap-3">
          <div>
            <div className={clsx("text-4xl font-semibold tracking-tight tnum", tierText[s.tier])}>{pct(s.risk)}</div>
            <div className="mt-0.5 text-xs text-muted">calibrated risk {context === "stay" ? "this hour" : "now"}</div>
          </div>
          {s.relative_risk !== null && (
            <div className="text-right">
              <div className="text-lg font-semibold text-ink tnum">{s.relative_risk.toFixed(1)}×</div>
              <div className="text-xs text-muted">ICU baseline</div>
            </div>
          )}
        </div>
        <div className="mt-3">
          <RiskBar risk={s.risk} tier={s.tier} cutoff={meta?.elevated_cutoff ?? 0.08} />
          <div className="mt-1 flex justify-between text-[10px] text-faint tnum">
            <span>0%</span>
            <span>40%</span>
          </div>
        </div>
        <dl className="mt-3 grid grid-cols-3 gap-2 border-t border-line pt-3 text-xs">
          <div>
            <dt className="text-faint">6 h change</dt>
            <dd className="mt-0.5">
              {context === "assessment" && s.record_hours < 7 ? (
                <span className="text-faint">needs 6 h of data</span>
              ) : (
                <RiskDelta now={s.risk} before={s.risk_6h_ago} />
              )}
            </dd>
          </div>
          <div>
            <dt className="text-faint">{context === "stay" ? "24 h peak" : "Peak in window"}</dt>
            <dd className="mt-0.5 font-medium text-ink tnum">{pct(s.risk_peak_24h)}</dd>
          </div>
          <div>
            <Tooltip
              content={`The alert fires when the uncalibrated ensemble score is ≥ ${meta?.threshold ?? 0.75} for 2 consecutive hours (or ≥ 0.90 once).`}
            >
              <dt className="cursor-help text-faint underline decoration-dotted underline-offset-2">Alert score</dt>
            </Tooltip>
            <dd className="mt-0.5 font-medium text-ink tnum">
              {s.raw_score.toFixed(2)} <span className="font-normal text-faint">/ {meta?.threshold ?? 0.75}</span>
            </dd>
          </div>
        </dl>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <AgreementBadge level={s.agreement.level} members={s.agreement.members} />
        </div>
        <p className="mt-3 rounded-lg bg-subtle px-3 py-2 text-xs leading-5 text-muted">
          {s.tier === "alert" ? (
            <>
              The precision-tuned alert has fired.
              {inTen !== null && ` On unseen stays, about ${inTen} in 10 alert hours fell within 6 h of Sepsis-3 onset.`} Review
              the patient; the score is a prompt, not a diagnosis.
            </>
          ) : s.tier === "elevated" ? (
            <>Risk is above the ICU baseline but below the alert threshold. Consider a closer look at trends and pending labs.</>
          ) : (
            <>
              <strong className="font-semibold text-ink">Not elevated — this does not rule out sepsis.</strong> The model misses most
              cases; rely on clinical assessment.
            </>
          )}
        </p>
      </div>
    </Card>
  );
}

const SIRS = ["HR > 90", "RR > 20", "Temp > 38 or < 36 °C", "WBC > 12 or < 4"];
const QSOFA = ["RR ≥ 22", "SBP ≤ 100"];

export function CriteriaCard({ s }: { s: PatientSummary }) {
  const Row = ({ label, met }: { label: string; met: boolean }) => (
    <li className="flex items-center justify-between py-1 text-[13px]">
      <span className={met ? "text-ink" : "text-faint"}>{label}</span>
      {met ? <Badge tone="warn">Met</Badge> : <span className="text-xs text-faint">—</span>}
    </li>
  );
  return (
    <Card>
      <CardHeader
        icon={<ClipboardCheck className="size-4" />}
        title="Bedside criteria"
        subtitle="From charted values; computed by rules, not the model"
      />
      <div className="grid grid-cols-1 gap-4 px-4 pb-4 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
        <div>
          <div className="mb-1 text-xs font-semibold text-muted">
            SIRS <span className="text-faint tnum">{s.sirs.length}/4</span>
          </div>
          <ul className="divide-y divide-line">
            {SIRS.map((c) => (
              <Row key={c} label={c} met={s.sirs.includes(c)} />
            ))}
          </ul>
        </div>
        <div>
          <div className="mb-1 text-xs font-semibold text-muted">
            qSOFA <span className="text-faint tnum">{s.qsofa.length}/2</span>
          </div>
          <ul className="divide-y divide-line">
            {QSOFA.map((c) => (
              <Row key={c} label={c} met={s.qsofa.includes(c)} />
            ))}
            <li className="py-1 text-[11px] text-faint">Mental status (GCS) is not in this model.</li>
          </ul>
        </div>
      </div>
    </Card>
  );
}

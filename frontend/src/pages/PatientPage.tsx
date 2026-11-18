import clsx from "clsx";
import {
  Activity,
  ArrowLeft,
  BellRing,
  BrainCircuit,
  CircleCheck,
  ClipboardCheck,
  FlaskConical,
  History,
  Microscope,
  Stethoscope,
  TriangleAlert,
  Undo2,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AgreementBadge, RiskBar, RiskDelta, tierText } from "../components/clinical";
import { Badge, Button, Card, CardHeader, EmptyState, Kbd, Skeleton, Tooltip } from "../components/ui";
import { useMeta, usePatient } from "../lib/api";
import { icuDay, pct, sexLabel, TIER_LABEL } from "../lib/format";
import { usePrefs } from "../lib/prefs";
import { ACK_REASONS, ackFor, useAcks } from "../lib/store";
import type { Meta, PatientSummary } from "../lib/types";
import { AckDialog } from "./patient/AckDialog";
import { BundlePanel } from "./patient/BundlePanel";
import { DriversPanel } from "./patient/DriversPanel";
import { LabsPanel } from "./patient/LabsPanel";
import { RiskTimeline } from "./patient/RiskTimeline";
import { VitalsPanel } from "./patient/VitalsPanel";

const SELECTION_TEXT: Record<string, string> = {
  consecutive_septic: "One of the first 7 eligible septic stays in the unseen pool (file order).",
  consecutive_non_septic: "One of the first 9 eligible non-septic stays in the unseen pool (file order).",
  alert_example: "Included because the model alerted on this stay, to demonstrate the alert workflow.",
};

export function PatientPage() {
  const { pid } = useParams();
  const { offset, research } = usePrefs();
  const [viewIndex, setViewIndex] = useState<number | null>(null);
  const { data, isLoading, isError, error, isFetching } = usePatient(pid, offset, viewIndex);
  const meta = useMeta();
  const [acks, setAcks] = useAcks();
  const [ackOpen, setAckOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => setViewIndex(null), [offset, pid]);

  const s = data?.summary;
  // Acknowledgement always refers to the alert episode active *now*, not the hour being viewed.
  const nowRow = data?.series[data.series.length - 1];
  const nowAlertSince = (() => {
    if (!data || !nowRow?.alert) return null;
    let j = data.series.length - 1;
    while (j > 0 && data.series[j - 1].alert) j--;
    return data.series[j].icu_hour;
  })();
  const ack = s ? ackFor(acks, s.pid, nowAlertSince) : undefined;

  const openAck = useCallback(() => {
    if (nowAlertSince !== null && !ack) setAckOpen(true);
  }, [nowAlertSince, ack]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "Escape") navigate("/");
      if (e.key === "a" || e.key === "A") openAck();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, openAck]);

  if (isError) {
    return (
      <EmptyState
        icon={<TriangleAlert className="size-5" />}
        title="Could not load this patient"
        action={
          <Link to="/" className="text-sm font-medium text-brand hover:underline">
            Back to ward
          </Link>
        }
      >
        {String((error as Error)?.message ?? error)}
      </EmptyState>
    );
  }

  if (isLoading || !data || !s) return <PatientSkeleton />;

  const viewing = viewIndex !== null && viewIndex < s.now_index;
  const viewIcuHour = s.icu_hour;
  const nowIcuHour = nowRow?.icu_hour ?? s.icu_hour;

  return (
    <div className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6">
      {/* Header */}
      <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <Link to="/" className="mb-2 inline-flex items-center gap-1 text-[13px] font-medium text-muted hover:text-ink">
            <ArrowLeft className="size-3.5" /> Ward overview <Kbd>Esc</Kbd>
          </Link>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <h1 className="text-2xl font-semibold tracking-tight text-ink">{s.bed}</h1>
            <span className="text-lg text-muted">{s.pid}</span>
            {Boolean(nowRow?.alert) && !ack && (
              <Badge tone="alert" icon={<BellRing className="size-3" />} className="pulse-alert">
                Alert needs review
              </Badge>
            )}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted">
            <span>
              {s.age} y · {sexLabel(s.sex)}
            </span>
            <span className="text-line-strong">|</span>
            <span>{s.unit === "MICU" ? "Medical ICU" : "Surgical ICU"}</span>
            <span className="text-line-strong">|</span>
            <span className="tnum">ICU {icuDay(nowIcuHour)}</span>
            {s.hosp_to_icu_hours > 0 && (
              <>
                <span className="text-line-strong">|</span>
                <span className="tnum">Admitted to ICU {Math.round(s.hosp_to_icu_hours)} h after hospital arrival</span>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {nowRow?.alert ? (
            ack ? (
              <div className="flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-1.5 shadow-card">
                <CircleCheck className="size-4 text-ok" />
                <div className="text-[13px] leading-tight">
                  <div className="font-medium text-ink">Reviewed</div>
                  <div className="max-w-56 truncate text-xs text-muted">
                    {ACK_REASONS.find((r) => r.id === ack.reason)?.label}
                  </div>
                </div>
                <button
                  onClick={() =>
                    setAcks((prev) => {
                      const next = { ...prev };
                      delete next[s.pid];
                      return next;
                    })
                  }
                  className="ml-1 text-faint hover:text-ink"
                  aria-label="Undo review"
                  title="Undo review"
                >
                  <Undo2 className="size-4" />
                </button>
              </div>
            ) : (
              <Button variant="danger" icon={<BellRing className="size-4" />} onClick={openAck}>
                Review alert <span className="ml-1 rounded bg-white/20 px-1 text-[11px]">A</span>
              </Button>
            )
          ) : null}
        </div>
      </div>

      {/* Time travel bar */}
      <div
        className={clsx(
          "mb-5 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border px-4 py-2.5",
          viewing ? "border-brand bg-brand-soft" : "border-line bg-surface shadow-card",
        )}
      >
        <span className={clsx("inline-flex items-center gap-2 text-sm font-medium", viewing ? "text-brand-ink" : "text-ink")}>
          <History className="size-4" />
          {viewing ? (
            <>
              Viewing chart as of ICU hour {viewIcuHour}
              <span className="font-normal">({nowIcuHour - viewIcuHour} h before latest)</span>
            </>
          ) : (
            <>Latest data · ICU hour {nowIcuHour}</>
          )}
        </span>
        <input
          type="range"
          min={0}
          max={s.now_index}
          value={viewIndex ?? s.now_index}
          onChange={(e) => {
            const v = Number(e.target.value);
            setViewIndex(v >= s.now_index ? null : v);
          }}
          aria-label="Scrub through the ICU stay"
          className="h-1.5 min-w-40 flex-1 cursor-pointer accent-[var(--brand)]"
        />
        {viewing ? (
          <Button size="sm" variant="primary" onClick={() => setViewIndex(null)}>
            Return to latest
          </Button>
        ) : (
          <span className="text-xs text-faint">Drag, or click any chart, to look back</span>
        )}
        {isFetching && <span className="text-xs text-faint">updating…</span>}
        {s.record_ended && (
          <Badge tone="neutral">Record ends at ICU hour {nowIcuHour}</Badge>
        )}
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {/* Right column first on mobile: the decision-relevant summary */}
        <div className="space-y-5 lg:order-2">
          <RiskSummaryCard s={s} meta={meta.data} viewing={viewing} />
          <Card>
            <CardHeader
              icon={<BrainCircuit className="size-4" />}
              title="What is driving this score"
              subtitle={`ICU hour ${viewIcuHour} · click a row for model inputs`}
            />
            <DriversPanel drivers={data.drivers} measurements={data.measurements} />
          </Card>
          <Card>
            <CardHeader
              icon={<ClipboardCheck className="size-4" />}
              title="Hour-1 bundle"
              subtitle="Surviving Sepsis Campaign 2021"
            />
            <BundlePanel pid={s.pid} items={data.bundle} />
          </Card>
          <CriteriaCard s={s} />
          {research && <ResearchCard s={s} meta={meta.data} />}
        </div>

        <div className="space-y-5 lg:order-1 lg:col-span-2">
          <Card>
            <CardHeader
              icon={<Activity className="size-4" />}
              title="Sepsis risk over the stay"
              subtitle="Calibrated hourly risk · red bands mark model alerts · dashed line is the elevated band"
            />
            <div className="px-2 pb-3">
              <RiskTimeline
                series={data.series}
                viewIcuHour={viewIcuHour}
                elevatedCutoff={meta.data?.elevated_cutoff ?? 0.08}
                onPick={(i) => setViewIndex(i >= s.now_index ? null : i)}
                research={
                  research ? { label: s.research.sepsis_label_icu_hour, onset: s.research.sepsis_onset_icu_hour } : null
                }
              />
            </div>
          </Card>
          <Card>
            <CardHeader
              icon={<Stethoscope className="size-4" />}
              title="Vital signs"
              subtitle="Shaded band = adult reference range · values shown as of the selected hour"
            />
            <div className="pb-2">
              <VitalsPanel
                series={data.series}
                viewIcuHour={viewIcuHour}
                onPick={(i) => setViewIndex(i >= s.now_index ? null : i)}
              />
            </div>
          </Card>
          <Card>
            <CardHeader
              icon={<Microscope className="size-4" />}
              title="Laboratory results"
              subtitle={`As of ICU hour ${viewIcuHour}`}
            />
            <LabsPanel labs={data.labs} />
          </Card>
        </div>
      </div>

      <AckDialog
        open={ackOpen}
        onClose={() => setAckOpen(false)}
        bed={s.bed}
        since={nowAlertSince}
        onConfirm={(reason, note) => {
          if (nowAlertSince === null) return;
          setAcks((prev) => ({
            ...prev,
            [s.pid]: { reason, note, at: new Date().toISOString(), alertSince: nowAlertSince },
          }));
          setAckOpen(false);
        }}
      />
    </div>
  );
}

function RiskSummaryCard({ s, meta, viewing }: { s: PatientSummary; meta?: Meta; viewing: boolean }) {
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
          {s.alert && ` · ${s.alert_hours} h`}
        </span>
        {viewing && <span className="text-xs font-medium opacity-80">historical view</span>}
      </div>
      <div className="p-4">
        <div className="flex items-end justify-between gap-3">
          <div>
            <div className={clsx("text-4xl font-semibold tracking-tight tnum", tierText[s.tier])}>{pct(s.risk)}</div>
            <div className="mt-0.5 text-xs text-muted">calibrated risk this hour</div>
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
              <RiskDelta now={s.risk} before={s.risk_6h_ago} />
            </dd>
          </div>
          <div>
            <dt className="text-faint">24 h peak</dt>
            <dd className="mt-0.5 font-medium text-ink tnum">{pct(s.risk_peak_24h)}</dd>
          </div>
          <div>
            <Tooltip content={`The alert fires when the uncalibrated ensemble score is ≥ ${meta?.threshold ?? 0.75} for 2 consecutive hours (or ≥ 0.90 once).`}>
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

function CriteriaCard({ s }: { s: PatientSummary }) {
  const SIRS = ["HR > 90", "RR > 20", "Temp > 38 or < 36 °C", "WBC > 12 or < 4"];
  const QSOFA = ["RR ≥ 22", "SBP ≤ 100"];
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
            <li className="py-1 text-[11px] text-faint">Mental status (GCS) is not in this dataset.</li>
          </ul>
        </div>
      </div>
    </Card>
  );
}

function ResearchCard({ s, meta }: { s: PatientSummary; meta?: Meta }) {
  const r = s.research;
  return (
    <Card className="border-dashed">
      <CardHeader
        icon={<FlaskConical className="size-4" />}
        title="Retrospective labels"
        subtitle="Research view · not available at the bedside"
      />
      <dl className="space-y-2 px-4 pb-4 text-[13px]">
        <div className="flex justify-between gap-3">
          <dt className="text-muted">Stay met Sepsis-3</dt>
          <dd className="font-medium text-ink">{r.septic_stay ? "Yes" : "No"}</dd>
        </div>
        {r.septic_stay && (
          <>
            <div className="flex justify-between gap-3">
              <dt className="text-muted">Onset (t_sepsis)</dt>
              <dd className="font-medium text-ink tnum">ICU hour {r.sepsis_onset_icu_hour}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-muted">Positive label from</dt>
              <dd className="font-medium text-ink tnum">ICU hour {r.sepsis_label_icu_hour}</dd>
            </div>
          </>
        )}
        <div className="flex justify-between gap-3">
          <dt className="text-muted">Alert score vs threshold</dt>
          <dd className="font-medium text-ink tnum">
            {s.raw_score.toFixed(3)} / {meta?.threshold ?? 0.75}
          </dd>
        </div>
        <div className="border-t border-line pt-2 text-xs text-muted">{SELECTION_TEXT[s.selection_reason]}</div>
      </dl>
    </Card>
  );
}

function PatientSkeleton() {
  return (
    <div className="mx-auto max-w-[1440px] space-y-5 px-4 py-5 sm:px-6">
      <Skeleton className="h-16 w-80" />
      <Skeleton className="h-11" />
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          <Skeleton className="h-72" />
          <Skeleton className="h-96" />
        </div>
        <div className="space-y-5">
          <Skeleton className="h-64" />
          <Skeleton className="h-72" />
        </div>
      </div>
    </div>
  );
}

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
import { CriteriaCard, RiskSummaryCard } from "../components/riskCards";
import { Badge, Button, Card, CardHeader, EmptyState, Kbd, Skeleton } from "../components/ui";
import { useAck, useMeta, usePatient, useReplay } from "../lib/api";
import { icuDay, sexLabel } from "../lib/format";
import { usePrefs } from "../lib/prefs";
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
  const { research } = usePrefs();
  const replay = useReplay();
  const offset = replay.data?.offset;
  const [viewIndex, setViewIndex] = useState<number | null>(null);
  const { data, isLoading, isError, error, isFetching } = usePatient(pid, offset, viewIndex);
  const meta = useMeta();
  const { review, undo } = useAck(pid ?? "");
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
  const ack = data?.now_ack ?? undefined;

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
                  <div className="max-w-64 truncate text-xs text-muted" title={ack.note || undefined}>
                    {ack.by} · {ack.reason_label}
                  </div>
                </div>
                <button
                  onClick={() => undo.mutate()}
                  disabled={undo.isPending}
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
        pending={review.isPending}
        error={review.error ? String((review.error as Error).message) : null}
        onConfirm={(reason, note) => {
          if (nowAlertSince === null) return;
          review.mutate({ reason, note }, { onSuccess: () => setAckOpen(false) });
        }}
      />
    </div>
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

import { BookOpenCheck, CircleAlert, CircleCheck, CircleX, Database, Gauge, ListChecks, Scale } from "lucide-react";
import type { ReactNode } from "react";
import { Card, CardHeader, Skeleton } from "../components/ui";
import { useMeta } from "../lib/api";
import { pct } from "../lib/format";

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-line p-3">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 text-xl font-semibold text-ink tnum">{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-faint">{hint}</div>}
    </div>
  );
}

function Item({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <li className="flex gap-2.5 text-sm leading-6 text-ink">
      <span className="mt-1 shrink-0">{icon}</span>
      <span>{children}</span>
    </li>
  );
}

export function ModelFactsPage() {
  const { data: m, isLoading } = useMeta();
  if (isLoading || !m) {
    return (
      <div className="mx-auto max-w-5xl space-y-4 px-4 py-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-64" />
      </div>
    );
  }
  const iv = m.internal_validation.validation_metrics;
  const p = m.pool;

  return (
    <div className="mx-auto max-w-5xl space-y-5 px-4 py-6 sm:px-6">
      <div>
        <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-ink">
          <BookOpenCheck className="size-5 text-brand" /> Model facts
        </h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          What the score means, how well it performs, and where it fails. Read this before acting on a number.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <Card>
          <CardHeader icon={<CircleCheck className="size-4" />} title="Intended use" />
          <ul className="space-y-2 px-4 pb-4">
            <Item icon={<CircleCheck className="size-4 text-ok" />}>
              Research and teaching: exploring how an hourly sepsis risk score behaves on real ICU stays.
            </Item>
            <Item icon={<CircleCheck className="size-4 text-ok" />}>
              Prompting a second look at adult ICU patients whose score rises, alongside normal assessment.
            </Item>
          </ul>
        </Card>
        <Card>
          <CardHeader icon={<CircleX className="size-4" />} title="Not intended for" />
          <ul className="space-y-2 px-4 pb-4">
            <Item icon={<CircleX className="size-4 text-alert" />}>
              Diagnosing sepsis, or ruling it out. <strong>Most septic patients never trigger an alert.</strong>
            </Item>
            <Item icon={<CircleX className="size-4 text-alert" />}>
              Clinical care of real patients. Not validated prospectively and not a regulated medical device.
            </Item>
          </ul>
        </Card>
      </div>

      <Card>
        <CardHeader
          icon={<Gauge className="size-4" />}
          title="How to read the numbers"
          subtitle="Three different things appear in the interface"
        />
        <dl className="grid gap-4 px-4 pb-4 md:grid-cols-3">
          <div>
            <dt className="text-sm font-semibold text-ink">Risk (%)</dt>
            <dd className="mt-1 text-[13px] leading-5 text-muted">
              Isotonic-calibrated probability that this hour falls within 6 h before Sepsis-3 onset (or after it). ICU
              baseline is {pct(m.base_rate, 1)}. The calibrator's maximum is about 38%, so scores move in steps.
            </dd>
          </div>
          <div>
            <dt className="text-sm font-semibold text-warn-ink">Elevated band (≥ {pct(m.elevated_cutoff)})</dt>
            <dd className="mt-1 text-[13px] leading-5 text-muted">
              A display band at roughly 3.5× baseline to help triage the list. It is a design choice, not a validated cut-off.
            </dd>
          </div>
          <div>
            <dt className="text-sm font-semibold text-alert-ink">Model alert</dt>
            <dd className="mt-1 text-[13px] leading-5 text-muted">
              Fires when the uncalibrated ensemble score is ≥ {m.threshold} for 2 consecutive hours (or ≥ 0.90 once). The
              threshold was tuned for precision, so alerts are rare.
            </dd>
          </div>
        </dl>
      </Card>

      <Card>
        <CardHeader
          icon={<Scale className="size-4" />}
          title="Performance"
          subtitle="Hour-level metrics against PhysioNet Sepsis-3 labels"
        />
        <div className="grid gap-5 px-4 pb-4 lg:grid-cols-2">
          <section>
            <h3 className="text-sm font-semibold text-ink">Unseen replay pool</h3>
            <p className="mb-3 text-xs text-muted">
              {p.patients.toLocaleString()} stays ({p.septic_patients} septic) never used in training · computed by this
              server
            </p>
            <div className="grid grid-cols-2 gap-2">
              <Metric label="ROC-AUC" value={p.hour_roc_auc.toFixed(3)} />
              <Metric label="PR-AUC" value={p.hour_pr_auc.toFixed(3)} hint={`baseline ${p.positive_hour_rate.toFixed(3)}`} />
              <Metric
                label="Alert precision"
                value={p.alert_precision != null ? pct(p.alert_precision) : "—"}
                hint={`${p.alert_hours} alert hours`}
              />
              <Metric label="Alert recall (hours)" value={pct(p.alert_recall, 1)} />
              <Metric
                label="Septic stays alerted"
                value={`${p.septic_patients_alerted} / ${p.septic_patients}`}
                hint={pct(p.septic_patients_alerted / Math.max(1, p.septic_patients))}
              />
              <Metric
                label="Non-septic stays alerted"
                value={`${p.non_septic_patients_alerted} / ${p.patients - p.septic_patients}`}
              />
            </div>
          </section>
          {iv && (
            <section>
              <h3 className="text-sm font-semibold text-ink">Internal validation (from training run)</h3>
              <p className="mb-3 text-xs text-muted">
                20% patient-level split of {m.internal_validation.total_patients?.toLocaleString()} stays ·
                models/precision_metrics.json
              </p>
              <div className="grid grid-cols-2 gap-2">
                <Metric label="ROC-AUC" value={iv.roc_auc.toFixed(3)} />
                <Metric label="PR-AUC" value={iv.pr_auc.toFixed(3)} />
                <Metric label="Precision @ alert" value={pct(iv.filtered_precision, 1)} />
                <Metric label="Recall @ alert" value={pct(iv.filtered_recall, 1)} />
                <Metric label="Specificity" value={pct(iv.specificity, 2)} />
                <Metric label="PhysioNet utility" value={iv.physionet_utility.toFixed(2)} hint="0 = never alert; 1 = perfect" />
              </div>
            </section>
          )}
        </div>
        <p className="mx-4 mb-4 rounded-lg bg-warn-soft px-3 py-2 text-xs leading-5 text-warn-ink">
          Discrimination on the unseen pool (ROC-AUC {p.hour_roc_auc.toFixed(2)}) is lower than the internal figure
          {iv ? ` (${iv.roc_auc.toFixed(2)})` : ""}. Expect the score to be weaker than the training report suggests.
        </p>
      </Card>

      <Card>
        <CardHeader icon={<ListChecks className="size-4" />} title="Known limitations" />
        <ul className="space-y-2 px-4 pb-4">
          {[
            "Single dataset (PhysioNet 2019, two US hospital systems). No external or prospective validation.",
            "Hour-level labels: the 6 h pre-onset window is labelled positive, so alerts long before onset count as false positives.",
            "The saved alert threshold was tuned on the uncalibrated score; this interface applies it on that scale. Applying it to the calibrated score, as the training script did, can never fire.",
            "The repository's persistence filter looks one hour ahead. This interface uses a causal version (current and previous hour only).",
            "\"Lab-ordering\" features are computed after forward-fill, so they encode whether a lab has ever been charted, not when it was ordered.",
            "Conformal prediction is not shown: the saved quantile (0.25) produces empty prediction sets for most scores.",
            "No mental-status (GCS), medication, culture or note data. Bundle prompts cannot see orders.",
          ].map((t) => (
            <Item key={t} icon={<CircleAlert className="size-4 text-warn" />}>
              {t}
            </Item>
          ))}
        </ul>
      </Card>

      <Card>
        <CardHeader icon={<Database className="size-4" />} title="About this ward" />
        <div className="space-y-2 px-4 pb-4 text-sm leading-6 text-ink">
          <p>
            The {m.ward_size} beds replay real, de-identified ICU stays from PhysioNet 2019 training set A, files p010000 and
            later (training used the first 1,500 files of each set). Composition:
          </p>
          <ul className="list-disc space-y-1 pl-5 text-[13px] text-muted">
            <li>{m.selection_counts.consecutive_septic} septic stays: the first eligible ones in file order</li>
            <li>{m.selection_counts.consecutive_non_septic} non-septic stays: the first eligible ones in file order</li>
            <li>
              {m.selection_counts.alert_example} stay(s) included <em>because</em> the model alerted, so the alert workflow
              can be demonstrated
            </li>
          </ul>
          <p className="text-xs text-faint">
            Data: Reyna MA et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology
            Challenge 2019. Crit Care Med 2020;48(2):210–217. Open Data Commons Attribution License v1.0.
          </p>
        </div>
      </Card>
    </div>
  );
}

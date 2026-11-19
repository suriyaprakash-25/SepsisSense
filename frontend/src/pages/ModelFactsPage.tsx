import { BookOpenCheck, CircleAlert, CircleCheck, CircleX, Cpu, Database, Gauge, ListChecks, Scale } from "lucide-react";
import type { ReactNode } from "react";
import { Card, CardHeader, Skeleton } from "../components/ui";
import { useMeta } from "../lib/api";
import { pct } from "../lib/format";
import type { EvalMetrics } from "../lib/types";

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

const HISTORY = [
  { v: "v1", desc: "3,000-patient subset", auc: "0.685", septic: "9 / 138" },
  { v: "v2", desc: "all patients, 67 features", auc: "0.833", septic: "368 / 587" },
];

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
  const iv = m.internal_validation;
  const test = iv.validation_metrics;
  const p = m.pool;
  const bySet = iv.test_by_set;
  const split = iv.split;

  const setRows: { k: string; f: (x: EvalMetrics) => string }[] = [
    { k: "Patients (septic)", f: (x) => `${x.patients.toLocaleString()} (${x.septic_patients})` },
    { k: "ROC-AUC", f: (x) => x.roc_auc.toFixed(3) },
    { k: "PR-AUC", f: (x) => x.pr_auc.toFixed(3) },
    { k: "PhysioNet utility", f: (x) => x.physionet_utility.toFixed(3) },
    { k: "Alert precision", f: (x) => (x.alert_precision == null ? "—" : pct(x.alert_precision, 1)) },
    { k: "Septic patients alerted", f: (x) => `${x.septic_patients_alerted} / ${x.septic_patients}` },
  ];

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
              Diagnosing sepsis, or ruling it out. <strong>Some septic patients never trigger an alert.</strong>
            </Item>
            <Item icon={<CircleX className="size-4 text-alert" />}>
              Clinical care of real patients. Not validated prospectively and not a regulated medical device.
            </Item>
          </ul>
        </Card>
      </div>

      <Card>
        <CardHeader icon={<Gauge className="size-4" />} title="How to read the numbers" subtitle="Three different things appear in the interface" />
        <dl className="grid gap-4 px-4 pb-4 md:grid-cols-3">
          <div>
            <dt className="text-sm font-semibold text-ink">Risk (%)</dt>
            <dd className="mt-1 text-[13px] leading-5 text-muted">
              Isotonic-calibrated probability that this hour falls within 6 h before Sepsis-3 onset (or after it). ICU
              baseline is {pct(m.base_rate, 1)}. Calibrated scores move in steps and top out near{" "}
              {pct(iv.calibrator_max ?? m.risk_scale_max)}.
            </dd>
          </div>
          <div>
            <dt className="text-sm font-semibold text-warn-ink">Elevated band (≥ {pct(m.elevated_cutoff, 1)})</dt>
            <dd className="mt-1 text-[13px] leading-5 text-muted">
              A display band (2.5× the baseline, below the alert threshold) to help triage the list. A design choice, not
              a validated cut-off.
            </dd>
          </div>
          <div>
            <dt className="text-sm font-semibold text-alert-ink">Model alert (≥ {pct(m.threshold, 1)})</dt>
            <dd className="mt-1 text-[13px] leading-5 text-muted">
              Fires when calibrated risk is at or above the threshold for 2 consecutive hours. The threshold maximises the
              official PhysioNet 2019 utility on a separate calibration split, which rewards early detection.
            </dd>
          </div>
        </dl>
      </Card>

      <Card>
        <CardHeader icon={<Scale className="size-4" />} title="Performance on held-out patients" subtitle="Hour-level metrics against PhysioNet Sepsis-3 labels" />
        <div className="space-y-5 px-4 pb-4">
          <section>
            <h3 className="text-sm font-semibold text-ink">Held-out test split</h3>
            <p className="mb-3 text-xs text-muted">
              {p.patients.toLocaleString()} patients ({p.septic_patients} septic), never used for training, calibration or
              threshold choice · computed by this server
            </p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <Metric label="ROC-AUC" value={p.hour_roc_auc.toFixed(3)} />
              <Metric label="PR-AUC" value={p.hour_pr_auc.toFixed(3)} hint={`baseline ${p.positive_hour_rate.toFixed(3)}`} />
              <Metric label="PhysioNet utility" value={(p.physionet_utility ?? test?.physionet_utility ?? 0).toFixed(3)} hint="0 = never alert · 1 = perfect" />
              <Metric label="Brier score" value={test ? test.brier_score.toFixed(4) : "—"} hint="calibration error" />
              <Metric label="Alert precision" value={p.alert_precision != null ? pct(p.alert_precision, 1) : "—"} hint={`${p.alert_hours.toLocaleString()} alert hours`} />
              <Metric label="Alert recall (hours)" value={pct(p.alert_recall, 1)} />
              <Metric
                label="Septic patients alerted"
                value={`${p.septic_patients_alerted} / ${p.septic_patients}`}
                hint={pct(p.septic_patients_alerted / Math.max(1, p.septic_patients))}
              />
              <Metric
                label="Non-septic patients alerted"
                value={`${p.non_septic_patients_alerted.toLocaleString()} / ${(p.patients - p.septic_patients).toLocaleString()}`}
                hint={pct(p.non_septic_patients_alerted / Math.max(1, p.patients - p.septic_patients))}
              />
            </div>
            {test?.median_hours_alert_before_onset != null && (
              <p className="mt-2 text-xs text-muted">
                For septic patients who were alerted, the first alert came a median of{" "}
                <strong className="text-ink">{test.median_hours_alert_before_onset} h</strong> before Sepsis-3 onset. Many of
                these first alerts fire well before the 12-hour window the PhysioNet utility rewards, so they count as false
                positives there; treat this as "flagged early in the stay", not as reliable early warning.
              </p>
            )}
          </section>

          {bySet && (
            <section>
              <h3 className="text-sm font-semibold text-ink">By hospital system</h3>
              <p className="mb-2 text-xs text-muted">The two PhysioNet sets come from different US hospital systems.</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-y border-line bg-subtle/60 text-[11px] font-semibold tracking-wide text-faint uppercase">
                      <th className="px-3 py-2 text-left">Held-out test</th>
                      <th className="px-3 py-2 text-right">Set A</th>
                      <th className="px-3 py-2 text-right">Set B</th>
                    </tr>
                  </thead>
                  <tbody className="tnum">
                    {setRows.map((r) => (
                      <tr key={r.k} className="border-b border-line last:border-0">
                        <td className="px-3 py-1.5 text-muted">{r.k}</td>
                        <td className="px-3 py-1.5 text-right text-ink">{r.f(bySet.A)}</td>
                        <td className="px-3 py-1.5 text-right text-ink">{r.f(bySet.B)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <p className="rounded-lg bg-subtle px-3 py-2 text-xs leading-5 text-muted">
            Earlier models, held-out ROC-AUC and septic patients alerted:{" "}
            {HISTORY.map((h) => `${h.v} (${h.desc}) ${h.auc}, ${h.septic}`).join(" · ")}. This model uses all{" "}
            {(iv.total_patients ?? 0).toLocaleString()} patients with a 70 / 10 / 20 patient-level split
            {split ? ` (${split.train.patients.toLocaleString()} train · ${split.calibration.patients.toLocaleString()} calibration · ${split.test.patients.toLocaleString()} test)` : ""}.
          </p>
        </div>
      </Card>

      {iv.devices && (
        <Card>
          <CardHeader icon={<Cpu className="size-4" />} title="How it was trained" />
          <ul className="space-y-1.5 px-4 pb-4 text-sm text-ink">
            {iv.recipe === "v3" ? (
              <>
                <li>
                  XGBoost on {iv.devices.xgboost} and LightGBM on {iv.devices.lightgbm}, equal soft-voting weights, isotonic
                  calibration on its own patient split.
                </li>
                <li>
                  {iv.n_features} features (src/features_v3.py): vital-sign trends and variability, measurement counts and
                  recency, partial SOFA and its 24 h rise, SIRS, NEWS, BUN/creatinine. Trees use missing values natively.
                </li>
                <li>
                  Utility-weighted loss: each hour is weighted by how much its prediction changes the PhysioNet utility,
                  so the model focuses on the 12 h before to 3 h after onset. {iv.training_rows_after_mining?.toLocaleString()} training hours.
                </li>
              </>
            ) : (
              <>
                <li>
                  XGBoost on {iv.devices.xgboost}, LightGBM on {iv.devices.lightgbm}
                  {iv.devices.random_forest ? `, random forest on ${iv.devices.random_forest}` : ""}; isotonic calibration.
                </li>
                <li>{iv.training_rows_after_mining?.toLocaleString()} training hours; missing values filled with training medians.</li>
              </>
            )}
          </ul>
        </Card>
      )}

      <Card>
        <CardHeader icon={<ListChecks className="size-4" />} title="Known limitations" />
        <ul className="space-y-2 px-4 pb-4">
          {[
            "Single dataset (PhysioNet 2019, two US hospital systems). No external or prospective validation.",
            "Hour-level labels: the 6 h pre-onset window is labelled positive, so alerts long before onset count as false positives.",
            "The threshold maximises PhysioNet utility, which values catching sepsis over avoiding false alarms. Expect more alerts than a precision-first tool.",
            "The score still depends on time in ICU: longer PhysioNet stays carry more sepsis labels, and the model uses that shortcut.",
            "Conformal prediction is not shown: it has not been re-calibrated for this model.",
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
            The {m.ward_size} beds replay real, de-identified ICU stays from the held-out test split of PhysioNet 2019. None
            of these patients were used to train or tune the model. Composition:
          </p>
          <ul className="list-disc space-y-1 pl-5 text-[13px] text-muted">
            <li>{m.selection_counts.consecutive_septic} septic stays: the first eligible ones in ID order</li>
            <li>{m.selection_counts.consecutive_non_septic} non-septic stays: the first eligible ones in ID order</li>
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

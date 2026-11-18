export type Tier = "alert" | "elevated" | "low";
export type Flag = "high" | "low" | null;

export interface Measurement {
  value: number | null;
  previous?: number | null;
  hours_ago: number | null;
  flag: Flag;
  stale?: boolean;
}

export type VitalKey = "HR" | "MAP" | "SBP" | "DBP" | "Resp" | "Temp" | "O2Sat" | "EtCO2";

export type SelectionReason = "consecutive_septic" | "consecutive_non_septic" | "alert_example";

export interface PatientSummary {
  pid: string;
  bed: string;
  unit: "MICU" | "SICU";
  age: number;
  sex: "M" | "F";
  hour_index: number;
  now_index: number;
  icu_hour: number;
  hosp_to_icu_hours: number;
  record_hours: number;
  record_ended: boolean;
  risk: number;
  risk_6h_ago: number;
  risk_peak_24h: number;
  relative_risk: number | null;
  raw_score: number;
  spark: number[];
  alert: boolean;
  alert_since_icu_hour: number | null;
  alert_hours: number;
  ack: ServerAck | null;
  tier: Tier;
  agreement: { level: "high" | "moderate" | "low"; sd: number; members: Record<string, number> };
  vitals: Record<VitalKey, Measurement>;
  lactate: Measurement;
  sirs: string[];
  qsofa: string[];
  selection_reason: SelectionReason;
  research: {
    sepsis_label_icu_hour: number | null;
    sepsis_onset_icu_hour: number | null;
    septic_stay: boolean;
  };
}

export interface WardResponse {
  offset: number;
  patients: PatientSummary[];
}

export interface SeriesRow {
  i: number;
  icu_hour: number;
  risk: number;
  raw_score: number;
  alert: number;
  [key: string]: number | null;
}

export interface DriverFeature {
  feature: string;
  label: string;
  value: number;
  contribution: number;
}

export interface Driver {
  concept: string;
  label: string;
  primary: string | null;
  contribution: number;
  features: DriverFeature[];
}

export interface LabRow extends Measurement {
  name: string;
  label: string;
  unit: string;
  group: string;
  low: number;
  high: number;
  dp: number;
  count: number;
  history: { i: number; icu_hour: number; value: number }[];
}

export interface BundleItem {
  id: string;
  title: string;
  detail: string;
  context: string;
  priority: "high" | "medium" | "low";
}

export interface MeasurementMeta {
  label: string;
  unit: string;
  low: number;
  high: number;
  dp: number;
}

export interface PatientDetail {
  summary: PatientSummary;
  now_ack: ServerAck | null;
  series: SeriesRow[];
  drivers: Driver[];
  labs: LabRow[];
  bundle: BundleItem[];
  measurements: Record<string, MeasurementMeta>;
}

export interface Meta {
  threshold: number;
  elevated_cutoff: number;
  risk_scale_max: number;
  alert_scale: string;
  base_rate: number;
  max_offset: number;
  n_features: number;
  ward_size: number;
  selection_counts: Record<SelectionReason, number>;
  pool: {
    patients: number;
    hours: number;
    septic_patients: number;
    hour_roc_auc: number;
    hour_pr_auc: number;
    positive_hour_rate: number;
    alert_hours: number;
    alert_precision: number | null;
    alert_recall: number;
    septic_patients_alerted: number;
    non_septic_patients_alerted: number;
    physionet_utility?: number;
  };
  internal_validation: {
    version?: number;
    total_patients?: number;
    total_records?: number;
    sepsis_patients?: number;
    calibrator_max?: number;
    training_rows_after_mining?: number;
    split?: Record<"train" | "calibration" | "test", { patients: number; septic: number }>;
    devices?: { xgboost: string; lightgbm: string; random_forest: string };
    validation_metrics?: EvalMetrics;
    test_by_set?: { A: EvalMetrics; B: EvalMetrics };
  };
}

// ------------------------------------------------------------ assessment

export interface AssessField {
  name: string;
  label: string;
  short: string;
  unit: string;
  kind: "vital" | "lab";
  group: string;
  ref_low: number;
  ref_high: number;
  min: number;
  max: number;
  dp: number;
}

export interface AssessFieldsResponse {
  fields: AssessField[];
  max_hours_back: number;
}

export interface AssessRequest {
  age: number;
  sex: "M" | "F";
  unit: "MICU" | "SICU";
  icu_hours: number;
  hosp_to_icu_hours: number;
  readings: { hours_ago: number; values: Record<string, number | null> }[];
}

export interface AssessSeriesRow {
  i: number;
  hours_ago: number;
  icu_hour: number;
  risk: number;
  raw_score: number;
  alert: number;
  entered: boolean;
}

export interface AssessResponse {
  summary: PatientSummary;
  series: AssessSeriesRow[];
  drivers: Driver[];
  labs: LabRow[];
  bundle: BundleItem[];
  measurements: Record<string, MeasurementMeta>;
  warnings: string[];
}

// ---------------------------------------------------------------- auth

export type Role = "clinician" | "patient";

export interface SessionUser {
  username: string;
  display_name: string;
  role: Role;
  pid: string | null;
  bed?: string | null;
  created_at?: string;
  last_login?: string | null;
}

export interface DemoAccount {
  username: string;
  password: string;
  role: Role;
  display_name: string;
}

export interface ServerAck {
  alert_since_icu_hour: number;
  reason: "workup" | "alternative" | "known" | "comfort";
  reason_label: string;
  note: string;
  by: string;
  by_username: string;
  at: string;
}

export interface AdminUser extends SessionUser {
  bed: string | null;
}

export interface AdminUsersResponse {
  users: AdminUser[];
  patients: { pid: string; bed: string }[];
  me: string;
}

export interface PublicSummary {
  pool: {
    patients: number;
    hours: number;
    septic_patients: number;
    hour_roc_auc: number;
    hour_pr_auc: number;
    alert_precision: number | null;
    alert_recall: number;
    septic_patients_alerted: number;
    non_septic_patients_alerted: number;
  };
  internal: { roc_auc?: number; pr_auc?: number; brier_score?: number };
  n_features: number;
  base_rate: number;
}

export interface EvalMetrics {
  patients: number;
  hours: number;
  septic_patients: number;
  positive_hour_rate: number;
  roc_auc: number;
  pr_auc: number;
  brier_score: number;
  alert_precision: number | null;
  alert_recall: number;
  specificity: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  septic_patients_alerted: number;
  non_septic_patients_alerted: number;
  median_hours_alert_before_onset: number | null;
  physionet_utility: number;
}

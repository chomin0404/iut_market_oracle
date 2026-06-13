// ---- /valuation ----

export interface DCFRequest {
  fcff_series: number[];
  terminal_growth_rate: number;
  wacc: number;
  shares_outstanding: number;
  net_debt: number;
}

export interface SensitivityRow {
  wacc: number;
  growth: number;
  value: number;
}

export interface DCFResponse {
  intrinsic_value: number;
  terminal_value: number;
  pv_fcff: number;
  sensitivity: SensitivityRow[];
}

export interface ReverseDCFRequest {
  market_price: number;
  shares_outstanding: number;
  net_debt: number;
  wacc: number;
  explicit_fcff: number[];
}

export interface ReverseDCFResponse {
  implied_growth_rate: number;
  terminal_value: number;
  pv_explicit: number;
}

// ---- /bayesian ----

export interface PriorSpec {
  distribution: string;
  params: Record<string, number>;
}

export interface Evidence {
  metric: string;
  value: number;
  weight?: number;
}

export interface BayesianUpdateRequest {
  prior: PriorSpec;
  evidence: Evidence[];
  n_samples?: number;
}

export interface PosteriorSummary {
  mean: number;
  std: number;
  credible_interval: [number, number];
  distribution: string;
}

export interface BayesianUpdateResponse {
  posterior: PosteriorSummary;
  bayes_factor: number;
  evidence_weight: number;
}

// ---- /entropy ----

export interface EntropyRequest {
  series: number[];
  window?: number;
  kl_reference?: number[];
  threshold?: number;
  experiment_id?: string;
}

export type AlertType = "entropy_spike" | "kl_divergence" | "entropy_rate";

export interface EntropyAlert {
  experiment_id: string;
  triggered_at: number;
  alert_type: AlertType;
  metric_value: number;
  threshold: number;
  message: string;
}

export interface EntropyReport {
  experiment_id: string;
  entropy_series: number[];
  kl_series: number[];
  entropy_rate_series: number[];
  alerts: EntropyAlert[];
}

// ---- /gnss/twin/run ----

export type RecommendedAction =
  | "nominal"
  | "monitor"
  | "reduce_trust"
  | "switch_source"
  | "ground_immediately";

export interface EpochReport {
  epoch: number;
  authenticity: { genuine: number; spoofed: number };
  integrity: { nominal: number; degraded: number };
  fault_posterior: {
    nominal: number;
    multipath: number;
    hardware_fault: number;
    spoofing: number;
  };
  diagnosis: string;
  confidence: number;
  recommended_action: RecommendedAction;
  action_reason: string;
  entropy_alert: boolean;
  gmm_n_fault: number;
  imm_spoof_weight: number;
  spectral_fiedler_ratio: number;
  ins_chi2_vel: number | null;
  ins_alert: boolean;
  coop_parity_chi2: number;
  coop_parity_alert: boolean;
  osnma_auth_fraction: number;
  structural_fiedler_streak: number;
  structural_alert: boolean;
  auth_p_spoofed: number;
}

export interface TwinRunReport {
  epoch_reports: EpochReport[];
  n_epochs: number;
  n_sats: number;
  dominant_diagnosis: string;
  mean_authenticity_genuine: number;
  mean_integrity_nominal: number;
  alert_epochs: number[];
  spoofing_window: [number, number] | null;
  worst_action: RecommendedAction;
  run_id: string;
  result_path?: string;
}

// ---- /gnss/resilience-sim ----

export interface ResilienceTwinReport {
  p_detection: number;
  p_false_alarm: number;
  auc: number;
  per_class_accuracy: Record<string, number>;
  confusion_matrix: number[][];
  mean_confidence: number;
  n_mc: number;
  n_mc_per_class: Record<string, number>;
  produced_at: string;
}

// ---- ObservationEpoch (input for twin/run) ----

export interface ObservationEpoch {
  epoch: number;
  doppler_residuals: number[];
  elevations_deg?: number[];
  ins_velocity_ms?: number[];
  osnma_auth_per_sat?: boolean[];
}

export interface TwinRunRequest {
  observations: ObservationEpoch[];
  n_sats: number;
  doppler_noise_std: number;
  graph_sigma: number;
  save: boolean;
}

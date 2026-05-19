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

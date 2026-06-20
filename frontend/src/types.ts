// ---- /valuation ----

/** POST /api/v1/valuation/dcf */
export interface DCFRequest {
  initial_fcf: number;
  growth_rate: number;
  discount_rate: number;
  forecast_years?: number;
  terminal_growth_rate?: number;
}

export interface DCFResponse {
  projected_fcfs: number[];
  discounted_fcfs: number[];
  terminal_value: number;
  discounted_terminal_value: number;
  enterprise_value: number;
}

/** POST /api/v1/valuation/dcf/reverse */
export interface ReverseDCFRequest {
  target_enterprise_value: number;
  initial_fcf: number;
  discount_rate: number;
  forecast_years?: number;
  terminal_growth_rate?: number;
}

export interface ReverseDCFResponse {
  implied_growth_rate: number;
}

// ---- /bayesian ----

export interface PriorSpec {
  distribution: string;
  params: Record<string, number>;
  description?: string;
}

export type EvidenceKind = "observation" | "expert_prior" | "market_data" | "backtest";

export interface Evidence {
  source: string;
  kind: EvidenceKind;
  value: number;
  weight?: number;
  tag?: string;
  notes?: string;
}

/** POST /api/v1/bayesian/update — returns PosteriorSummary */
export interface BayesianUpdateRequest {
  prior: PriorSpec;
  evidence: Evidence[];
}

/** Mirrors schemas.PosteriorSummary */
export interface PosteriorSummary {
  mean: number;
  variance: number;
  credible_interval_95: [number, number];
  n_evidence: number;
  updated_at: string;
}

// ---- /entropy ----

/** POST /api/v1/entropy/detect */
export interface EntropyDetectRequest {
  posteriors: PosteriorSummary[];
  prior: PriorSpec;
  experiment_id: string;
  kl_threshold?: number;
  entropy_gradient_threshold?: number;
  rolling_window?: number;
}

export type AlertType = "KL_THRESHOLD" | "ENTROPY_GRADIENT";

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

// ---- /graph (T300) ----

export interface NodeMeta {
  node_id: string;
  label?: string;
  category?: string;
  weight?: number;
}

export interface EdgeMeta {
  source: string;
  target: string;
  strength?: number;
  label?: string;
}

export interface GraphInput {
  nodes: NodeMeta[];
  edges: EdgeMeta[];
}

export interface PortfolioMetrics {
  basis_diversity: number;
  dependency_concentration: number;
  portfolio_score: number;
  node_count: number;
  edge_count: number;
  notes: string;
}

// ---- /yield-twin (T1600) ----

export interface FactorSpec {
  name: string;
  low: number;
  high: number;
}

export interface ExperimentPoint {
  factors: Record<string, number>;
  yield_obs: number | null;
}

export interface YieldTwinRequest {
  factor_specs: FactorSpec[];
  observations?: ExperimentPoint[];
  random_seed?: number;
  n_candidates?: number;
  ei_xi?: number;
}

export interface DOERecommendation {
  factors: Record<string, number>;
  expected_improvement: number;
  d_leverage: number;
  fusion_score: number;
  predicted_yield: number;
  predicted_std: number;
  acquisition_mode: "doe_explore" | "fused" | "ei_exploit";
  n_observations: number;
}

// ---- /strategy (T1700) ----

export interface MoatScore {
  dimension: string;
  score: number;
  weight?: number;
}

export interface BusinessUnit {
  name: string;
  initial_fcf: number;
  growth_rate: number;
  discount_rate: number;
  terminal_growth_rate?: number;
  forecast_years?: number;
  moat_scores?: MoatScore[];
}

export interface MacroEnvironment {
  gdp_growth: number;
  risk_free_rate: number;
  market_risk_premium: number;
  inflation: number;
  tam: number;
}

export interface BLView {
  assets: Record<string, number>;
  expected_return: number;
  uncertainty: number;
}

export interface CausalEdge {
  cause: string;
  effect: string;
  coefficient: number;
}

export interface StrategyRunRequest {
  business_units: BusinessUnit[];
  macro: MacroEnvironment;
  views?: BLView[];
  causal_edges?: CausalEdge[];
  target_ev?: number;
  net_debt?: number;
  current_market_share?: number;
  current_fcf_margin?: number;
}

export interface SOTPSegment {
  unit_name: string;
  enterprise_value: number;
  moat_adjusted_wacc: number;
  moat_adjusted_terminal_growth: number;
  moat_composite_score: number;
}

export interface BLResult {
  equilibrium_returns: Record<string, number>;
  posterior_returns: Record<string, number>;
  posterior_std: Record<string, number>;
  market_weights: Record<string, number>;
}

export interface ViabilityCondition {
  metric: string;
  minimum_required: number;
  current_estimate: number;
  gap: number;
  is_met: boolean;
  explanation: string;
}

export interface StrategyTwinReport {
  sotp_segments: SOTPSegment[];
  sotp_total_ev: number;
  bl_result: BLResult;
  viability_conditions: ViabilityCondition[];
  implied_growth_rate: number | null;
  causal_effects: Array<{ cause: string; effect: string; total_effect: number; n_paths: number }>;
  verdict: string;
  produced_at: string;
}


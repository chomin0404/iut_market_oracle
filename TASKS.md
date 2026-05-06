# TASKS.md

# Status Legend
- [ ] not started
- [~] in progress
- [x] done
- [!] blocked

## T000 Repository Skeleton
- [x] Create directories:
  - `src/`
  - `tests/`
  - `configs/`
  - `notes/`
  - `papers/`
  - `data/raw/`
  - `data/processed/`
  - `experiments/`
  - `reports/`
  - `notebooks/`
- [x] Add `.gitignore`
- [x] Add `README.md`

Acceptance:
- repository structure exists
- raw vs processed data separation is visible

## T001 Claude Code Foundation
- [x] Add `CLAUDE.md`
- [x] Add `.claude/settings.json`
- [x] Add local-only settings pattern in `.gitignore`

Acceptance:
- project rules exist
- settings restrict dangerous operations
- local-only config is excluded from version control

## T010 Specification Lock
- [x] Add `SPEC.md`
- [x] Review objective, scope, and non-goals
- [x] Ensure directory contract is consistent

Acceptance:
- system behavior is defined before implementation starts

## T020 Task Planning
- [x] Add `TASKS.md`
- [x] Break work into modules with acceptance criteria
- [x] Mark dependencies between tasks

Acceptance:
- next implementation step is always unambiguous

## T100 Schemas
- [x] Create `src/schemas.py`
- [x] Define typed objects for evidence, graph edges, assumptions, experiment metadata
- [x] Add tests for validation behavior

Acceptance:
- schemas are type-checkable
- invalid examples fail predictably

## T200 Bayesian Engine
- [x] Add `configs/priors.yaml`
- [x] Implement update engine
- [x] Add posterior summary outputs
- [x] Write tests for prior -> posterior movement

Acceptance:
- engine updates beliefs from evidence
- tests cover nominal and edge cases

## T300 Dependency / Skill Graph
- [x] Implement basis distribution logic
- [x] Implement dependency concentration metric
- [x] Implement portfolio score
- [x] Add tests

Acceptance:
- graph metrics are deterministic
- score behavior is interpretable on toy examples

## T400 Valuation / Scenario Module
- [x] Implement baseline valuation model
- [x] Add scenario assumptions
- [x] Add sensitivity analysis
- [ ] Save summary outputs to `reports/`  ← T600 report pipeline で対応

Acceptance:
- scenario assumptions are explicit
- outputs are reproducible from code/config

## T500 Experiment Registry
- [x] Define experiment folder template
- [x] Create `experiments/registry.md`
- [x] Require per-run metadata and summary

Acceptance:
- each run has an ID and reproducible footprint

## T600 Report Pipeline
- [x] Implement report generation entry point
- [x] Save charts, tables, and summary markdown
- [x] Link reports to experiment IDs

Acceptance:
- one command can generate a report artifact set

## T700 Review Workflow
- [x] Add review checklist in `notes/` or `docs/`
- [x] Verify test, lint, type-check sequence
- [x] Add security and secret-handling checks

Acceptance:
- each merge-ready change can be reviewed consistently

## T800 Digital Twin Engine
Dependencies: T200 (Bayesian), T400 (Valuation)

- [x] Define `DigitalTwinState` schema in `src/schemas.py`
  - market state vector
  - model parameter snapshot
  - timestamp and experiment ID
- [x] Implement `src/twin/simulator.py`
  - state transition function (linear Gaussian, local linear trend)
  - Monte Carlo forward simulation (vectorised, N samples)
- [x] Implement `src/twin/calibrator.py`
  - parameter estimation from observed data (Normal-Normal conjugate)
  - prior / posterior linkage with Bayesian engine
- [x] Add configs: `configs/twin_defaults.yaml`
  - simulation horizon, random seed, state dimension, process noise
- [x] Write tests: `tests/test_twin.py` (33 tests, all pass)
  - determinism under fixed seed
  - state dimension consistency
  - calibration reduces posterior variance

Acceptance:
- [x] forward simulation is reproducible from config + seed
- [x] calibrated parameters update the Bayesian posterior
- [ ] outputs saved to `experiments/exp-xxx/` per run  ← T600 report連携で対応

---

## T900 Exit Strategy Engine
Dependencies: T400 (Valuation), T800 (Digital Twin)

- [x] Define `ExitOption` / `ExitValueSummary` / `TimingDistribution` schemas
  - exit type: IPO / M&A / secondary / wind-down
  - timing distribution (earliest, expected, latest)
  - value estimate per scenario
- [x] Implement `src/exit/option_pricer.py`
  - expected exit value under each scenario
  - option-style payoff: max(V - floor, 0)
  - discount to present value
  - central-difference sensitivity: ∂EV/∂r, ∂EV/∂t, ∂EV/∂floor
- [x] Implement `src/exit/timing_map.py`
  - triangular distribution over [earliest, latest]
  - probability-weighted exit timing distribution (normalised)
  - timing-adjusted EV: Σ_k P(T=t_k) · EV_payoff / (1+r)^{t_k}
  - sensitivity: ∂EV_timing/∂r, ∂EV_timing/∂t_mode
- [x] Write tests: `tests/test_exit.py` (46 tests, all pass)
  - zero-value floor is respected
  - timing map integrates to 1.0
  - sensitivity direction matches economic intuition

Acceptance:
- [x] exit options are quantitatively comparable across types
- [x] timing map is normalized probability distribution
- [ ] sensitivity surface is saved to `reports/`  ← report pipeline で対応

---

## T1000 Entropy Layer
Dependencies: T200 (Bayesian), T400 (Valuation)

- [x] Implement `src/entropy/monitor.py`
  - Shannon entropy of posterior distribution (Normal and Beta closed-form)
  - KL divergence from prior to posterior (Normal and Beta)
  - entropy rate over time (rolling window, first differences)
- [x] Implement `src/entropy/detector.py`
  - distribution shift detection (KL threshold)
  - structural break detection (entropy gradient)
  - regime change alert generation (`run_detection`, `save_entropy_report`)
- [x] Add configs: `configs/entropy_thresholds.yaml`
  - KL divergence alert threshold
  - entropy gradient alert threshold
  - rolling window length
- [x] Write tests: `tests/test_entropy.py` (59 tests, all pass)
  - uniform distribution maximizes entropy (Beta(1,1))
  - concentrated posterior has lower entropy than uniform
  - alert fires above threshold, silent below
- [x] Integrate alert output into report pipeline (T600)
  - save entropy alert report to `reports/` via `save_entropy_report`

Acceptance:
- [x] Shannon entropy is computed from posterior state
- [x] alerts are triggered by thresholds defined in config
- [x] entropy alert report is generated per experiment run

---

---

## T1300 GNSS Signal-Level Spoofing Detection (Monte Carlo)
Dependencies: T100 (Schemas)

- [x] Implement `src/gnss/spoof_sim.py`
  - Doppler deviation simulation (genuine vs. meaconing attack)
  - Similarity graph: `w_{ij} = exp(−|Δf_i − Δf_j|² / σ²)`
  - All-forests count `m(t) = det(I + L_w)` (cycle matroid)
  - Chi-squared Doppler test `χ(t) = Σ(Δf_i − mean)² / σ_D²`
  - Fiedler-value greedy subset selection (subset size k)
  - Fisher-combined detection score `T = rᵀ diag(w_S) r`
  - ROC curve, AUC, detection delay, PVT degradation statistics
- [x] Add `MCSimReport` / `RunResult` to `src/schemas.py`
- [x] Expose `POST /gnss/spoof-sim` in `src/api/routers/gnss.py`
- [x] Write `tests/test_gnss_spoof_sim.py` (67 tests, all pass)

Acceptance:
- [x] MC simulation is deterministic under fixed seed
- [x] ROC/AUC and detection delay statistics are returned
- [x] API endpoint validates parameters and returns `MCSimReport`

---

## T1350 Multi-Sensor GNSS Spoofing Detection (Monte Carlo)
Dependencies: T1300

- [x] Implement `src/gnss/multi_sensor_sim.py`
  - 4-sensor fusion: pseudorange (PR), Doppler, AoA, INS residuals
  - Gradual meaconing attack model with ramp capture: `α = min(1, (t−t₀+1)/L)`
  - Lorentz AoA deviation detection
  - Percolation graph for sensor connectivity
  - Weighted detection score `s = w₁·m + w₂·clip(χ/χ₀) + w₃·clip(lor_dev)`
- [x] Add `MSSimReport` / `MSRunResult` to `src/schemas.py`
- [x] Expose `POST /gnss/multi-sensor-sim` in `src/api/routers/gnss.py`
- [x] Write `tests/test_gnss_multi_sensor_sim.py` (42 tests, all pass)

Acceptance:
- [x] gradual carry-off attack is distinguishable from nominal
- [x] ROC curve and per-run traces are returned

---

## T1400 ModelForge Governance
Dependencies: T100 (Schemas)

- [x] Implement `src/models/` (forge, trace, verifier, automation, __main__)
- [x] Add model registry YAML at `configs/model_registry/<id>.yaml`
- [x] Implement traceability chain: YAML → VerificationReport → impl_skeleton → TraceNode → AuditEntry
- [x] Expose `POST /forge/*` in `src/api/routers/forge.py`
- [x] Slash commands: `/modelforge-run`, `/modelforge-verify`, `/modelforge-trace`, `/modelforge-audit`, `/modelforge-report`
- [x] Audit log at `.claude/audit/modelforge.jsonl` (append-only)
- [x] Write `tests/test_modelforge.py` (60 tests, all pass)

Acceptance:
- [x] no model exists in code without a verified YAML spec
- [x] trace.jsonl is append-only and idempotent
- [x] all forge events are auditable

---

## T1500 GNSS Resilience Twin — 10-Layer Fault Discrimination
Dependencies: T1300, T1350, T200 (Bayesian), T1000 (Entropy)

### Phase A. Authentication Pillar

- [x] Implement `src/gnss/core.py` — OSNMA/TESLA verification engine
  - `TESLAKeyChain`: SHA-256 hash chain key derivation
  - `OSNMAAuthority` / `OSNMAReceiver`: sign and verify NAV messages
  - MAC tag verification + receipt safety (timing-based replay guard)
  - 4 attack types: naive_replay / modified_replay / key_disclosure / late_injection
  - Quantum fidelity layer (key_compromise detection)
- [x] Layer 10 `OSNMALayer` in `src/gnss/resilience_twin.py`
  - `auth_fraction` = authenticated satellites / total
  - posterior update: `p_spoof ∝ (1 − auth_fraction)`
- [x] Expose `POST /gnss/simulate`, `POST /gnss/verify-key`, `POST /gnss/detect`
- [x] Add `ObservationEpoch` / `OSNMALayerResult` / `AuthenticationScore` schemas

### Phase B. Integrity Pillar

- [x] Layer 1 `GMMRaim` — 2-component GMM per satellite
  - Gaussian mixture: `p(Δf_i | nominal)` vs `p(Δf_i | fault)`
  - Elevation-adjusted noise; `γ_i` fault posterior per satellite
  - Sign correlation + elevation pattern detection
- [x] Layer 2 `IMMKalman` — 3-mode Interacting Multiple Model Kalman filter
  - Modes: nominal / multipath / spoofing
  - Mode weights `μ = [μ_nom, μ_mp, μ_spoof]`; PVT re-estimation under each mode
- [x] Layer 5 `INSCouplingLayer` — INS velocity chi² test
  - `chi2_vel = ||Δv||² / σ_INS²`; alert when `chi2_vel > χ²_{crit}`
  - Graceful fallback when INS unavailable
- [x] Layer 6 `CoopRAIMLayer` — cooperative parity RAIM
  - Parity vector `p = (I − H H†) z`; chi² test on `||p||²`
- [x] Layer 9 `HuhSubsetSelector` — log-concavity + matroid subset selection
  - D-optimal design criterion; log-concavity ratio
  - Fault-flagged satellites excluded from PVT

### Phase C. Structural Pillar

- [x] Layer 3 `SpectralMonitor` — Laplacian Fiedler ratio + RMT anomaly
  - Fiedler value `λ₂(L)` and ratio `ρ_F = λ₂ / λ_n`
  - Marchenko-Pastur bulk edge comparison (RMT)
- [x] Layer 7 `StructuralDependencyMonitor` — Fiedler streak tracking
  - Consecutive epochs with Fiedler anomaly → structural alert
- [x] Layer 10 `DuminilCopinPhaseMonitor` — percolation phase transition
  - Edge weight threshold sweep; susceptibility peak detection
  - `percolation_threshold`, `susceptibility_peak`, `phase_alert`

### Phase D. Entropy / Decision Pillar

- [x] Layer 4 `FaultEntropyMonitor` — Shannon entropy + KL divergence on 4-class posterior
  - `H = −Σ p_k log p_k`; KL from uniform; alert when `H > θ_H`

### Phase E. MVP Pipeline & Action Engine

- [x] Implement `src/gnss/mvp.py`
  - `ReceiverAgent`: ingests `ObservationEpoch`, converts to internal arrays
  - `TwinCore`: wires all 10 layers; returns `EpochDiagnosis`
  - `ActionPlanner`: maps `EpochDiagnosis` → `_ActionPlan`
  - `MVPPipeline`: per-epoch loop + history accumulation
- [x] Implement `src/gnss/action_engine.py`
  - `SatelliteScorer`: per-satellite trust weight computation
  - `FailsafeManager`: NOMINAL / DEGRADED / INS_ONLY / DEAD_RECKONING
  - `AlertBuilder`: severity levels (info / caution / warning / critical)
- [x] Implement `src/gnss/edge_collector.py`
  - `EdgeCollector`: accumulate per-epoch signals into `EdgeSnapshot`
  - `EdgeArrays`: stacked NumPy arrays ready for plotting / export
- [x] Add `ResilienceTwinConfig` / `ResilienceTwinReport` / `TwinRunReport` schemas
- [x] Expose `POST /gnss/resilience-sim`, `POST /gnss/twin/run`

### Phase F. Diagnostic Report

- [x] Implement `src/gnss/report.py` — `plot_gnss_report`
  - Panel 1: fault_posterior time series + alert event markers
  - Panel 2: Doppler residual heatmap (epochs × sats)
  - Panel 3: IMM mode weights + Shannon entropy + alert markers
  - Panel 4: satellite weight heatmap + n_active / n_excluded step plot
  - Panel 5: confidence + INS weight + failsafe bands + mc_auc scatter

### Tests

- [x] `tests/test_gnss_resilience_twin.py` (37 tests, all pass)
- [x] `tests/test_gnss_mvp.py` (35 tests, all pass)
- [x] `tests/test_gnss_action_engine.py` (58 tests, all pass)
- [x] `tests/test_gnss_edge_collector.py` (43 tests, all pass)

Acceptance:
- [x] 4-class confusion matrix is computed over MC trials
- [x] per-epoch fault posterior sums to 1.0
- [x] `POST /gnss/twin/run` returns per-epoch `EpochReport` with recommended action
- [x] 5-panel diagnostic figure is generated from `EdgeArrays`

---

## T1600 Yield Twin — GP Surrogate + D-Optimal Design
Dependencies: T100 (Schemas)

- [x] Implement `src/yield_twin/gp_surrogate.py`
  - GP with ARD RBF kernel; log-marginal-likelihood optimisation
  - D-optimal design: maximise `det(K_XX)` for next experiment point
  - Expected Improvement (EI) acquisition function
  - LOO-R² cross-validation
- [x] Add `YieldTwinReport` / `FactorSpec` schemas
- [x] Expose `/yield-twin/*` API router
- [x] Write `tests/test_yield_twin.py`

Acceptance:
- [x] GP predictions are reproducible under fixed hyperparameters
- [x] EI acquisition returns valid next experiment point

---

## T1700 Strategy Twin
Dependencies: T200 (Bayesian), T800 (Digital Twin), T1000 (Entropy)

- [x] Implement `src/strategy_twin/`
- [x] Add `StrategyTwinReport` schema
- [x] Expose `POST /strategy/run`
- [x] Write `tests/test_strategy_twin.py`

Acceptance:
- [x] causal effects and macro environment are returned
- [x] verdict is derived from Bayesian posterior

---

## Suggested Execution Order
1. T000
2. T001
3. T010
4. T020
5. T100
6. T200
7. T300
8. T400
9. T500
10. T600
11. T700
12. T800 (Digital Twin Engine)
13. T900 (Exit Strategy Engine)  ← requires T800
14. T1000 (Entropy Layer)
15. T1300 (GNSS Signal-Level Spoofing MC)
16. T1350 (Multi-Sensor MC)  ← requires T1300
17. T1400 (ModelForge Governance)
18. T1500 (GNSS Resilience Twin)  ← requires T1300, T1350
19. T1600 (Yield Twin)
20. T1700 (Strategy Twin)

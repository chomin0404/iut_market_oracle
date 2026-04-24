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

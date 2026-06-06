# Experiment Registry

| ID | Title | Created | Tags | Summary |
|----|-------|---------|------|---------|
| exp-001 | DCF report — 2026-04-18 | 2026-04-18 09:01 | dcf, report, automated | Automated report: 3 scenarios. Base EV = 6,942 JPY millions. |
| exp-002 | DCF report — 2026-04-29 | 2026-04-29 12:20 | dcf, report, automated | Automated report: 3 scenarios. Base EV = 6,942 JPY millions. |
| T1700 | Strategy Twin — 2026-05-01 | 2026-05-01 | strategy, sotp, black-litterman, causal, viability | SOTP moat-adjusted DCF + Black-Litterman posterior + reverse-DCF viability (g*, s*, m*) + linear SCM causal ATE. FastAPI POST /strategy/run. 44 tests green. |
| T1400 | ModelForge Unification — 2026-05-01 | 2026-05-01 | modelforge, registry, traceability, verification, automation | YAML spec → static verify (7 checks) → deterministic skeleton codegen → TraceNode DAG → audit JSONL. FastAPI /forge/* (5 endpoints). Cloud Functions automation. 60 tests green. |
| exp-003 | DCF report — 2026-05-06 | 2026-05-06 00:24 | dcf, report, automated | Automated report: 3 scenarios. Base EV = 6,942 JPY millions. |
| exp-004 | DCF report — 2026-05-11 | 2026-05-11 11:41 | dcf, report, automated | Automated report: 3 scenarios. Base EV = 6,942 JPY millions. |
| exp-005 | DCF report — 2026-05-16 | 2026-05-16 12:55 | dcf, report, automated | Automated report: 3 scenarios. Base EV = 6,942 JPY millions. |
| exp-006 | DCF report — 2026-05-20 | 2026-05-20 10:34 | dcf, report, automated | Automated report: 3 scenarios. Base EV = 6,942 JPY millions. |
| exp-007 | DCF report — 2026-05-24 | 2026-05-24 01:38 | dcf, report, automated | Automated report: 3 scenarios. Base EV = 6,942 JPY millions. |
| exp-008 | DCF report — 2026-05-24 | 2026-05-24 02:18 | dcf, report, automated | Automated report: 3 scenarios. Base EV = 6,942 JPY millions. |
| exp-009 | DCF report — 2026-06-06 | 2026-06-06 02:26 | dcf, report, automated | Automated report: 3 scenarios. Base EV = 6,942 JPY millions. |
| T1500-mc | GNSS Resilience Twin MC — 2026-06-06 | 2026-06-06 | gnss, resilience-twin, montecarlo, spoofing, 4-class | 4-pillar GNSS fault discrimination MC benchmark (n=400, seed=42). Classes: NOMINAL/MULTIPATH/HARDWARE_FAULT/SPOOFING. P_FA≈1%, per-class accuracy ≥80%. CLI: `uv run python -m src.gnss --n-mc 400 --seed 42`. Output: output/resilience_report.json. |
| T1500-ml | GNSS IsolationForest Detector — 2026-06-06 | 2026-06-06 | gnss, ml, isolation-forest, spoofing-detection, far | Unsupervised IsolationForest spoofing detector trained on MC-generated Doppler features. FAR target ≤1e-4 via exact discrete calibration. CLI: `uv run python -m src.gnss ml train`. Model: output/if_detector.joblib. |
| T1500-lstm | GNSS LSTM Detector — 2026-06-06 | 2026-06-06 | gnss, ml, lstm, spoofing-detection, sliding-window | Supervised stacked LSTM (2-layer, hidden=64) over W=16 epoch sliding windows. BCEWithLogitsLoss, FAR calibration via Geyer ESS. Seed=42 for reproducibility. Defined in src/gnss/ml_detector.py. |
| T200-bayesian | Bayesian Update API — 2026-06-06 | 2026-06-06 | bayesian, mcmc, metropolis-hastings, hmc, ess | MH and HMC samplers with ESS (Geyer 1992 monotone positive sequence estimator). Multivariate normal target with discriminator union type. POST /bayesian/mcmc. 30 tests. |
| T200-network | Bayesian Network (Water Demand) — 2026-06-06 | 2026-06-06 | bayesian, network, dirichlet, water-demand, calibration | 5-node DAG for Fukuoka water demand prediction. Dirichlet posterior update via observe_batch + apply_dirichlet_posterior. Weekday/holiday high-demand delta=0.084. src/bayesian/water_demand_net.py. |

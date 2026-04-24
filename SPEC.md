# System Name
Research Operating System

## Objective
Build a reproducible research platform that integrates:
1. mathematical reasoning
2. Bayesian updating
3. experiment tracking
4. valuation / scenario analysis
5. report generation
6. Claude Code assisted implementation workflow

## Primary Users
- quantitative researcher
- mathematical modeler
- valuation analyst
- AI-assisted coding researcher

## Core Capabilities

### C1. Mathematical Note System
The system shall support structured research notes containing:
- definitions
- assumptions
- lemmas
- propositions
- proof sketches
- open questions
- failure modes

Acceptance criteria:
- notes are stored as markdown files in `notes/`
- notation changes are explicit
- claims can be tagged as proven / heuristic / empirical / todo

### C2. Bayesian Modeling
The system shall support prior specification, evidence updates, and posterior summaries.

Acceptance criteria:
- priors are stored in config files under `configs/`
- update logic is implemented in `src/`
- tests verify posterior movement under controlled inputs
- outputs are serializable for reports

### C3. Skill / Dependency Graph Analysis
The system shall support graph-based analysis of dependency concentration, basis diversity, and portfolio-style metrics.

Acceptance criteria:
- graph inputs are represented as typed schemas
- metrics are computed deterministically
- tests cover edge cases and simple benchmark cases

### C4. Empirical / Valuation Analysis
The system shall support valuation or scenario models using explicit assumptions and saved outputs.

Acceptance criteria:
- assumptions are stored in code or config, not hidden in notebooks
- charts and tables are written to `reports/`
- scenario runs are captured in `experiments/exp-xxx/`

### C5. Reproducible Experiments
The system shall support experiment-scoped folders with metadata, configs, outputs, and short narrative summaries.

Acceptance criteria:
- each experiment has a unique ID
- each experiment folder contains at least config, result, and note
- generated outputs are linked to the run that created them

### C6. Reviewable Engineering Workflow
The system shall support a spec-driven, task-driven, test-backed development workflow.

Acceptance criteria:
- `SPEC.md` defines expected system behavior
- `TASKS.md` tracks execution order and progress
- each meaningful implementation change updates tests or validation steps

### C7. Entropy Layer
The system shall monitor information-theoretic properties of model state and signal structural changes.

Techniques:
- Adjoint state method
- Jacobian sensitivity
- singular perturbation
- Hilbert space representation
- resonance analysis
- Voigt / Breit-Wigner line shape modeling

Capabilities:
- Shannon entropy monitoring
- distribution shift detection
- structural break detection
- regime change alerts

Acceptance criteria:
- entropy metrics are computed from posterior state
- alerts are triggered by configurable thresholds
- outputs are saved to `reports/` per experiment run

## 5. Product Scope
本プロジェクトは以下の5モジュールで構成する。

1. Skill Graph Engine
2. Bayesian Scenario Engine
3. Valuation Engine
4. Digital Twin Engine
5. Exit Strategy Engine

## 6. Input Data

### Required
- macro indicators
- market size assumptions
- business model descriptors
- moat indicators
- growth assumptions
- discount rate assumptions
- scenario priors
- project level observations

### Optional
- management quality scores
- competitor network data
- patent signals
- customer cohort data
- experimental design logs
- quantum chemistry outputs

## 7. Output
- 5-year skill roadmap
- posterior scenario distribution
- expected enterprise value range
- reverse DCF implied expectations
- exit timing map
- sensitivity surface
- entropy alert report
- recommended next experiments

## 8. Quantitative Defaults
- baseline growth rate: 10%
- baseline discount rate: 10%
- explicit forecast horizon: 5 years
- scenario count: 3 to 7
- posterior refresh cadence: monthly
- major model review cadence: quarterly

## 9. Optimization Goal
最適化対象は単一の期待値ではない。
以下の多目的最適化を行う。

- maximize learning velocity
- maximize robustness of decision basis
- maximize expected value creation
- minimize dependency concentration
- minimize model fragility
- minimize irreversible downside

## 10. Skill Basis Design
5年後に必要なスキルは次の5基底へ圧縮する。

1. Probabilistic inference
2. Valuation and capital allocation
3. Inverse problems and simulation
4. Optimal experiment design
5. Strategic implementation with Claude Code

各学習項目は必ずこの5基底のどこに属するかを明示する。

## 11. Acceptance Criteria
システムは次を満たすこと。

- 任意の学習項目が少なくとも1つの基底へ分類される
- 基底間の依存集中度を定量化できる
- DCF / Reverse DCF / SOTP の整合性を確認できる
- scenario posterior を更新できる
- sensitivity analysis を自動生成できる
- exit option を定量比較できる
- entropy based alert を出せる
- Claude Code で反復改善できる

## 12. Non-Goals
The system is not intended to:
- auto-prove theorems universally
- replace domain judgment
- infer undocumented assumptions
- mutate raw data sources
- produce publication claims without human review
- full production trading system
- fully automated investment execution
- complete quantum simulation stack
- enterprise ERP integration
- end-to-end GUI product

## 13. Risks
- モデル過剰複雑化
- prior の恣意性
- valuation と causal claim の混同
- 学習項目の過積載
- 依存構造の見かけ上の多様化
- Claude Code に対する曖昧指示

## 14. Design Rules
- one objective per module
- one measurable output per task
- uncertainty must be explicit
- all assumptions are versioned
- no hidden spreadsheet logic
- every important model has a failure mode section

## 15. Claude Code Operating Rules
- Claude Code must read this SPEC before editing core modules
- all implementation work must map to a task id
- every model change must update assumptions and tests
- every analysis must produce reproducible outputs
- avoid large undifferentiated files
- prefer small composable modules

## Directory Contract
- `src/` reusable implementation
- `tests/` verification
- `configs/` priors, scenarios, experiment settings
- `notes/` mathematical and conceptual notes
- `papers/` draft manuscripts
- `data/raw/` immutable source data
- `data/processed/` derived datasets
- `experiments/` run-scoped outputs
- `reports/` charts, tables, summaries
- `notebooks/` exploration only

## Quality Standards
- All core calculations must be testable.
- Reproducibility takes priority over convenience.
- Conclusions must remain proportional to evidence.
- Every reportable result should have a discoverable origin.

## Minimum Viable Release
The first usable release shall include:
1. repository structure
2. CLAUDE.md + settings
3. typed schemas
4. one Bayesian module
5. one graph-analysis module
6. one valuation or scenario module
7. tests
8. one report pipeline
